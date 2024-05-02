#!/usr/bin/env python3
import MDAnalysis as mda
import torch
import torch.nn as nn
import numpy as np


def read_reference_pdb(pdb_filename: str):
    u = mda.Universe(pdb_filename)
    sel = u.atoms[u.atoms.tempfactors > 0]
    print(f'Selected {sel.n_atoms} by MDAnalysis')
    serials = sel.indices
    return np.float64(sel.positions.T), serials.tolist()


def build_rmsd_cv(atom_positions, atom_serials):

    cv_title = 'step,rmsd,bias_energy\n'

    class RMSD(nn.Module):
        def __init__(self, reference_positions, atom_serials, traj_title):
            super().__init__()
            self.ref_pos_centered = self.bring_to_center(
                torch.tensor(reference_positions, dtype=torch.float, device='cuda'))
            self.ref_pos_centered_sq = torch.square(self.ref_pos_centered)
            self.num_atoms = len(atom_serials)
            self.atom_serials = atom_serials
            self.step = 0
            self.cv = 0.0
            self.mts = 1
            self.energy_val = 0.0
            self.traj_title = traj_title

        @torch.jit.export
        def request_atoms(self):
            return self.atom_serials

        @torch.jit.export
        def request_pos_grads(self):
            return False

        @torch.jit.export
        def set_step(self, step: int):
            self.step = step

        @torch.jit.export
        def total_force_on(self):
            return True

        @torch.jit.export
        def output_filenames(self):
            return ['test-stmv-pytorch.traj']

        @torch.jit.export
        def output_frequencies(self):
            return [self.mts]

        @torch.jit.export
        def output_lines(self):
            s = ''
            if self.step == 0:
                s += self.traj_title
            s += '%d,%e,%e\n' % (self.step, self.cv, self.energy_val)
            #s += f'{self.step:10d}'
            # print(s)
            return [s]

        @torch.jit.export
        def update_positions(self):
            if self.step % self.mts == 0:
                return True
            else:
                return False

        @torch.jit.export
        def update_mass(self):
            if self.step == 0:
                return True
            else:
                return False

        @torch.jit.export
        def update_charge(self):
            if self.step == 0:
                return True
            else:
                return False

        def bring_to_center(self, data):
            center_of_geometry = data.mean(dim=1, keepdim=True)
            return data - center_of_geometry

        def build_matrix_F(self, pos_target, pos_reference):
            mat_R = torch.matmul(pos_target, pos_reference.T)
            F00 = mat_R[0][0] + mat_R[1][1] + mat_R[2][2]
            F01 = mat_R[1][2] - mat_R[2][1]
            F02 = mat_R[2][0] - mat_R[0][2]
            F03 = mat_R[0][1] - mat_R[1][0]
            F10 = F01
            F11 = mat_R[0][0] - mat_R[1][1] - mat_R[2][2]
            F12 = mat_R[0][1] + mat_R[1][0]
            F13 = mat_R[0][2] + mat_R[2][0]
            F20 = F02
            F21 = F12
            F22 = -mat_R[0][0] + mat_R[1][1] - mat_R[2][2]
            F23 = mat_R[1][2] + mat_R[2][1]
            F30 = F03
            F31 = F13
            F32 = F23
            F33 = -mat_R[0][0] - mat_R[1][1] + mat_R[2][2]
            row0 = torch.stack((F00, F01, F02, F03))
            row1 = torch.stack((F10, F11, F12, F13))
            row2 = torch.stack((F20, F21, F22, F23))
            row3 = torch.stack((F30, F31, F32, F33))
            F = torch.stack((row0, row1, row2, row3))
            return F

        def buil_rotation_matrix(self, v):
            q = v[:, -1]
            R00 = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
            R01 = 2.0 * (q[1] * q[2] - q[0] * q[3])
            R02 = 2.0 * (q[1] * q[3] + q[0] * q[2])
            R10 = 2.0 * (q[1] * q[2] + q[0] * q[3])
            R11 = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
            R12 = 2.0 * (q[2] * q[3] - q[0] * q[1])
            R20 = 2.0 * (q[1] * q[3] - q[0] * q[2])
            R21 = 2.0 * (q[2] * q[3] + q[0] * q[1])
            R22 = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
            row0 = torch.stack((R00, R01, R02))
            row1 = torch.stack((R10, R11, R12))
            row2 = torch.stack((R20, R21, R22))
            R = torch.stack((row0, row1, row2))
            return R.T

        @torch.jit.export
        def run_calculate(self):
            if self.step % self.mts == 0:
                return True
            else:
                return False

        @torch.jit.export
        def calculate(self, position, total_force, mass, charge, lattice):
            position_float = position.to(torch.float32)
            pos_centered = self.bring_to_center(position_float)
            matrix_F = self.build_matrix_F(pos_centered, self.ref_pos_centered)
            w, v = torch.linalg.eigh(matrix_F)
            rotation_matrix = self.buil_rotation_matrix(v)
            ref_pos_rotated = rotation_matrix @ self.ref_pos_centered
            pos_diff = pos_centered - ref_pos_rotated
            # WARNING: w[-1] is numerically instable!!
            rmsd = torch.sqrt(torch.sum(torch.square(pos_diff)) / self.num_atoms)
            # Manual derivative calcuation seems faster than rmsd.backward()
            pos_grad_factor = torch.nan_to_num(1.0 / (self.num_atoms * rmsd))
            pos_grad = pos_grad_factor * pos_diff
            applied_force = -1.0 * 0.01 * (self.cv - 0.0) * pos_grad
            # Apply restraint at RMSD = 0
            self.energy_val = 0.5 * 0.01 * (self.cv - 0.0) * (self.cv - 0.0)
            self.cv = float(rmsd.item())
            # print(self.cv)
            # position.grad.zero_()
            return applied_force.to(torch.float64)

        @torch.jit.export
        def energy(self):
            return self.energy_val

    m = torch.jit.script(RMSD(atom_positions, atom_serials, cv_title))
    torch.jit.save(m, 'RMSD_opt.pt')
    return m


def main():
    atom_positions, atom_serials = read_reference_pdb('stmv_ref.pdb')
    model = build_rmsd_cv(atom_positions, atom_serials)
    # Perform a few tests
    input_pos = torch.tensor(atom_positions, dtype=torch.float64, device='cuda', requires_grad=False)
    f = model.calculate(input_pos, None, None, None, None)
    print(model.cv)
    print(model.output_lines())
    print(f)


if __name__ == '__main__':
    main()
