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
                torch.tensor(reference_positions, dtype=torch.float64, device='cuda'))
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
        def update_atoms(self):
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

        @torch.jit.export
        def run_calculate(self):
            if self.step % self.mts == 0:
                return True
            else:
                return False

        @torch.jit.export
        def calculate(self, position, total_force, mass, charge):
            # position.retain_grad()
            pos_centered = self.bring_to_center(position)
            matrix_F = self.build_matrix_F(pos_centered, self.ref_pos_centered)
            # print(position)
            # if self.step == 0:
            #     print(mass)
            w, v = torch.linalg.eigh(matrix_F)
            max_eig_val = w[-1]
            s = torch.sum(torch.square(pos_centered) + torch.square(self.ref_pos_centered))
            rmsd = torch.sqrt((s - 2.0 * max_eig_val) / self.num_atoms)
            rmsd.backward()
            applied_force = -1.0 * 0.01 * (self.cv - 0.0) * position.grad
            # print(position.grad)
            # Apply restraint at RMSD = 0
            self.energy_val = 0.5 * 0.01 * (self.cv - 0.0) * (self.cv - 0.0)
            self.cv = float(rmsd.item())
            # print(self.cv)
            return applied_force

        @torch.jit.export
        def energy(self):
            return self.energy_val

    m = torch.jit.script(RMSD(atom_positions, atom_serials, cv_title))
    torch.jit.save(m, 'RMSD.pt')
    return m


def main():
    atom_positions, atom_serials = read_reference_pdb('stmv_ref.pdb')
    model = build_rmsd_cv(atom_positions, atom_serials)
    # Perform a few tests
    input_pos = torch.tensor(atom_positions, dtype=torch.float64, device='cuda', requires_grad=True)
    f = model.calculate(input_pos, None, None, None)
    print(model.cv)
    print(model.output_lines())


if __name__ == '__main__':
    main()
