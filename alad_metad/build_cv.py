#!/usr/bin/env python3
import torch
import torch.nn as nn
# from pprint import pprint
from typing import Final


class alad_metad(nn.Module):

    bias_temperature: Final[float]
    ts: Final[int]
    new_hill_freq: Final[float]
    hill_initial_height: Final[float]

    def __init__(self):
        super().__init__()
        self.step = 0
        # Request atoms
        self.atom_serials = list(map(lambda x: x-1, [5, 7, 9, 15, 17]))
        self.ts = 1
        self.cv_phi = 0.0
        self.cv_psi = 0.0
        self.energy_val = 0.0
        # MetaD parameters
        self.new_hill_freq = 1000
        self.hill_initial_height = 1.0
        self.hill_height = 0.0
        self.hill_sigmas_square = torch.square(torch.tensor([6.0 * 5.0 / 2.0, 6.0 * 5.0 / 2.0], device='cuda', dtype=torch.float))
        self.bias_temperature = 3000
        # Create a MetaD grid
        self.metad_energy = torch.zeros((72, 72), device='cuda', dtype=torch.float)
        self.metad_grad = torch.zeros((72, 72, 2), device='cuda', dtype=torch.float)
        gx, gy = torch.meshgrid(torch.linspace(-177.5, 177.5, 72, device='cuda', dtype=torch.float),
                                torch.linspace(-177.5, 177.5, 72, device='cuda', dtype=torch.float), indexing='ij')
        # Grid centers
        self.metad_centers = torch.stack((gy, gx)).permute(dims=(2, 1, 0))
        # self.training = False

    @torch.jit.export
    def request_atoms(self):
        # print(self.atom_serials)
        return self.atom_serials

    @torch.jit.export
    def request_pos_grads(self):
        return False

    @torch.jit.export
    def set_step(self, step: int):
        self.step = step

    @torch.jit.export
    def total_force_on(self):
        return False

    @torch.jit.export
    def output_filenames(self):
        return ['test-alad-pytorch.traj']

    @torch.jit.export
    def output_lines(self):
        s = ''
        if self.step == 0:
            s += "step,phi,psi,energy,hill_height\n"
        s += '%d,%e,%e,%e,%e\n' % (self.step, self.cv_phi, self.cv_psi, self.energy_val, self.hill_height)
        return [s]

    @torch.jit.export
    def output_frequencies(self):
        return [self.ts * self.new_hill_freq]

    @torch.jit.export
    def update_positions(self):
        if self.step % self.ts == 0:
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

    @torch.jit.export
    def update_lattice(self):
        if self.step == 0:
            return True
        else:
            return False

    @torch.jit.export
    def run_calculate(self):
        if self.step % self.ts == 0:
            return True
        else:
            return False

    def forward(self, x):
        return x

    def dihedral(self, r12, r23, r34):
        A = torch.linalg.cross(r12, r23)
        B = torch.linalg.cross(r23, r34)
        cos_alpha = torch.dot(A, B)
        sin_alpha = torch.dot(A, r34) * torch.linalg.norm(r23)
        alpha = torch.rad2deg(torch.arctan2(sin_alpha, cos_alpha))
        # Derivative (code swiped from Colvars)
        rA = torch.linalg.norm(A)
        rB = torch.linalg.norm(B)
        C = torch.linalg.cross(r23, A)
        rC = torch.linalg.norm(C)
        cos_phi = torch.dot(A, B) / (rA * rB)
        sin_phi = torch.dot(C, B) / (rC * rB)
        f = torch.zeros((3, 3), dtype=torch.float, device='cuda')
        rB = 1.0 / rB
        B *= rB
        if torch.abs(sin_phi) > 0.1:
            rA = 1.0 / rA
            A *= rA
            dcosdA = rA * (cos_phi * A - B)
            dcosdB = rB * (cos_phi * B - A)
            K = (1.0 / sin_phi) * (180.0 / torch.pi)
            f[0] += K * torch.linalg.cross(r23, dcosdA)
            f[2] += K * torch.linalg.cross(dcosdB, r23)
            f[1] += K * (torch.linalg.cross(dcosdA, r12) + torch.linalg.cross(r34, dcosdB))
        else:
            rC = 1.0 / rC
            C *= rC
            dsindC = rC * (sin_phi * C - B)
            dsindB = rB * (sin_phi * B - C)
            K = -(1.0 / cos_phi) * (180.0 / torch.pi)
            f[0][0] = K * ((r23[1] * r23[1] + r23[2] * r23[2]) * dsindC[0]
                           - r23[0] * r23[1] * dsindC[1]
                           - r23[0] * r23[2] * dsindC[2])
            f[0][1] = K * ((r23[2] * r23[2] + r23[0] * r23[0]) * dsindC[1]
                           - r23[1] * r23[2] * dsindC[2]
                           - r23[1] * r23[0] * dsindC[0])
            f[0][2] = K * ((r23[0] * r23[0] + r23[1] * r23[1]) * dsindC[2]
                           - r23[2] * r23[0] * dsindC[0]
                           - r23[2] * r23[1] * dsindC[1])
            f[2] += K * torch.linalg.cross(dsindB, r23)
            f[1][0] = K*(-(r23[1]*r12[1] + r23[2]*r12[2])*dsindC[0]
                         + (2.0*r23[0]*r12[1] - r12[0]*r23[1])*dsindC[1]
                         + (2.0*r23[0]*r12[2] - r12[0]*r23[2])*dsindC[2]
                         + dsindB[2]*r34[1] - dsindB[1]*r34[2])
            f[1][1] = K*(-(r23[2]*r12[2] + r23[0]*r12[0])*dsindC[1]
                         + (2.0*r23[1]*r12[2] - r12[1]*r23[2])*dsindC[2]
                         + (2.0*r23[1]*r12[0] - r12[1]*r23[0])*dsindC[0]
                         + dsindB[0]*r34[2] - dsindB[2]*r34[0])
            f[1][2] = K*(-(r23[0]*r12[0] + r23[1]*r12[1])*dsindC[2]
                         + (2.0*r23[2]*r12[0] - r12[2]*r23[0])*dsindC[0]
                         + (2.0*r23[2]*r12[1] - r12[2]*r23[1])*dsindC[1]
                         + dsindB[1]*r34[0] - dsindB[0]*r34[1])
        grad = torch.stack((-f[0], -f[1] + f[0], -f[2] + f[1], f[2]))
        return alpha, grad

    def wrap_dist(self, dist_vec, unit_cells, reciprocal_cell):
        # print(reciprocal_cell)
        shifts = torch.floor(torch.sum(reciprocal_cell * dist_vec, dim=-1) + 0.5)
        return dist_vec - unit_cells.T @ shifts

    @torch.jit.export
    def calculate(self, position, total_force, mass, charge, lattice):
        unit_cells = lattice[0:3].to(torch.float).detach()
        v = torch.linalg.cross(unit_cells[[1, 2, 0]], unit_cells[[2, 0, 1]])
        reciprocal_cell = v / torch.sum(v * unit_cells, dim=-1)
        # print(lattice)
        position_float = position.to(torch.float).detach()
        # position_float.requires_grad_()
        vecs = torch.diff(position_float.T, dim=0)
        r12 = self.wrap_dist(vecs[0], unit_cells, reciprocal_cell)
        r23 = self.wrap_dist(vecs[1], unit_cells, reciprocal_cell)
        r34 = self.wrap_dist(vecs[2], unit_cells, reciprocal_cell)
        r45 = self.wrap_dist(vecs[3], unit_cells, reciprocal_cell)
        phi, phi_grad = self.dihedral(r12, r23, r34)
        psi, psi_grad = self.dihedral(r23, r34, r45)
        self.cv_phi = float(phi.item())
        self.cv_psi = float(psi.item())
        pos = torch.stack((phi, psi))
        # Calculate the bin indicies
        i_phi = torch.floor((self.cv_phi - (-180.0)) / 5.0)
        i_psi = torch.floor((self.cv_psi - (-180.0)) / 5.0)
        if i_phi > 71:
            i_phi = 71
        if i_psi > 71:
            i_psi = 71
        # Project a new hill
        if self.step > 0 and self.step % self.new_hill_freq == 0:
            wt_factor = torch.exp(-1.0 * self.metad_energy[i_phi][i_psi] / (0.0019872041 * self.bias_temperature))
            diff_tmp = (pos - self.metad_centers).detach()
            # Wrap the dihedrals
            diff = torch.where(diff_tmp > 180.0, diff_tmp - 360.0,
                               torch.where(diff_tmp < -180.0, diff_tmp + 360.0, diff_tmp))
            # print(diff)
            tmp2 = diff / (2.0 * self.hill_sigmas_square)
            # print(tmp2.permute((2, 0, 1))[0])
            # Energy
            self.hill_height = float(wt_factor.item()) * self.hill_initial_height
            hill_energy = self.hill_height * torch.exp(-1.0 * torch.sum(tmp2 * diff, dim=-1))
            self.metad_energy += hill_energy
            # Gradient of the MetaD potential
            # Flip the sign here because we need the gradient wrt the hill centers
            self.metad_grad += 2.0 * hill_energy.reshape((72, 72, 1)) * tmp2
        self.energy_val = float(self.metad_energy[i_phi][i_psi].item())
        applied_force = torch.zeros(position_float.shape, device=position.device, dtype=position.dtype)
        grad = -self.metad_grad[i_phi][i_psi]
        gphi = grad[0] * phi_grad
        gpsi = grad[1] * psi_grad
        # WARNING: The following operations are very slow and I don't know why!
        applied_force[:, 0:4] += gphi.T
        applied_force[:, 1:5] += gpsi.T
        # WARNING: backward() is slow so I avoid it
        # s = torch.sum(self.metad_grad[i_phi][i_psi] * pos)
        # s.backward()
        # position.grad.zero_()
        return applied_force

    @torch.jit.export
    def energy(self):
        return self.energy_val


if __name__ == '__main__':
    m = alad_metad()
    m = torch.jit.script(m)
    # No performance gain
    m = torch.jit.optimize_for_inference(m, [
        'request_atoms', 'request_pos_grads', 'calculate',
        'set_step', 'total_force_on', 'output_filenames',
        'output_lines', 'output_frequencies', 'update_positions',
        'update_mass', 'update_charge', 'update_lattice',
        'run_calculate', 'energy', 'wrap_dist',
        'dihedral'])
    m.step = 1000
    torch.jit.save(m, 'alad_metad.pt')
    # Perform some tests
    input_pos = torch.tensor([[3.51160389171208e+00,  1.70574483578321e+01, -6.30392765911769e+00],
                              [3.64258449902727e+00,  1.60489063177129e+01, -7.23230662747935e+00],
                              [3.48108575285343e+00,  1.46989520160925e+01, -6.92648031292247e+00],
                              [4.72147728727491e+00,  1.40143873284908e+01, -6.28288296566333e+00],
                              [5.89441656924542e+00,  1.42071794170370e+01, -6.88735797287428e+00]],
                             dtype=torch.float, device='cuda')
    input_lattice = torch.tensor([[27.91555256,  0.        ,  0.        ],
                                  [ 0.        , 27.82781972,  0.        ],
                                  [ 0.        ,  0.        , 27.82492717],
                                  [-0.11176319, -0.12706061,  0.04258038]], dtype=torch.float, device='cuda')
    input_pos_T = input_pos.T.detach()
    input_pos_T.requires_grad_()
    f = m.calculate(input_pos_T, None, None, None, input_lattice)
    print(m.cv_phi)
    print(m.cv_psi)
    m.step = 1001
    print(f)
    f = m.calculate(input_pos_T, None, None, None, input_lattice)
    print(f)
    print(m.energy_val)
    print(m)
    # pprint(vars(_m))
