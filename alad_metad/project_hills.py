#!/usr/bin/env python3
# import numpy as np
import torch
import pandas as pd

def project_hills(hill_traj_filename, metad_energy, metad_centers, metad_sigma_square):
    traj = pd.read_csv(hill_traj_filename)
    hill_heights = traj['hill_height'].to_numpy()
    hill_centers = traj[['phi', 'psi']].to_numpy()
    for i in range(len(hill_heights)):
        pos = torch.tensor(hill_centers[i], device='cuda', dtype=torch.float)
        diff_tmp = pos - metad_centers
        diff = torch.where(diff_tmp > 180.0, diff_tmp - 360.0,
                           torch.where(diff_tmp < -180.0, diff_tmp + 360.0, diff_tmp))
        tmp2 = diff / (2.0 * metad_sigma_square)
        hill_energy = hill_heights[i] * torch.exp(-1.0 * torch.sum(tmp2 * diff, dim=-1))
        metad_energy += hill_energy
    return metad_energy


def write_pmf(metad_centers, metad_energy, bias_temperature, simulation_temperature, output_file):
    metad_energy *= -(bias_temperature + simulation_temperature) / bias_temperature
    with open(output_file, 'w') as output:
        for i in range(len(metad_centers)):
            for j in range(len(metad_centers[i])):
                centers = metad_centers[i][j]
                output.write(f'{float(centers[0].item()):10.3f} {float(centers[1].item()):10.3f} {float(metad_energy[i][j].item()):12.5e}\n')


if __name__ == '__main__':
    metad_sigma_square = torch.square(torch.tensor([6.0 * 5.0 / 2.0, 6.0 * 5.0 / 2.0], device='cuda', dtype=torch.float))
    metad_energy = torch.zeros((72, 72), device='cuda', dtype=torch.float)
    gx, gy = torch.meshgrid(torch.linspace(-177.5, 177.5, 72, device='cuda', dtype=torch.float),
                            torch.linspace(-177.5, 177.5, 72, device='cuda', dtype=torch.float), indexing='ij')
    metad_centers = torch.stack((gy, gx)).permute(dims=(2, 1, 0))
    metad_energy = project_hills('test-alad-pytorch.traj', metad_energy, metad_centers, metad_sigma_square)
    write_pmf(metad_centers, metad_energy, 3000.0, 300.0, 'pmf.dat')
