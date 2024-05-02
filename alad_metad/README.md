Steps for running the NAMD with PytorchForces

1. Compile the NAMD with PytorchForces and CudaGlobalMaster branch;
2. Run `python3 ./build_cv.py` to compile the RMSD CV calculation into a torchscript archive (`alad_metad.pt`);
3. Run `namd3 +p1 ./alad-test.namd`
4. After simulation, run `python3 ./project_hills.py` to sum the hills from trajectory, and save the PMF to `pmf.dat`
5. Run `./draw_fes_2D.py pmf.dat -o pmf.png` to plot the PMF.
