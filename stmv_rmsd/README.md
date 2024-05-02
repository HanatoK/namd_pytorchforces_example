Steps for running the NAMD with PytorchForces

1. Compile the NAMD with PytorchForces and CudaGlobalMaster branch;
2. Unzip stmv.tar.gz;
3. Run `python3 ./build_rmsd_cv_opt.py` to compile the RMSD CV calculation into a torchscript archive (`RMSD_opt.pt`);
4. Run `namd3 +p1 ./test_stmv.namd`
