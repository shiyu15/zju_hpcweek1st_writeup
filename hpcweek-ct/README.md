# HPCWeek CT

This repository contains the source code for the HPCWeek CT problem.

## Environment Setup
The program is supposed to be run on armv82 paritition of the cluster. The login node is kp01.

## Compilation
To compile the code, use the following command:
```bash
cmake -B build
cmake --build build
```

## Run
To run the program, use the following command:
```bash
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

./build/ct_recon ./input/sinogram.hdf5
```

You will see the output log in the terminal. 