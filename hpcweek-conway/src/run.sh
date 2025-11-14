#!/bin/bash
#SBATCH --job-name=conwey
#SBATCH --partition=armv82
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of MPI ranks
#SBATCH --cpus-per-task=16        # Number of OpenMP threads for each MPI rank
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log     # Standard output and error log
## Command(s) to run (example):

cd ..
pip3 install -r requirements.txt
rm -f NG.cpython-311-x86_64-linux-gnu.so
rm -f src/NG.cpython-311-x86_64-linux-gnu.so
python3 setup.py build_ext --inplace
cp NG.cpython-311-x86_64-linux-gnu.so src/NG.cpython-311-x86_64-linux-gnu.so
cd src
python3 main.py -S 200 200 -I 200