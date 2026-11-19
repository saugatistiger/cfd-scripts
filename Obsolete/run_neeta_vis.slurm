#!/bin/bash
#SBATCH --job-name=neeta_vis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --account=e773-Exeter
#SBATCH --partition=standard
#SBATCH --qos=standard

module load cray-python

cd /work/e773/e773/ss1604/Python_codes
export N_WORKERS=64   # 256 GB node -> ~4 GB per worker, plenty of headroom
export OMP_NUM_THREADS=1

python3 build_neeta_vis_20261013all.py
