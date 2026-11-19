#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --account=e773-Exeter
#SBATCH --partition=serial
#SBATCH --qos=serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=cleanup_%j.out

BASE=/work/e773/e773/ss1604/20260820

clean() {
    local dir="$BASE/$1"; shift
    cd "$dir" || { echo "SKIP: $dir not found"; return; }
    echo "=== $dir ==="
    ls -d "$@" 2>/dev/null
    rm -rf "$@"
    echo "done $(date)"
}

shopt -s nullglob   # unmatched patterns like CFD_new* expand to nothing

clean 30_circle/SIM009_us8.896  CFD1 CFD2 CFD3 CFD_new*
clean 40_circle/SIM009_us8.896  CFD1 CFD2 CFD3 CFD_new*
clean 30_square/SIM009_us8.896  CFD1 CFD2 CFD3 CFD_new*
clean 40_square/SIM009_us8.896  CFD1 CFD2 CFD_new*

echo "All finished $(date)"
