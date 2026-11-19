#!/bin/bash
# Slurm job for neeta_pipeline_20261027.py on ARCHER2.
# Normally submitted by submit_neeta_pipeline.sh (asks for the case for you):
#
#   ./submit_neeta_pipeline.sh
#
# or directly:
#
#   sbatch neeta_pipeline.slurm /work/e773/e773/ss1604/.../CFD_22 [steps]
#
#SBATCH --job-name=neeta_pipeline
#SBATCH --account=e773-Exeter
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00

CFD="$1"
STEPS="${2:-centres,extract,vtk}"
if [ -z "$CFD" ]; then
    echo "usage: sbatch neeta_pipeline.slurm /path/to/CFD [steps]"
    exit 1
fi
PIPELINE="${PIPELINE:-$SLURM_SUBMIT_DIR/neeta_pipeline_20261027.py}"

module load cray-python   # numpy + scipy

# --- OpenFOAM: only needed for the centres step (postProcess) --------------
# Same OpenFOAM build as the CFD runs (submit.sh); CFDEM isn't needed here.
source /work/e773/e773/ss1604/OpenFOAM/OpenFOAM-2506/etc/bashrc
# ---------------------------------------------------------------------------

NPROCS=$(ls -d "$CFD"/processor[0-9]* | wc -l)
export MPI_LAUNCHER="srun --hint=nomultithread --distribution=block:block -n $NPROCS"
export N_WORKERS="${N_WORKERS:-32}"   # extract + vtk workers; each needs ~1-2 GB
export OMP_NUM_THREADS=1              # stop numpy threads oversubscribing the workers

echo "case:      $CFD ($NPROCS processor dirs)"
echo "steps:     $STEPS"
echo "workers:   $N_WORKERS"
echo "pipeline:  $PIPELINE"

python3 "$PIPELINE" --cfd "$CFD" --steps "$STEPS"
