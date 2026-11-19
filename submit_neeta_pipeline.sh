#!/bin/bash
# Run on the ARCHER2 login node: asks for one or more case directories, then
# submits one neeta_pipeline.slurm job per case (they run side by side).
# Press Enter to accept the [default] answers.
#
#   ./submit_neeta_pipeline.sh                      # asks for the paths
#   ./submit_neeta_pipeline.sh CFD/ CFD11/ CFD22/   # or give them here
cd "$(dirname "$(readlink -f "$0")")" || exit 1

case "$PWD" in
    /home/*) echo "WARNING: $PWD is on /home, which compute nodes can't see -- keep these scripts on /work" ;;
esac

clean() {
    local p="${1//[\"\']/}"      # pasted quotes
    p="${p%>}"; p="${p%/}"
    p="${p/#\~/$HOME}"
    echo "$p"
}

add_case() {
    local p
    p="$(clean "$1")"
    if [ -d "$p/processor0" ]; then
        CASES+=("$(readlink -f "$p")")
        echo "  added $p"
    else
        echo "  '$p' has no processor0/ -- not added"
    fi
}

CASES=()
for p in "$@"; do add_case "$p"; done
if [ $# -eq 0 ]; then
    echo "Enter the decomposed CFD case paths, one per line. Empty line when done."
    while true; do
        read -e -r -p "Case $(( ${#CASES[@]} + 1 )): " LINE
        if [ -z "$LINE" ]; then
            [ ${#CASES[@]} -gt 0 ] && break
            echo "  enter at least one case"
            continue
        fi
        for p in $LINE; do add_case "$p"; done   # several on one line also works
    done
fi
[ ${#CASES[@]} -gt 0 ] || { echo "no valid cases"; exit 1; }

echo "${#CASES[@]} case(s):"
printf '  %s\n' "${CASES[@]}"

read -r -p "Steps [centres,extract,vtk]: " STEPS
read -r -p "Wall time per case [02:00:00]: " TIME
read -r -p "Workers for extract/vtk [32]: " WORKERS
STEPS="${STEPS:-centres,extract,vtk}"
TIME="${TIME:-02:00:00}"
WORKERS="${WORKERS:-32}"

for CFD in "${CASES[@]}"; do
    NPROCS=$(ls -d "$CFD"/processor[0-9]* | wc -l)
    NODES=$(( (NPROCS + 127) / 128 ))   # centres step needs one MPI rank per processor dir
    # job name e.g. neeta_30_circle_CFD11, so squeue tells the cases apart
    NAME="neeta_$(basename "$(dirname "$(dirname "$CFD")")")_$(basename "$CFD")"

    N_WORKERS="$WORKERS" PIPELINE="$PWD/neeta_pipeline_20261027.py" \
    sbatch --nodes="$NODES" --time="$TIME" \
           --job-name="$NAME" \
           --output="$CFD/log.neeta_pipeline_%j" \
           neeta_pipeline.slurm "$CFD" "$STEPS" \
      | sed "s|\$| ($NAME)|"
done
echo "logs: <case>/log.neeta_pipeline_<jobid>   (watch with: squeue -u \$USER)"
