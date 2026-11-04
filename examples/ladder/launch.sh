#!/usr/bin/env bash
# Runs a batch of ladder jobs in parallel, each on its own copy of the case.
#
#   bash launch.sh batches/batch1_seeds.txt [parallel jobs, default 12]
#
# A batch file holds one job per line, "<name> <flags of run_linepack_formulations.py>";
# blank lines and lines starting with # are skipped. Every job gets --copy-case and
# --no-viewer, so jobs on the same rung do not rewrite each other's case.
#
# Logs go to logs/<batch>/<name>.txt, one line per finished job to logs/<batch>/status.tsv
# (name, exit code, seconds, result folder). The batch keeps running after logout:
#
#   nohup bash launch.sh batches/batch1_seeds.txt > /dev/null 2>&1 &
#   tail -f logs/batch1_seeds/status.tsv
#
# PYTHON picks the interpreter (default: python3).

set -u

batch_file=${1:?usage: bash launch.sh <batch file> [parallel jobs]}
parallel=${2:-12}

ladder_dir=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(cd "$ladder_dir/../.." && pwd)
batch=$(basename "$batch_file" .txt)
log_dir="$ladder_dir/logs/$batch"
batch_file=$(cd "$(dirname "$batch_file")" && pwd)/$(basename "$batch_file")

mkdir -p "$log_dir"
cp "$batch_file" "$log_dir/jobs.txt"
printf "name\texit\tseconds\tresult\n" > "$log_dir/status.tsv"

export PYTHON=${PYTHON:-python3}
# gurobi's Threads does not reach numpy/sklearn, whose pools otherwise take the whole
# machine in every job (vault: 2026-09-25 library threads, stage 17)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export ladder_dir log_dir parallel

run_job() {
    local index=$1
    local name=$2
    shift 2
    # result folders are named after the second the solve starts, so the first wave
    # of identical jobs is staggered to keep two of them from claiming the same one
    if [ "$index" -le "$parallel" ]; then
        sleep $(((index - 1) * 5))
    fi
    local log="$log_dir/$name.txt"
    local start=$SECONDS
    cd "$ladder_dir" || return 1
    echo "$PYTHON run_linepack_formulations.py $* --copy-case --no-viewer" > "$log"
    # shellcheck disable=SC2086
    $PYTHON run_linepack_formulations.py "$@" --copy-case --no-viewer >> "$log" 2>&1
    local code=$?
    local result
    result=$(grep -o "userData[/\\][0-9]*-[0-9]*" "$log" | tail -1)
    printf "%s\t%s\t%s\t%s\n" "$name" "$code" "$((SECONDS - start))" "${result:--}" \
        >> "$log_dir/status.tsv"
}
export -f run_job

echo "batch $batch: $(grep -cv '^\s*\(#\|$\)' "$batch_file") jobs, $parallel at a time"
echo "logs in $log_dir"

grep -v '^\s*\(#\|$\)' "$batch_file" \
    | nl -ba -w1 -s' ' \
    | xargs -P "$parallel" -L 1 bash -c 'run_job "$@"' _

echo "batch $batch done"
