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
# Races: jobs named "<group>@<anything>" (e.g. des336@s3) race each other. The first of
# a group whose solver log says "Optimal solution found" wins, the others of the group
# are killed (exit 143 in status.tsv) and those not started yet are skipped (exit -).
# Same job with different --seed values gives about the fastest seed instead of the
# median one (seed spread x2 to x14 measured in batches 5-6).
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

mkdir -p "$log_dir/race"
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
    local group=""
    [ "${name#*@}" != "$name" ] && group=${name%%@*}
    if [ -n "$group" ] && [ -d "$log_dir/race/$group.won" ]; then
        printf "%s\t-\t0\tskipped, %s already won\n" "$name" "$group" \
            >> "$log_dir/status.tsv"
        return 0
    fi
    cd "$ladder_dir" || return 1
    echo "$PYTHON run_linepack_formulations.py $* --copy-case --no-viewer" > "$log"
    # setsid puts the job in a process group of its own, so that a race can kill it
    # together with anything the solver started (without setsid only the job is killed)
    local detach=""
    command -v setsid > /dev/null && detach=setsid
    # shellcheck disable=SC2086
    $detach $PYTHON run_linepack_formulations.py "$@" --copy-case --no-viewer \
        >> "$log" 2>&1 &
    local pid=$!
    [ -n "$group" ] && echo "$pid" > "$log_dir/race/$name.pid"
    wait "$pid"
    local code=$?
    if [ -n "$group" ]; then
        rm -f "$log_dir/race/$name.pid"
        # mkdir is atomic: only one job of the group can claim the win
        if [ "$code" -eq 0 ] && grep -q "Optimal solution found" "$log" \
            && mkdir "$log_dir/race/$group.won" 2>/dev/null; then
            echo "$name" > "$log_dir/race/$group.won/winner"
            local other
            for other in "$log_dir/race/$group"@*.pid; do
                [ -e "$other" ] || continue
                kill -TERM -- "-$(cat "$other")" 2>/dev/null \
                    || kill -TERM "$(cat "$other")" 2>/dev/null
            done
        elif [ -d "$log_dir/race/$group.won" ]; then
            # a killed job leaves its copy of the case behind (named after its pid)
            rm -rf "$ladder_dir"/*/userData/case_"$pid"
        fi
    fi
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
