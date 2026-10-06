#!/bin/bash
# Laptop-side, read-only status check for a training job submitted by submit.sh / run_pipeline.sh.
# Shows Slurm state, the log tail, whether the job packaged + pushed its results folder, and what reached GitHub.
#
#   scripts/check_job.sh <jobid>          one check
#   scripts/check_job.sh <jobid> --wait   poll every 2 min until the job leaves the queue, then check
set -o pipefail
LOGIN=noroian.r@login.explorer.northeastern.edu
JOB=${1:?usage: check_job.sh <jobid> [--wait]}
cd "$(dirname "$0")/.."
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "$LOGIN")

if [ "$2" = "--wait" ]; then
  while state=$("${SSH[@]}" "squeue -h -j $JOB -o '%T %M %R'" 2>/dev/null) && [ -n "$state" ]; do
    echo "$(date +%H:%M) job $JOB: $state"
    sleep 120
  done
fi

echo "== Slurm"
"${SSH[@]}" "squeue -h -j $JOB -o 'queue: %T  elapsed %M  %R' 2>/dev/null
  sacct -j $JOB -X -n -P -o State,ExitCode,Elapsed,Start,End,Partition,NodeList | tr '|' ' '"

# train.sbatch args: <task> <dataset> <run_name> <mode> <note>; recover task + run name from the submit line
read -r TASK RUN < <("${SSH[@]}" "sacct -j $JOB -X -n -P -o SubmitLine%1000" |
  grep -o 'train.sbatch.*' | awk '{print $2, $4}')
echo "   run: ${TASK:-?}/${RUN:-?}"

echo "== Log tail (progress bars stripped)"
"${SSH[@]}" "f=\$(ls ~/CapstoneProject/logs/*_$JOB.out 2>/dev/null | head -1); [ -n \"\$f\" ] || { echo '(no log yet)'; exit; }
  echo \"   \$f\"; tr '\r' '\n' < \"\$f\" | grep -v -E '[0-9]+%\|.*\||it/s\]|s/it\]' | grep -v '^\s*$' | tail -n 15
  echo; grep -E '^== Packaged' \"\$f\" || echo '(not packaged yet)'"

[ -n "$RUN" ] || exit 0
echo "== Results folder on the cluster"
"${SSH[@]}" "cd ~/CapstoneProject/Capstone/results/$TASK/$RUN 2>/dev/null && { ls; ls -d videos/*/ 2>/dev/null | tr '\n' ' '; echo;
  ls checkpoints 2>/dev/null | sed 's/^/checkpoint: /'; } || echo '(missing)'"

echo "== GitHub (origin/main)"
git fetch -q origin
files=$(git ls-tree -r --name-only origin/main -- "results/$TASK/$RUN" 2>/dev/null)
if [ -z "$files" ]; then
  echo "not pushed yet"
else
  echo "pushed: $(echo "$files" | wc -l) files in results/$TASK/$RUN ($(git log -1 --format='%h %s' origin/main -- "results/$TASK/$RUN"))"
  echo "video folders: $(echo "$files" | grep -o 'videos/[^/]*' | sort -u | sed 's|videos/||' | tr '\n' ' ')"
  echo "$files" | grep -q '\.ckpt$' && echo "WARNING: a checkpoint was pushed" || echo "no checkpoint in git (good)"
  echo "-> git pull --rebase to get it"
fi
