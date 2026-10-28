#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tmp runs/experiments
for doctor in 2B 4B 9B; do
  smoke=$(sbatch --parsable scripts/slurm/smoke.sbatch "$doctor")
  smoke=${smoke%%;*}
  printf 'smoke %s %s\n' "$doctor" "$smoke"
  for learning in self key key_dx; do
    trained=$(sbatch --parsable --dependency="afterok:$smoke" scripts/slurm/train.sbatch "$doctor" "$learning")
    printf 'training %s %s %s\n' "$doctor" "$learning" "$trained"
  done
  for which in test1 test2; do
    baseline=$(sbatch --parsable --array=9 --dependency="afterok:$smoke" scripts/slurm/test.sbatch "$which" "$doctor" key_dx)
    printf 'baseline %s %s %s\n' "$which" "$doctor" "$baseline"
  done
done
