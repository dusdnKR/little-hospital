#!/bin/bash
#SBATCH --job-name=ksc_cm_augment
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=tmp/cm_augment_%j.log
set -euo pipefail
cd "${KSC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONPATH='' PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 TMPDIR="$PWD/tmp"
if [ "${1:-}" = todo ]; then
  python -m clinical_memory.data.augment todo
  exit 0
fi
python -m clinical_memory.data.augment images "$@"
python -m clinical_memory.data.augment build
python -m clinical_memory.data.augment check
