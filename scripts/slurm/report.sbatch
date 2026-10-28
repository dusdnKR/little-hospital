#!/bin/bash
#SBATCH --job-name=ksc_cm_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=tmp/cm_report_%j.log
set -euo pipefail
cd "${KSC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONPATH='' PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 TMPDIR="$PWD/tmp"
python -m clinical_memory.analysis.report
for run in "$@"; do
  python -m clinical_memory.analysis.notebook --run-id "$run" --lessons 12
done
