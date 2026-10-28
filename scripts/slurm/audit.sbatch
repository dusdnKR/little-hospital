#!/bin/bash
#SBATCH --job-name=ksc_cm_audit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=tmp/cm_audit_%j.log
set -uo pipefail
cd "${KSC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONPATH='' PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 TMPDIR="$PWD/tmp"
failed=0
for run in "$@"; do
  python -m clinical_memory.evaluation.audit --run-id "$run" || failed=1
done
exit "$failed"
