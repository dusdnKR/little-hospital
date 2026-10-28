#!/bin/bash
#SBATCH --job-name=ksc_cm_smoke
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=tmp/cm_smoke_%A_%a.log
set -euo pipefail
cd "${KSC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONPATH='' PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
doctor="$1"
shift
case "$doctor" in 2B|4B|9B) ;; *) echo "Invalid doctor" >&2; exit 2;; esac

tag="$(echo "$doctor" | tr '[:upper:]' '[:lower:]')"
ways=(self key key_dx)
learning="${ways[$SLURM_ARRAY_TASK_ID]}"
run_id="cm_smoke_${tag}_${learning}"
python -m clinical_memory.training.run --run-id "$run_id" --mode smoke --doctor "$doctor" --learning "$learning" "$@"
python -m clinical_memory.evaluation.audit --run-id "$run_id"
