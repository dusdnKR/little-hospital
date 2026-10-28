#!/bin/bash
#SBATCH --job-name=ksc_cm_test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=tmp/cm_test_%A_%a.log
set -euo pipefail
cd "${KSC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONPATH='' PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0 TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
held=""
claim() {
  if mkdir "runs/experiments/$1.lock" 2>/dev/null; then
    echo "${SLURM_JOB_ID:-unknown}" > "runs/experiments/$1.lock/job"
    held="$1"
    return 0
  fi
  echo "$1 not run here: job $(cat "runs/experiments/$1.lock/job" 2>/dev/null) is working on it. If that job is no longer in squeue, remove runs/experiments/$1.lock and submit again."
  return 1
}
release() {
  if [ -n "$held" ]; then rm -rf "runs/experiments/${held}.lock"; held=""; fi
}
trap release EXIT
mkdir -p runs/experiments
which="$1"
doctor="$2"
learning="$3"
step="${4:-302}"
number="$((10#$step))"
case "$doctor" in 2B|4B|9B) ;; *) echo "Invalid doctor" >&2; exit 2;; esac
case "$learning" in self|key|key_dx) ;; *) echo "Invalid learning mode" >&2; exit 2;; esac

tag="$(echo "$doctor" | tr '[:upper:]' '[:lower:]')"
curricula=(random easy_to_hard hard_to_easy)
seeds=(42 43 44)
if [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
  run_id="cm_${which}_${tag}_t000"
  named=(--step 0)
else
  curriculum="${curricula[$((SLURM_ARRAY_TASK_ID / 3))]}"
  seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
  run_id="cm_${which}_${tag}_${learning}_${curriculum}_s${seed}_t$(printf '%03d' "$number")"
  named=(--step "$number" --learning "$learning" --curriculum "$curriculum" --seed "$seed")
  trained="cm_${tag}_${learning}_${curriculum}_s${seed}"
  if [ ! -f "runs/experiments/${trained}/audit.json" ]; then
    echo "${run_id} not run: the training run ${trained} is not completed and audited yet. Submit this task again later."
    exit 0
  fi
fi
if [ -f "runs/experiments/${run_id}/audit.json" ]; then
  echo "${run_id} already audited"
  exit 0
fi
if [ "$which" = test2 ] && [ ! -f data/augmented/manifest.json ]; then
  echo "${run_id} not run: the augmented patients are not built yet. Run augment.sbatch first."
  exit 0
fi
if ! claim "$run_id"; then exit 0; fi
again=()
if [ -f "runs/experiments/${run_id}/seal.json" ]; then again=(--resume); fi
python -m clinical_memory.training.run --run-id "$run_id" --mode test --doctor "$doctor" --set "$which" "${named[@]}" ${again[@]+"${again[@]}"}
python -m clinical_memory.evaluation.audit --run-id "$run_id"
