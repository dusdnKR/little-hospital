#!/bin/bash
#SBATCH --job-name=ksc_cm_train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-8
#SBATCH --output=tmp/cm_train_%A_%a.log
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
doctor="$1"
learning="$2"
shift 2
case "$doctor" in 2B|4B|9B) ;; *) echo "Invalid doctor" >&2; exit 2;; esac
case "$learning" in self|key|key_dx) ;; *) echo "Invalid learning mode" >&2; exit 2;; esac

tag="$(echo "$doctor" | tr '[:upper:]' '[:lower:]')"
curricula=(random easy_to_hard hard_to_easy)
seeds=(42 43 44)
curriculum="${curricula[$((SLURM_ARRAY_TASK_ID / 3))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
name="${tag}_${learning}_${curriculum}_s${seed}"
run_id="cm_${name}"
if [ -f "runs/experiments/${run_id}/audit.json" ]; then
  echo "${run_id} already audited"
elif claim "$run_id"; then
  python -m clinical_memory.training.run --run-id "$run_id" --mode train --doctor "$doctor" --learning "$learning" \
    --curriculum "$curriculum" --seed "$seed" "$@"
  python -m clinical_memory.evaluation.audit --run-id "$run_id"
  release
else
  exit 0
fi
for which in test1 test2; do
  seen_id="cm_${which}_${name}_t302"
  if [ -f "runs/experiments/${seen_id}/audit.json" ]; then
    echo "${seen_id} already audited"
    continue
  fi
  if [ "$which" = test2 ] && [ ! -f data/augmented/manifest.json ]; then
    echo "${seen_id} not run: the augmented patients are not built yet. Submit test.sbatch test2 for this task later."
    continue
  fi
  if ! claim "$seen_id"; then continue; fi
  again=()
  if [ -f "runs/experiments/${seen_id}/seal.json" ]; then again=(--resume); fi
  python -m clinical_memory.training.run --run-id "$seen_id" --mode test --doctor "$doctor" --set "$which" --step 302 \
    --learning "$learning" --curriculum "$curriculum" --seed "$seed" ${again[@]+"${again[@]}"}
  python -m clinical_memory.evaluation.audit --run-id "$seen_id"
  release
done
