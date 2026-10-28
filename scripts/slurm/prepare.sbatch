#!/bin/bash
#SBATCH --job-name=clinical_memory_prepare
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=tmp/prepare_%j.log
set -euo pipefail
cd "${KSC_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONHASHSEED=0
python -m clinical_memory download-data --download
python -m clinical_memory prepare-data
python -m clinical_memory augment images
python -m clinical_memory augment build
python -m clinical_memory augment check
python -m clinical_memory check
