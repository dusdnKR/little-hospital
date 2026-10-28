# Running and analyzing experiments

Use Python 3.11 with the pinned dependencies. Run all commands from the repository root. Run data preparation, augmentation and `python -m clinical_memory check` before inference. Changing source/configuration/inputs invalidates the gate receipt. Each run seals its code, inputs, model revisions, settings and schedule.

## Individual conditions

The main function is `clinical_memory/training/run.py:main`. Doctor size is `2B`, `4B` or `9B`; learning is `self`, `key` or `key_dx`; order is `random`, `easy_to_hard` or `hard_to_easy`. Seeds are 42, 43 and 44. `configs/conditions.json` enables the released conditions. An audited smoke run is required for the same model and learning mode.

After training and auditing `cm_4b_key_dx_random_s42`, evaluate Test 2:

```bash
python -m clinical_memory run --run-id cm_test2_4b_key_dx_random_s42_t302 --mode test --doctor 4B --learning key_dx --curriculum random --seed 42 --set test2 --step 302
python -m clinical_memory audit --run-id cm_test2_4b_key_dx_random_s42_t302
```

Evaluate the empty notebook once per model and test set, without learning/order/seed arguments:

```bash
python -m clinical_memory run --run-id cm_test1_4b_t000 --mode test --doctor 4B --set test1 --step 0
python -m clinical_memory audit --run-id cm_test1_4b_t000
python -m clinical_memory run --run-id cm_test2_4b_t000 --mode test --doctor 4B --set test2 --step 0
python -m clinical_memory audit --run-id cm_test2_4b_t000
```

The run reads the notebook at the requested checkpoint (0, 151 or 302). Evaluation cannot write lessons. `self` never receives an answer sheet. Diagnostic notes reach reasoning/final-answer roles only in `key_dx`, retrieved from all candidate diagnosis names.

The reflection prompt targets approximately 45 words per lesson; this is not a truncation or rejection rule. Learning changes the external notebook, not model weights.

## Slurm

The supplied GPU jobs request A100 GPUs. Jobs use the cluster's default partition; supply `sbatch --partition=NAME` when your cluster requires a specific CPU or A100 partition. Activate the intended Python environment before submission. CPU jobs also run through Slurm on clusters that prohibit work on login nodes; set `CLINICAL_MEMORY_REQUIRE_SLURM=1` there to enforce allocation checks. Create `tmp/` before submitting because Slurm opens its log path before running the job body.

```bash
mkdir -p tmp
sbatch scripts/slurm/prepare.sbatch
# After preparation succeeds:
sbatch scripts/slurm/gate.sbatch
sbatch scripts/slurm/smoke.sbatch 4B
# After the corresponding smoke checks and audits succeed:
sbatch scripts/slurm/train.sbatch 4B key_dx
sbatch --array=9 scripts/slurm/test.sbatch test1 4B key_dx
sbatch --array=9 scripts/slurm/test.sbatch test2 4B key_dx
```

`smoke.sbatch MODEL` has three tasks for the learning modes. `train.sbatch MODEL MODE` has nine tasks: `array_index // 3` selects order and `array_index % 3` selects seed. Each training task audits its run and then evaluates/audits Test 1 and Test 2. Run all nine model/mode pairs for 81 training runs. Empty-notebook evaluations add three runs per test set, giving 84 evaluations per set when complete.

`bash scripts/submit_grid.sh` submits the smoke, training and baseline jobs with dependencies. Run it **once**, after data/model downloads and the CPU gate have succeeded. It prints all submitted IDs; it does not monitor or resubmit failed jobs.

## Resume

Inspect the failed run's log before retrying. Use the same command and run ID with `--resume`:

```bash
python -m clinical_memory run --run-id cm_4b_key_dx_random_s42 --mode train --doctor 4B --learning key_dx --curriculum random --seed 42 --resume
python -m clinical_memory audit --run-id cm_4b_key_dx_random_s42
```

Changed source, prompts, settings or input bytes are refused. Do not point the refactored runner at original experiment archives: their source seals and schema names differ. A notebook is reconstructed only from committed journal episodes, so interrupted writes cannot add a lesson twice. Slurm jobs use run locks to avoid concurrent writers; remove a stale lock only after confirming its recorded job has stopped.

## Reports and logs

```bash
python -m clinical_memory analyze
python -m clinical_memory notebook --run-id cm_4b_key_dx_random_s42 --lessons 12
```

`results/summary.json` contains run completion/exclusion details, accuracy, action counts, image delivery, memory use, token/time costs, paired changes and bootstrap intervals. Paired comparisons need the matching baseline. `runs/experiments/<run-id>/` contains `seal.json`, `episodes/`, `memory_snapshot.json`, `private_calls.jsonl`, `summary.json` and `audit.json`. Results are from external-memory adaptation, not weight updates.

The bundled simulation uses a separate exported record format. Analyze it with `python -m clinical_memory analyze-replay`; it must not be treated as the full 81-run report.
