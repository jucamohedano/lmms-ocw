# Project Knowledge Map

This is the maintained map for **TTW + GRPO** in `lmms-ocw/`. For a narrative overview, start with [`project_overview.md`](./project_overview.md).

## Current status snapshot

| Topic                      | Canonical location                                                                               | Notes                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Project overview           | [`project_overview.md`](./project_overview.md)                                                   | Repository roles and current TTW↔GRPO workflow.                                   |
| TTW architecture           | [`ttw/architecture.md`](./ttw/architecture.md)                                                   | Source of truth for package layout, training paths, restore, workers, and losses. |
| TTW runbook                | [`ttw/run_eval.md`](./ttw/run_eval.md)                                                           | How to launch TTW eval and point it at GRPO checkpoints.                          |
| TTW hyperparameters        | [`ttw/hyperparameters.md`](./ttw/hyperparameters.md)                                             | Mirrors current SLURM launchers and model defaults.                               |
| TTW logs / memory          | [`ttw/logging.md`](./ttw/logging.md)                                                             | CSV memory logging and console behavior.                                          |
| GRPO overview              | [`grpo/overview.md`](./grpo/overview.md)                                                         | Dataset, reward, training, and TTW handoff map.                                   |
| GRPO reward                | [`grpo/reward_design_v3.md`](./grpo/reward_design_v3.md)                                         | Must stay in sync with `verl/utils/reward_score/classification.py`.               |
| GRPO checkpoint conversion | [`grpo/checkpoint_conversion.md`](./grpo/checkpoint_conversion.md)                               | verl model merger command and TTW loading instructions.                           |
| Results                    | [`../analysis/comprehensive_results_analysis.md`](../analysis/comprehensive_results_analysis.md) | Empirical result summary, not architecture source of truth.                       |

## TTW implementation map

| Layer                           | File(s)                                                                                                                                                                                                                                                                       | Purpose                                                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Model registration              | [`src/models/_qwen2_vl.py`](../src/models/_qwen2_vl.py)                                                                                                                                                                                                                       | `qwen2-vl-{2b,7b}-ttw` factories wrap `Qwen2VL` with `TTWModel`; accepts `model_name_or_path` for GRPO-converted checkpoints. |
| TTW package                     | [`src/models/ttw/`](../src/models/ttw)                                                                                                                                                                                                                                        | `_wrapper.py`, `_strategy.py`, `_training.py`, `_batch.py`, `_config.py`, `_worker.py`, `_liger.py`.                          |
| Offline captions + GRPO parquet | [`src/utils/ttw_offline.py`](../src/utils/ttw_offline.py)                                                                                                                                                                                                                     | JSONL caption generation for TTW and parquet generation for verl GRPO.                                                        |
| Launchers                       | [`scripts/schedule_sbatch_ttw_eval_full.sh`](../scripts/schedule_sbatch_ttw_eval_full.sh), [`scripts/schedule_sbatch_ttw_eval_lora.sh`](../scripts/schedule_sbatch_ttw_eval_lora.sh), [`scripts/schedule_sbatch_ttw_eval_svf.sh`](../scripts/schedule_sbatch_ttw_eval_svf.sh) | Cluster launchers for each TTW method.                                                                                        |
| Upload tooling                  | [`data/upload_to_hf.md`](./data/upload_to_hf.md), [`upload_ttw_captions.py`](../upload_ttw_captions.py)                                                                                                                                                                       | Upload generated JSONL/parquet artifacts to HF datasets.                                                                      |

## GRPO implementation map

| Component                | Location                                                                           | Notes                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Dataset generation       | [`src/utils/ttw_offline.py`](../src/utils/ttw_offline.py), `generate_grpo_dataset` | Emits verl-compatible parquet with `prompt`, `images`, and `reward_model.ground_truth`.   |
| Reward design            | [`grpo/reward_design_v3.md`](./grpo/reward_design_v3.md)                           | Current design: format reward + answer reward; ConceptNet is not on the v3 critical path. |
| Reward implementation    | `../verl/verl/utils/reward_score/classification.py`                                | Custom rule-based reward used by verl.                                                    |
| Training launcher        | `../verl/examples/grpo_trainer/run_qwen2_5_vl_oxford_pets_grpo.sh`                 | Qwen2.5-VL GRPO + LoRA training.                                                          |
| Historical reward design | [`archive/reward_design_v2.md`](./archive/reward_design_v2.md)                     | Old ConceptNet/tiered reward; archived.                                                   |

## Analysis directory policy

Keep these in `analysis/` because they are empirical or run-specific:

- `comparison_tables.md`
- `comprehensive_results_analysis.md`
- `gpu_memory_analysis.md`
- `plot_gpu_memory.py`

If a note becomes a reusable operational instruction, move it into `docs/operations/` or `docs/ttw/`.

## Maintenance rules

1. **TTW behavior change** → update [`ttw/architecture.md`](./ttw/architecture.md); if hyperparameters or scripts change, update [`ttw/hyperparameters.md`](./ttw/hyperparameters.md).
2. **GRPO reward change** → update [`grpo/reward_design_v3.md`](./grpo/reward_design_v3.md) and `verl/utils/reward_score/classification.py` together.
3. **Dataset/upload workflow change** → update [`data/upload_to_hf.md`](./data/upload_to_hf.md) and relevant docstrings in `src/utils/ttw_offline.py`.
4. **New results** → update `analysis/`, not canonical docs, unless the code behavior changes.
5. **No loose Markdown** under repo root or `src/`; use `docs/`, `analysis/`, or `docs/archive/`.
