# Code map

## Entry points and ownership

`scripts/run_pipeline.py` reads a profile and invokes `run.py`. The latter
validates paths/settings and dispatches individual stages to `src/`.
Keep experiment outputs in the directory passed through `--work-dir`.

| Responsibility | Implementation | Primary output |
|---|---|---|
| Recover calibration selection | `scripts/prepare_calibration.py` | Hash-checked calibration JSONL |
| Collect covariance | `src/tools/collect_covariance.py` | `reference/covariance_reference.pt` |
| Collect routed rank inputs | `src/training/rank_c*.py`, `src/tools/itcmoe_rank_calibration.py` | `base_training/rank_routed.pt` |
| Whitened Tucker decomposition | `src/tools/itcmoe_decompose.py` | Factors used by the shared library |
| Reuse ordered factor prefixes | `src/training/shared_basis.py`, `factor_store.py` | Per-layer/projection factor shards |
| Learn and discretize ranks | `src/training/rank_c10.py`, `rank_c20.py`, `rank_c30.py` | Rank plan and `base_model/` |
| Collect real Hot states | `src/compensation/collect_hot.py` | Two rounds of train/dev states |
| Initialize residual factors | `src/compensation/build_svd.py` | `hot/initial/` |
| Fit compensation gains | `src/compensation/fit_gains.py` | `hot/gains/` |
| Export compensation | `src/compensation/export_model.py` | `model/` |
| Shared-factor inference and Triton | `src/runtime/modeling_sdar_moe.py` | Model forward operations |
| Independent runtime controls | `src/runtime/options.py` | Operator/Hot/candidate settings |
| Diffusion decoding | `src/evaluation/decoder.py` | Generated responses |
| Dataset/scorer configuration | `src/evaluation/eval_five.py`, `arithmetic.py` | Task scores |
| Per-example measurements | `src/evaluation/itcmoe_timed.py` | Time, output hashes, forward counts |
| Paired operator experiment | `scripts/benchmark_operator.py` | Off/on timing and outputs |

Paths without a repeated directory in a table cell are relative to the first
path in that cell.

## Runtime organization

Shared projection dispatch, residual handling and Triton kernels live together
in `modeling_sdar_moe.py`. Exports copy this file and the architecture configuration
into the checkpoint for Hugging Face remote-code loading. Moving kernels into
another package would also require changing the checkpoint export/loading
contract; the current release retains the self-contained model source.

`teacher_runtime/` supplies the reference model implementation for calibration.
It is distinct from the optimized compressed-model runtime. `options.py`
controls the operator, candidate restriction and Hot residual independently.

## Training profiles and utilities

The three rank-training modules are explicit budget profiles. Shared factor
storage and parameter counting are factored into common modules. These modules
contain overlapping training code and expose separate entry points for each
compression budget.

`src/tools/` contains both utilities used by the default pipeline and historical
rank-selection helpers. The default route is the three-mode learner reached
through `run.py rank`; individual utility CLIs are not interchangeable with it.

## Suggested reading order

1. `run.py` and `configs/sdar_c10.json` for the complete stage sequence.
2. `shared_basis.py` and `rank_c10.py` for compression and rank selection.
3. `src/compensation/` for residual initialization and gain fitting.
4. `src/runtime/options.py` and `modeling_sdar_moe.py` for optimized execution.
5. `eval_five.py` and `benchmark_operator.py` for accuracy and speed protocols.

The tests cover small numerical invariants and workflow behavior.
