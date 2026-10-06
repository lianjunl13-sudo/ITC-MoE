# Training and evaluation guide

The pipeline consumes an original checkpoint, calibration data and evaluation
data. It produces a compressed base model and a compensated model. This guide
describes the stage commands and their expected outputs.

## 1. Prepare the inputs

Follow [data preparation](../data/README.md) and
[checkpoint requirements](../checkpoints/README.md). Reconstruct calibration
with `scripts/prepare_calibration.py`. All commands below run from the source
root in Bash after installing the documented environment.

```bash
export CUDA_VISIBLE_DEVICES=0
ORIGINAL=/path/to/original
WORK=/path/to/run-c10
FACTORS=/path/to/shared-factors
CALIBRATION=/path/to/calibration.jsonl
MBPP=/path/to/mbpp.jsonl
OPENCOMPASS=/path/to/opencompass
EVAL_DATA=/path/to/eval-data
BUDGET=10
```

Select a GPU available for your job. Use one work directory per budget or
training variant. `OPENCOMPASS` must contain its `run.py` entry point and match
the evaluation environment. `FACTORS` can be created by training, or reused
when its recorded input identities match.

## 2. Run compression and compensation

The following runs the six pre-evaluation stages in order. Add `--dry-run` to
the Python command to inspect them without loading the model.

```bash
for STAGE in prepare covariance rank-inputs rank prepare-hot hot; do
  python run.py "$STAGE" \
    --budget "$BUDGET" --original-model "$ORIGINAL" --work-dir "$WORK" \
    --factor-library "$FACTORS" --calibration-jsonl "$CALIBRATION" \
    --mbpp-source-jsonl "$MBPP" || exit 1
done
```

| Stage | Main work | Output or completion evidence |
|---|---|---|
| `prepare` | Check calibration selection and original model layout | `reference/input_manifest.json` |
| `covariance` | Collect input/output statistics | `reference/covariance_reference.pt` |
| `rank-inputs` | Collect teacher routed activations | `base_training/rank_routed.pt` |
| `rank` | Construct/reuse ordered bases, optimize ranks, export 48 layers | `base_training/rank_plan.json`, `final_parameter_audit.json`, `base_model/` |
| `prepare-hot` | Create model view and fixed training/development prompts | `hot/prompts.json`, initial `model/` view |
| `hot` | Collect, initialize, recollect, fit and export residual compensation | `hot/round1/complete.json`, `hot/round2/complete.json`, `hot/export_final.json`, final `model/` |

The `hot` stage runs `collect_hot.py --round 1`, `build_svd.py`, initial export,
`collect_hot.py --round 2`, `fit_gains.py`, and final export. Gain fitting uses
the anchor coefficient supplied through `--anchor-lambda` (default 0.1).

`--layers` supports partial rank jobs. A complete base checkpoint is packed
after all 48 layer records exist. Full-model Hot calibration requires the
complete export. Existing `prepare-hot` outputs are protected against overwrite;
to continue an already prepared run, invoke the needed subsequent stage directly.

## 3. Evaluate the final checkpoint

```bash
python run.py evaluate \
  --budget "$BUDGET" --original-model "$ORIGINAL" --work-dir "$WORK" \
  --opencompass-root "$OPENCOMPASS" --data-dir "$EVAL_DATA" \
  --dataset singleq --run-name singleq-default
```

Change `--dataset` to `mbpp`, `gsm8k`, `multiarith` or `singleop` for other tasks.
The complete configuration-driven workflow in the main README invokes all five.
Inspect the predictions and score files below
`$WORK/evaluation/singleq-default/singleq/` as well as `timing.jsonl`.
Confirm the expected sample count; a successfully created output directory is
not evidence of a completed evaluation.

## 4. Reuse calibration for 20% and 30%

Use separate work directories and the same original model and factor library.
With `scripts/run_pipeline.py`, select the appropriate profile and pass
`--reuse-calibration-from /path/to/run-c10`. The pipeline skips covariance and
rank-input collection. It still runs rank training, Hot compensation and
evaluation for the selected budget. Do not allow concurrent writers to the
shared factor library.

## 5. Run ablations and retain evidence

Follow [ablations.md](ablations.md) for operator, candidate, Hot and anchor
comparisons. Keep input identities, rank plans, selected gains, exported weight
hashes, runtime settings, predictions, scores and timings together with each run.
