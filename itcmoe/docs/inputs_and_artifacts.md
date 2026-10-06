# Inputs and artifacts

## Original checkpoint

The supported architecture has 48 layers, 128 routed experts, hidden size 2048,
intermediate size 768, and top-8 routing. The original checkpoint must contain
`config.json`, tokenizer files, generation configuration, runtime source,
`layer-0-ep-0-of-1.safetensors` through `layer-47-ep-0-of-1.safetensors`,
`others.safetensors`, and `model.safetensors.index.json`.

Expert keys follow `model.layers.{layer}.mlp.experts.{expert}.{role}_proj.weight`.
Arbitrary HF shard layouts are not silently reshaped.

## Calibration and evaluation data

Calibration uses 128 GSM8K training examples and 128 MBPP training examples.
`prepare_calibration.py` reconstructs text from the supplied data and checks the
recorded SHA256 hashes. Test examples are not used to select calibration inputs.

The evaluation directory contains `sanitized-mbpp.jsonl`,
`multiarith_full600.jsonl`, `singleop_full562.jsonl`, and `singleq_full109.jsonl`.
Arithmetic JSONL rows contain `question` and `answer`. GSM8K uses the
OpenCompass `opencompass/gsm8k` dataset/cache convention. Dataset revision and
scoring implementation are part of the evaluation configuration.

## Work directory

| Directory | Contents |
|---|---|
| `reference/` | Calibration JSONL, covariance and input manifests |
| `base_training/` | Routed inputs, layer logs, rank plan and parameter audit |
| `base_model/` | Compressed checkpoint without residual compensation |
| `hot/round1`, `hot/round2` | Real Hot states with train/development labels |
| `hot/initial` | Rank-11 residual factors |
| `hot/gains` | Selected gains, checkpoints and selection records |
| `model/` | Final compensated checkpoint |
| `evaluation/{run-name}/{dataset}` | Per-run scores and timings |

The factor library contains one safetensors shard and hash record per layer and
projection. It is selected with `--factor-library`, or defaults to
`shared_factors/` within the work directory. All budgets should reuse one
library. The largest gate/up basis is `(124,752,2016)`; down swaps the last two
dimensions. The 20%/30% training profiles use shorter prefixes of this library.

## Export an existing shared-factor model

```bash
python scripts/export_shared_checkpoint.py \
  --original-model /path/to/original --library /path/to/shared-factors \
  --rank-view /path/to/case/rank_view.json --hot-root /path/to/case \
  --output /path/to/exported-model
```

The Hot root must contain matching `initial/` and `gains/` shards and checksum
records. Omit `--hot-root` to export a base model. Export verifies source hashes
and does not retrain. A complete export requires additional disk space.
