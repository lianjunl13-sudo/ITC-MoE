# Checkpoints

Weights are not bundled and no public download is specified in this anonymous
package. The supported original architecture is SDAR-30B-A3B-Chat-b32: 48 layers,
128 routed experts, hidden size 2048, expert intermediate size 768 and top-8
routing. See [the input contract](../docs/inputs_and_artifacts.md) for required
shard names and tensor keys.

| Artifact | How it is obtained | Contents |
|---|---|---|
| Original checkpoint | Supplied separately | Teacher weights and tokenizer |
| Shared factor library | Built/reused during rank training | Ordered maximum-basis factors with identity records |
| `base_model/` | `run.py rank` | Compressed expert weights without Hot residuals |
| `model/` | `prepare-hot` followed by `hot` | Base weights, residual factors, gains folded into export, runtime |

The base budget is 10%, 20% or 30% expert compression. Report the final audit
ratio when including compensation; the budget label alone does not describe
the compensated parameter count.

## Existing artifacts

Export a metadata-only shared-factor view into a complete checkpoint:

```bash
python scripts/export_shared_checkpoint.py \
  --original-model /path/to/original --library /path/to/shared-factors \
  --rank-view /path/to/case/rank_view.json --hot-root /path/to/case \
  --output /path/to/exported-model
```

The Hot directory must contain matching initialization and gains; omit
`--hot-root` for a base export. This exports existing artifacts without training.

To install the optimized runtime in an existing complete checkpoint:

```bash
python scripts/install_runtime.py \
  --source /path/to/old-model --output /path/to/new-model
```

Runtime installation and training are different operations. Evaluation through
`run.py evaluate` expects the final checkpoint at `<work-dir>/model`.
The standalone operator benchmark accepts its model directory directly.
