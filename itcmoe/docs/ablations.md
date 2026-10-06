# Ablations

Run from the package root on an explicitly selected available GPU. Different
comparisons must use separate output directories; existing outputs are not
overwritten.

## Complete operator presence at three budgets

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_operator.py --model /path/to/c10-model --samples results/operator_samples.json --output /path/to/speed-c10
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_operator.py --model /path/to/c20-model --samples results/operator_samples.json --output /path/to/speed-c20
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_operator.py --model /path/to/c30-model --samples results/operator_samples.json --output /path/to/speed-c30
```

Default protocol: three samples, at most 256 generated tokens each, one model
load, one excluded warmup for each mode, alternating mode order by sample,
seed 16037+sample index. Record generated text, generation time and forward
counts. This short benchmark does not score task accuracy.

Off: plain PyTorch factor operations, no shared projection reuse, no effective
core precomputation/cache. On: hybrid_down_effective, Triton, effective-core
caching and shared projections. Both retain the same weights, rank-11 Hot
compensation, candidate size 48 and decoding settings.

## Independent inference switches

```bash
python run.py evaluate --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10 --opencompass-root /path/to/opencompass --data-dir /path/to/data --dataset singleq --operator off --run-name operator-off
python run.py evaluate --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10 --opencompass-root /path/to/opencompass --data-dir /path/to/data --dataset singleq --hot-compensation off --run-name hot-off
python run.py evaluate --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10 --opencompass-root /path/to/opencompass --data-dir /path/to/data --dataset singleq --candidate off --run-name candidate-off
```

Unspecified switches remain on. Candidate-off removes expert candidate filtering
while preserving Hot/Cold classification. Hot-off disables the residual without
deleting stored parameters. `--candidate-size` changes the candidate count.
For dense-model candidate ablations, install the runtime in a separate complete
dense model view; Tucker remains disabled in its configuration.

## Anchor-coefficient presence

Keep base weights, both calibration rounds, SVD initialization and data splits
fixed. Refit only the gains:

```bash
python scripts/prepare_lambda_ablation.py --source-work /path/to/run-c10 --output-work /path/to/run-c10-lambda0
python run.py fit-hot --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10-lambda0 --anchor-lambda 0
python run.py export-hot --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10-lambda0
python run.py evaluate --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10-lambda0 --opencompass-root /path/to/opencompass --data-dir /path/to/data --dataset singleq --run-name lambda0
```

The preparation step does not copy trained gains. Temporary-file replacement
preserves original hard-linked weights during export. Development selection
continues to use only the main reconstruction error.

## Hot-token thresholds

Use `--hot-distance 3 --hot-confidence 0.7` with `run.py evaluate`.
These control Cold classification, not routing-loss weights or the generation
acceptance threshold.
