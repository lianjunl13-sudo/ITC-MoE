# Results and benchmark inputs

| File | Purpose | Status |
|---|---|---|
| `summary.json` | Selected historical experiment summaries | Not measurements from current release validation |
| `operator_samples.json` | Three fixed prompts for short operator timing | Benchmark inputs, not a task evaluation dataset |

The short operator benchmark records wall time, generated outputs and forward
counts. Its generation speedup includes the actual decoding trajectory; when
outputs or forward counts differ, it is not a fixed-work kernel speedup.
It also does not establish task accuracy from three prompts.

Fresh task runs write predictions, scores and `timing.jsonl` under
`<work-dir>/evaluation/<run-name>/<dataset>/`. Full training artifacts and
per-run outputs are kept outside the source package.

The ongoing paired validation covers the 10% profile, all 48 rank-training
layers, Hot compensation and SingleQ 109. Until its comparison completes,
no claim of successful before/after reproduction is made here.
