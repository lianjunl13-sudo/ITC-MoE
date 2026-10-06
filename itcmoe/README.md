# itcmoe

Importance-aware low-rank expert compression with Hot-token residual compensation
and shared-factor inference for SDAR-30B-A3B-Chat-b32.

This repository includes calibration, whitened Tucker decomposition, three-mode
rank learning, compensation training, checkpoint export, task evaluation and
operator ablations. Base expert-compression profiles: **10%, 20% and 30%**.

## Overview

```text
Original checkpoint + training-only calibration
    -> covariance and routed input collection
    -> shared whitened factors + three-mode rank learning
    -> compressed base checkpoint
    -> Hot-state collection + residual SVD
    -> second Hot-state collection + gain fitting
    -> compensated checkpoint
    -> task scores / operator latency
```

The optimized runtime computes the shared gate/up input projection once per
token and aggregates down-projection contributions in low-rank space before
one shared output projection. It also includes Triton kernels and effective-core
caching. Candidate restriction and Hot compensation have independent controls.

| Default | Value |
|---|---|
| Candidate size | 48 |
| Hot residual rank | 11 |
| Compensation anchor coefficient | 0.1 |
| Hot distance / confidence threshold | 3 / 0.7 |
| Rank learning / gain fitting | 90 / 32 steps |

Full definitions and seeds: [hyperparameters](docs/hyperparameters.md).
The base compression budgets exclude residual compensation. Rank-11 compensation
adds 570,949,632 parameters, reducing net expert compression by about 1.9694
percentage points.

## Installation

Use Linux, Python 3.12 and a CUDA-capable PyTorch installation. The reference
stack uses PyTorch 2.7.1+cu128, Transformers 4.52.4 and Triton 3.3.1.
Run all commands from the repository root.

```bash
conda create -n itcmoe python=3.12 -y
conda activate itcmoe
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python scripts/check_environment.py
```

The full-model entry point conservatively checks for 70 GiB free GPU memory,
96 GiB cgroup memory headroom when a finite root cgroup limit is visible, and
160 GiB free disk for the rank stage. These checks are not measured peak usage;
additional storage is needed for multiple runs and checkpoints.

The dependency pins specify the reference environment. The reference evaluation
uses a locally adapted OpenCompass 0.5.0 checkout with the bundled adapters.
Provide the OpenCompass source directory through `--opencompass-root`.

## Quick check

These commands check the source and display a short timing protocol without
loading model weights. They do not perform model inference.

```bash
python scripts/verify_manifest.py
python -m unittest discover -s tests -v
python scripts/benchmark_operator.py --model /path/to/model \
  --samples results/operator_samples.json --output /path/to/timing --dry-run
```

CPU-only source checks can use `requirements-cpu.txt` and
`python scripts/check_environment.py --cpu-only`.

## Data and checkpoints

Supply the original SDAR checkpoint and the training/evaluation datasets.
Weights, covariance caches, factor libraries and full datasets are not bundled.

- [Data preparation and expected schemas](data/README.md)
- [Checkpoint format and export](checkpoints/README.md)
- [Input contracts and generated artifacts](docs/inputs_and_artifacts.md)

## Compression and compensation training

First reconstruct the recorded calibration selection from the original datasets:

```bash
python scripts/prepare_calibration.py \
  --gsm8k-train /path/to/gsm8k/train.jsonl \
  --mbpp-full /path/to/mbpp.jsonl --output /path/to/calibration.jsonl
```

Run the 10% pipeline on an available GPU selected by `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py \
  --config configs/sdar_c10.json \
  --original-model /path/to/original --work-dir /path/to/run-c10 \
  --factor-library /path/to/shared-factors \
  --calibration-jsonl /path/to/calibration.jsonl \
  --mbpp-source-jsonl /path/to/mbpp.jsonl \
  --opencompass-root /path/to/opencompass --data-dir /path/to/eval-data
```

Add `--dry-run` to inspect the commands first. The supplied configuration runs
all training stages and then all five evaluation datasets. The trained outputs
are `base_model/` and `model/` inside the work directory.

For 20% or 30%, select `configs/sdar_c20.json` or `configs/sdar_c30.json`, use a
separate work directory, and add `--reuse-calibration-from /path/to/run-c10`.
Use the same factor library and run budgets serially when writing that library.
See [the stage-by-stage guide](docs/usage.md) for individual commands,
expected outputs and completion checks.

## Task evaluation

Evaluate a completed `model/` in a training work directory:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
  --budget 10 --original-model /path/to/original --work-dir /path/to/run-c10 \
  --opencompass-root /path/to/opencompass --data-dir /path/to/eval-data \
  --dataset singleq --run-name singleq-default
```

Datasets: `mbpp`, `gsm8k`, `multiarith`, `singleop`, `singleq`.
Scores and predictions are written under
`evaluation/singleq-default/singleq/`; per-example timing is in `timing.jsonl`.
Task evaluation uses a maximum of 4096 generated tokens, unlike the short
operator benchmark below.

## Operator speed and ablations

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_operator.py \
  --model /path/to/run-c10/model --samples results/operator_samples.json \
  --output /path/to/speed-c10
```

This compares the same compressed checkpoint with the complete operator off/on:
plain PyTorch factor operations versus optimized shared-factor execution. Both
retain Hot compensation and candidate restriction. The default protocol uses
three prompts, 256 generated tokens, separate warmups, alternating mode order
and matched seeds. It records generation time, outputs and forward counts.

Repeat with the 20% and 30% checkpoints in separate result directories.
[Ablation instructions](docs/ablations.md) also cover candidate restriction,
Hot compensation, Hot thresholds and anchor-coefficient refitting.

## Repository organization

```text
run.py                Individual stage entry point
configs/              Compression profiles and calibration selection
scripts/              Pipeline, validation, export and benchmark entry points
src/
  tools/              Calibration, covariance and decomposition utilities
  training/           Rank learning, factor storage and base export
  compensation/       Hot sampling, residual SVD, gain fitting and export
  runtime/            Optimized inference, Triton kernels and switches
  evaluation/         Diffusion decoding, task adapters and timing
  teacher_runtime/    Reference architecture used by calibration
data/                 Data preparation instructions
checkpoints/          Weight format and export instructions
docs/                 Usage, code map, hyperparameters and ablations
tests/                Numerical and workflow checks
results/              Benchmark prompts
licenses/             Third-party license texts
```

Start with [the code map](docs/code_map.md) to locate each method component.
Generated work directories belong outside the source tree.

## Attribution and license

This is an anonymous source package. Author-identifying paper links and citation
metadata are not included. Original third-party notices remain in the source.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component attribution,
upstream references and the bundled license texts. These third-party licenses
apply to their respective components and do not license the entire project.

The method name is `itcmoe`. Existing checkpoint fields and architecture names
remain compatible; details are in [version notes](docs/version_notes.md).
