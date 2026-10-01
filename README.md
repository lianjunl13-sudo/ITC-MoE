# ITC-MoE

This repository provides the official implementation of **ITC-MoE: Importance-guided Token-aware Compression for MoE Diffusion Language Models**. An overview of the proposed ITC-MoE framework is shown below.
<img width="4200" height="1860" alt="tdmoe_method_code_aligned_cropped" src="https://github.com/user-attachments/assets/2ba3939e-dd0b-4532-aeff-65ae1c6f323b" />

ITC-MoE is designed to reduce the computation and storage costs of MoE-based diffusion language models through two components:

- **Importance-guided Adaptive Tucker Compression (IATC):** performs importance-aware Tucker decomposition and adaptive rank allocation.
- **Token-aware Compensation and Routing (TCR):** applies low-rank compensation to hot tokens and candidate-restricted expert routing to cold tokens.

## Environment

The reference environment uses Python 3.12, PyTorch 2.7.1, CUDA 12.8, and Transformers 4.52.4.

```bash
conda create -n itcmoe python=3.12 -y
conda activate itcmoe

pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

Check the environment:

```bash
python scripts/check_environment.py
```

## Data Preparation

Prepare the calibration data from GSM8K and MBPP:

```bash
python scripts/prepare_calibration.py \
  --gsm8k-train /path/to/gsm8k/train.jsonl \
  --mbpp-full /path/to/mbpp.jsonl \
  --output /path/to/calibration.jsonl
```<img width="4200" height="1860" alt="tdmoe_method_code_aligned_cropped" src="https://github.com/user-attachments/assets/0386f57b-57bd-4b79-9ca0-7cbf541c4902" />
<img width="4200" height="1860" alt="tdmoe_method_code_aligned_cropped" src="https://github.com/user-attachments/assets/0ecdae3f-dfa8-43f6-bff5-282be4f0affc" />


The original **SDAR-30B-A3B-Chat-b32** checkpoint and evaluation datasets should be prepared separately.

## Run ITC-MoE

For the 10% compression setting:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py \
  --config configs/sdar_c10.json \
  --original-model /path/to/original-model \
  --work-dir /path/to/itcmoe-c10 \
  --factor-library /path/to/shared-factors \
  --calibration-jsonl /path/to/calibration.jsonl \
  --mbpp-source-jsonl /path/to/mbpp.jsonl \
  --opencompass-root /path/to/opencompass \
  --data-dir /path/to/eval-data
```

For other compression ratios, use:

```text
configs/sdar_c20.json
configs/sdar_c30.json
```

The compressed models are saved under:

```text
work-dir/
├── base_model/     # Tucker-compressed model
└── model/          # Final model with token-aware compensation
```

## Evaluation

ITC-MoE supports evaluation on:

```text
MBPP
GSM8K
MultiArith
SingleOp
SingleQ
```

For example:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
  --budget 10 \
  --original-model /path/to/original-model \
  --work-dir /path/to/itcmoe-c10 \
  --opencompass-root /path/to/opencompass \
  --data-dir /path/to/eval-data \
  --dataset mbpp \
  --run-name mbpp-eval
```



## Acknowledgements

This implementation is built upon and inspired by existing MoE diffusion language model and compression frameworks, including **SDAR** and **TEAM**. We thank the authors of these projects for making their work publicly available.

