# Third-party notices

itcmoe includes adapted third-party model and decoding code. The notices below
identify those components; they do not identify the authors of this submission.
Upstream copyright and license notices in source files are retained.

| Included component | Source and applicable notices | Bundled license text |
|---|---|---|
| `src/runtime/modeling_sdar_moe.py`, `src/teacher_runtime/modeling_sdar_moe.py` | SDAR model code based on Qwen / Hugging Face Transformers; Apache-2.0 file notices. TEAM adaptations are also acknowledged. | [Apache-2.0](licenses/Apache-2.0.txt), [TEAM MIT](licenses/MIT-TEAM-MoE-dLLM.txt) |
| `src/runtime/configuration_sdar_moe.py`, `src/teacher_runtime/configuration_sdar_moe.py` | SDAR model configuration, with Qwen / Hugging Face copyright and Apache-2.0 notices. | [Apache-2.0](licenses/Apache-2.0.txt) |
| `src/evaluation/decoder.py` | Adapted from TEAM-MoE-dLLM's SDAR/OpenCompass diffusion decoder. TEAM and SDAR have MIT root licenses; their embedded OpenCompass tree carries Apache-2.0 and OpenCompass attribution. | [TEAM MIT](licenses/MIT-TEAM-MoE-dLLM.txt), [SDAR MIT](licenses/MIT-SDAR.txt), [OpenCompass Apache-2.0](licenses/Apache-2.0-OpenCompass.txt) |

Applicable upstream attributions include:

- Copyright (c) 2026 PKU SEC Lab.
- Copyright (c) 2025 JetAstra.
- Copyright 2020 OpenCompass Authors.
- Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
  All rights reserved. The configuration files retain their original 2024 notice.

The license texts are reproduced without changing their copyright holders or
terms. The SDAR repository's MIT license does not replace the Apache-2.0 notices
on its Qwen/Transformers-derived model files. Likewise, the TEAM repository's
root license does not replace its embedded OpenCompass license.

## Upstream references

These fixed revisions identify the upstream materials inspected for attribution
and licensing. Adapted local files are not presented as pristine upstream copies.

- [TEAM-MoE-dLLM](https://github.com/PKU-SEC-Lab/TEAM-MoE-dLLM/tree/e9c502e5753ce79f660371e2fb4a8666f66cae75):
  `modeling_sdar_moe.py` and
  `evaluation/opencompass/opencompass/models/huggingface_bd3_decoded_jump_expert_limit_speculative.py`.
- [SDAR](https://github.com/JetAstra/SDAR/tree/4c2749ba103448f45520e8411533710a1e66574d):
  `evaluation/opencompass/opencompass/models/huggingface_bd3.py`.
- [SDAR model source](https://huggingface.co/JetLM/SDAR-30B-A3B-Chat-b32/tree/c351bbc37d240aa6871f167e8f92d694281b0c22):
  `modeling_sdar_moe.py` and `configuration_sdar_moe.py`.
- [Transformers v4.52.4](https://github.com/huggingface/transformers/tree/51f94ea06d19a6308c61bbb4dc97c40aabd12bad):
  the Qwen model implementation identified by the retained source headers.
- [OpenCompass 0.5.0](https://github.com/open-compass/opencompass/tree/a88f26845235517275885cfbbe5c473a9964e8ae):
  model-adapter infrastructure and Apache-2.0 license text.

Modifications for itcmoe include low-rank expert execution, Hot compensation,
runtime controls, calibration integration and profiling. Modified model and
decoder files carry modification notices. Their executable implementation is
unaffected by these notices.

## External dependencies and scope

OpenCompass, PyTorch, Transformers, Triton, FlashAttention and other installed
dependencies retain their own licenses. Their complete distributions, model
weights and datasets are not included in this source archive.

These third-party licenses apply to the corresponding upstream portions. This
notice does not grant an additional license for original itcmoe contributions
or relicense the package as a whole.
