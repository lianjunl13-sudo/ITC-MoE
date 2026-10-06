# Data preparation

This directory documents inputs; complete datasets are supplied separately.
Dataset ordering and text revisions matter because calibration hashes are
checked against `configs/calibration_selection.json`.

## Calibration sources

| File | Required JSONL fields | Use |
|---|---|---|
| GSM8K training split | `question`, `answer` | Row-indexed selection of 128 training examples |
| Full MBPP source | `task_id`, `text`, `code`, `test_list` | ID-indexed selection of 128 training examples and Hot prompts |

```bash
python scripts/prepare_calibration.py \
  --gsm8k-train /path/to/gsm8k/train.jsonl \
  --mbpp-full /path/to/mbpp.jsonl --output /path/to/calibration.jsonl
```

Do not reorder the GSM8K source or substitute the sanitized MBPP evaluation
file for the full training source. A text-hash failure indicates an input
mismatch and should be resolved before training. The resulting calibration
file contains `text`, `prompt_text`, `source_id`, `dataset`, `split`, and `bucket`.

## Evaluation files

Pass the evaluation directory through `--data-dir`:

```text
eval-data/
  sanitized-mbpp.jsonl
  multiarith_full600.jsonl
  singleop_full562.jsonl
  singleq_full109.jsonl
```

Arithmetic rows have `question` and `answer`; scoring compares the last extracted
number with the reference. The sanitized MBPP file is read by OpenCompass's
`SanitizedMBPPDataset`. GSM8K is loaded through `opencompass/gsm8k`, rather than
an additional local file selected by this entry point.

The supported full evaluation counts are MBPP 257, GSM8K 1319, MultiArith 600,
SingleOp 562 and SingleQ 109. See `src/evaluation/eval_five.py` for prompts and
scorers. Supply the corresponding dataset versions and record their checksums
alongside each evaluation. Dataset files are not bundled.
