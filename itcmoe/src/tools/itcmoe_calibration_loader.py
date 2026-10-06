#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch


LEGAL_BUCKETS = ("gsm8k_train", "mbpp_train")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(jsonl_path: str) -> List[dict]:
    records: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return records


def assert_records(records: List[dict]) -> Dict[str, int]:
    counts = Counter(r.get("bucket") for r in records)
    illegal = set(counts) - set(LEGAL_BUCKETS)
    if illegal:
        raise AssertionError(f"Invalid data buckets: {sorted(illegal)}")
    if len(records) != 256:
        raise AssertionError(f"Expected 256 calibration records, found {len(records)}")
    for bucket in LEGAL_BUCKETS:
        if counts.get(bucket) != 128:
            raise AssertionError(
                f"Bucket {bucket} must contain 128 records, found {counts.get(bucket, 0)}"
            )
    for rec in records:
        bucket = rec.get("bucket")
        expected_dataset = "gsm8k" if bucket == "gsm8k_train" else "mbpp"
        if rec.get("dataset") != expected_dataset or rec.get("split") != "train":
            raise AssertionError(
                f"Record is not from an allowed training split: bucket={bucket} "
                f"dataset={rec.get('dataset')} split={rec.get('split')}"
            )
    return dict(counts)


def build_bucket_stream(records: List[dict], bucket: str, tokenizer, seed: int):
    bucket_records = [r for r in records if r.get("bucket") == bucket]
    rng = random.Random(f"{seed}:{bucket}")
    rng.shuffle(bucket_records)
    eos = getattr(tokenizer, "eos_token_id", None)
    stream: List[int] = []
    sources: List[object] = []
    for rec in bucket_records:
        ids = tokenizer.encode(rec.get("text") or "", add_special_tokens=False)
        if not ids:
            continue
        source_id = rec.get("source_id")
        stream.extend(ids)
        sources.extend([source_id] * len(ids))
        if eos is not None:
            stream.append(int(eos))
            sources.append(source_id)
    if not stream:
        raise RuntimeError(f"Bucket {bucket} produced no tokens")
    return stream, sources


def _take_windows(stream, sources, count: int, seqlen: int):
    need = count * seqlen
    if len(stream) < need:
        repeats = (need + len(stream) - 1) // len(stream)
        stream = (stream * repeats)[:need]
        sources = (sources * repeats)[:need]
    windows = []
    for index in range(count):
        start = index * seqlen
        windows.append((stream[start : start + seqlen], sources[start : start + seqlen]))
    return windows


def prefix_bucket_token_counts(sequence_buckets: List[str], seqlen: int, limit: int):
    counts = {bucket: 0 for bucket in LEGAL_BUCKETS}
    remaining = int(limit)
    for bucket in sequence_buckets:
        if remaining <= 0:
            break
        used = min(seqlen, remaining)
        counts[bucket] += used
        remaining -= used
    return counts


def build_balanced_batches(
    tokenizer,
    nsamples: int,
    seqlen: int,
    batch_size: int,
    jsonl_path: str | None = None,
    seed: int = 13037,
    balance_mode: str = "two_bucket_equal_interleaved",
) -> Tuple[List[torch.Tensor], dict]:
    if balance_mode != "two_bucket_equal_interleaved":
        raise ValueError(f"Unsupported balance mode: {balance_mode}")
    if nsamples <= 0 or nsamples % 2 != 0:
        raise AssertionError(f"nsamples must be a positive even integer, got {nsamples}")
    if seqlen <= 0 or batch_size <= 0:
        raise ValueError("seqlen and batch_size must be positive")
    if batch_size != 1:
        raise ValueError("This implementation requires batch_size=1 for per-bucket token auditing")
    if jsonl_path is None:
        raise ValueError("calib-jsonl is required")

    records = load_records(jsonl_path)
    source_record_counts = assert_records(records)
    per_bucket = nsamples // 2
    bucket_windows = {}
    for bucket in LEGAL_BUCKETS:
        stream, sources = build_bucket_stream(records, bucket, tokenizer, seed)
        bucket_windows[bucket] = _take_windows(stream, sources, per_bucket, seqlen)

    sequences: List[torch.Tensor] = []
    sequence_buckets: List[str] = []
    contributing_source_ids = set()
    all_ids: List[int] = []
    for index in range(per_bucket):
        for bucket in LEGAL_BUCKETS:
            window, sources = bucket_windows[bucket][index]
            sequences.append(torch.tensor(window, dtype=torch.long))
            sequence_buckets.append(bucket)
            all_ids.extend(window)
            contributing_source_ids.update(s for s in sources if s is not None)

    batches = [sequence.unsqueeze(0) for sequence in sequences]
    bucket_sequence_counts = Counter(sequence_buckets)
    bucket_token_counts = {
        bucket: int(bucket_sequence_counts[bucket]) * seqlen for bucket in LEGAL_BUCKETS
    }
    info = {
        "calibration_jsonl": str(jsonl_path),
        "calibration_jsonl_sha256": sha256_file(str(jsonl_path)),
        "seed": int(seed),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "batch_size": int(batch_size),
        "balance_mode": balance_mode,
        "legal_buckets": list(LEGAL_BUCKETS),
        "source_record_counts": source_record_counts,
        "bucket_sequence_counts": dict(bucket_sequence_counts),
        "bucket_token_counts": bucket_token_counts,
        "gsm8k_effective_token_count": bucket_token_counts["gsm8k_train"],
        "mbpp_effective_token_count": bucket_token_counts["mbpp_train"],
        "contributing_source_ids": sorted(contributing_source_ids),
        "sequence_buckets": sequence_buckets,
        "vocab_size": getattr(tokenizer, "vocab_size", None) or len(tokenizer),
        "sequence_input_ids_sha256": hashlib.sha256(
            ",".join(str(x) for x in all_ids).encode("utf-8")
        ).hexdigest(),
    }
    return batches, info


def write_sampling_manifest(info: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    manifest = {key: value for key, value in info.items() if key != "sequence_buckets"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced GSM8K and MBPP calibration batches")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calib-jsonl", required=True)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13037)
    parser.add_argument("--balance-mode", default="two_bucket_equal_interleaved")
    parser.add_argument("--manifest-out")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    batches, info = build_balanced_batches(
        tokenizer,
        args.nsamples,
        args.seqlen,
        args.batch_size,
        jsonl_path=args.calib_jsonl,
        seed=args.seed,
        balance_mode=args.balance_mode,
    )
    print(f"Batches={len(batches)}, sequence length={args.seqlen}")
    print(json.dumps({k: v for k, v in info.items() if k != "sequence_buckets"}, ensure_ascii=False, indent=2))
    if args.manifest_out:
        write_sampling_manifest(info, args.manifest_out)
        print(f"Sampling manifest written to: {args.manifest_out}")


if __name__ == "__main__":
    main()
