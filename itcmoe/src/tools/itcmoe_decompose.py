#!/usr/bin/env python3
"""Build a Tucker-compressed SDAR checkpoint for itcmoe.

The script preserves non-expert weights, jointly decomposes experts in each MoE
layer, optionally applies input/output whitening, and writes a trust_remote_code
checkpoint whose MoE blocks instantiate Tucker factors instead of dense experts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROLE_TO_PARAM = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="itcmoe Tucker decomposition for SDAR")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--ratio", type=float, default=0.6)
    parser.add_argument("--rank-expert", type=int, default=64)
    parser.add_argument("--rank-multiple", type=int, default=16)
    parser.add_argument("--batude-min-rank", type=int, default=1024)
    parser.add_argument(
        "--batude-normalize-scores",
        action="store_true",
        help="Normalize per tensor spectral scores before BATUDE budget allocation.",
    )
    parser.add_argument(
        "--batude-role-balanced",
        action="store_true",
        help="Allocate BATUDE rank budget independently inside gate/up/down roles for safer no-compensation compression.",
    )
    parser.add_argument(
        "--rank-plan",
        type=Path,
        default=None,
        help="JSON rank plan overriding automatic rank allocation.",
    )
    parser.add_argument(
        "--rank-policy",
        choices=["balanced", "full-small", "batude"],
        default="balanced",
        help=(
            "Rank allocation policy. balanced keeps input/output rank ratios similar; "
            "full-small preserves the smaller matrix dimension and compresses the larger one."
        ),
    )
    parser.add_argument("--whiten-type", choices=["input", "output", "both", "none"], default="input")
    parser.add_argument(
        "--output-whiten-mode",
        choices=["inverse", "direct", "transpose"],
        default="transpose",
        help=(
            "inverse decomposes S_out^{-1} W and fuses S_out into U_out; "
            "direct decomposes S_out W and fuses S_out^{-1} into U_out. "
            "direct is safer when Sigma_out is a forward output/residual covariance."
        ),
    )
    parser.add_argument(
        "--output-whiten-roles",
        choices=["gate", "up", "down"],
        nargs="*",
        default=["gate", "up", "down"],
        help="Linear roles that receive output whitening when --whiten-type uses output whitening.",
    )
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--kernel", choices=["triton", "torch"], default="triton")
    parser.add_argument("--operator", choices=["effective", "rank"], default="effective")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--cov-cache", type=Path, default=None)
    parser.add_argument("--skip-covariance", action="store_true")
    parser.add_argument("--rand-oversample", type=int, default=16)
    parser.add_argument("--rand-iters", type=int, default=1)
    parser.add_argument("--cholesky-eps", type=float, default=1e-3)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def choose_balanced_ranks(
    n_expert: int,
    d_out: int,
    d_in: int,
    ratio: float,
    rank_expert: int,
    multiple: int = 16,
) -> Tuple[int, int, int]:
    rank_expert = min(rank_expert, n_expert)
    original_params = n_expert * d_out * d_in
    target_params = max(1, int(original_params * (1.0 - ratio)))

    best = None
    best_score = float("inf")
    out_candidates = sorted(set([d_out] + list(range(multiple, d_out + 1, multiple))))
    in_candidates = sorted(set([d_in] + list(range(multiple, d_in + 1, multiple))))

    for r_out in out_candidates:
        for r_in in in_candidates:
            compressed = (
                rank_expert * r_out * r_in
                + n_expert * rank_expert
                + d_out * r_out
                + d_in * r_in
            )
            param_error = abs(compressed - target_params) / target_params
            balance_error = abs(r_out / d_out - r_in / d_in)
            over_budget = 2.0 if compressed > target_params else 0.0
            score = over_budget + 10.0 * param_error + balance_error
            if score < best_score:
                best_score = score
                best = (rank_expert, r_out, r_in, compressed)

    assert best is not None
    rank_tuple = best[:3]
    actual_ratio = 1.0 - (best[3] / original_params)
    print(
        f"ranks for ({n_expert}, {d_out}, {d_in}) -> {rank_tuple}, "
        f"actual compression={actual_ratio:.4f}"
    )
    return rank_tuple


def choose_full_small_ranks(
    n_expert: int,
    d_out: int,
    d_in: int,
    ratio: float,
    rank_expert: int,
    multiple: int = 16,
) -> Tuple[int, int, int]:
    rank_expert = min(rank_expert, n_expert)
    original_params = n_expert * d_out * d_in
    target_params = max(1, int(original_params * (1.0 - ratio)))

    out_candidates = sorted(set([d_out] + list(range(multiple, d_out + 1, multiple))))
    in_candidates = sorted(set([d_in] + list(range(multiple, d_in + 1, multiple))))

    best = None
    best_score = float("inf")
    for r_out in out_candidates:
        for r_in in in_candidates:
            compressed = (
                rank_expert * r_out * r_in
                + n_expert * rank_expert
                + d_out * r_out
                + d_in * r_in
            )
            # Keep at least the requested compression ratio when the multiple grid allows it.
            if compressed > target_params:
                continue
            param_error = abs(compressed - target_params) / target_params
            if d_out <= d_in:
                small_dim_loss = 1.0 - (r_out / d_out)
                large_dim_loss = 1.0 - (r_in / d_in)
            else:
                small_dim_loss = 1.0 - (r_in / d_in)
                large_dim_loss = 1.0 - (r_out / d_out)
            score = 100.0 * small_dim_loss + large_dim_loss + 10.0 * param_error
            if score < best_score:
                best_score = score
                best = (rank_expert, r_out, r_in, compressed)

    if best is None:
        return choose_balanced_ranks(n_expert, d_out, d_in, ratio, rank_expert, multiple)

    rank_tuple = best[:3]
    actual_ratio = 1.0 - (best[3] / original_params)
    print(
        f"ranks for ({n_expert}, {d_out}, {d_in}) -> {rank_tuple}, "
        f"actual compression={actual_ratio:.4f}, policy=full-small"
    )
    return rank_tuple


def choose_tucker_ranks(
    n_expert: int,
    d_out: int,
    d_in: int,
    args: argparse.Namespace,
) -> Tuple[int, int, int]:
    if args.rank_policy == "full-small":
        return choose_full_small_ranks(n_expert, d_out, d_in, args.ratio, args.rank_expert)
    return choose_balanced_ranks(n_expert, d_out, d_in, args.ratio, args.rank_expert)



def _dict_get_layer(d: dict, layer_idx: int):
    if layer_idx in d:
        return d[layer_idx]
    key = str(layer_idx)
    if key in d:
        return d[key]
    raise KeyError(layer_idx)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(max(1, value) / multiple) * multiple)


def _round_down_to_multiple(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return int(math.floor(max(1, value) / multiple) * multiple)


def _batude_cov_spectrum(
    cov: torch.Tensor | None,
    dim: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return descending nonnegative spectral scores for one compressed mode.

    BATUDE selects Tucker ranks under a global parameter budget using spectral
    importance. For this SDAR/TD-MoE one-shot no-compensation setting we use
    collected activation/output covariance spectra as the budget scores, keeping
    the selection data-aware without introducing a finetuning pass.
    """
    if cov is None:
        scores = torch.ones(dim, dtype=torch.float32)
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        mat = cov.to(device=device, dtype=torch.float32)
        mat = (mat + mat.T) * 0.5
        scores = torch.linalg.eigvalsh(mat).clamp_min_(0).flip(0).cpu()
        del mat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if scores.numel() < dim:
        scores = torch.cat([scores, torch.zeros(dim - scores.numel(), dtype=scores.dtype)])
    elif scores.numel() > dim:
        scores = scores[:dim]
    if args.batude_normalize_scores:
        scores = scores / scores.sum().clamp_min(1e-12)
    return scores.float().contiguous()


def _tucker_param_count(n_expert: int, d_out: int, d_in: int, ranks: Tuple[int, int, int]) -> int:
    rank_exp, rank_out, rank_in = ranks
    return int(rank_exp * rank_out * rank_in + n_expert * rank_exp + d_out * rank_out + d_in * rank_in)


def choose_batude_dynamic_ranks(
    config: dict,
    layers: List[int],
    cov: dict | None,
    args: argparse.Namespace,
) -> Tuple[Dict[int, Dict[str, Tuple[int, int, int]]], dict]:
    """BATUDE-style global budget rank selection for TD-MoE tensors.

    We keep TD-MoE's full-small structure for stability: expert rank is full and
    the smaller matrix dimension is full. The larger hidden dimension receives a
    dynamic per-layer/per-role rank selected by greedy marginal spectral gain
    under the global compression budget.
    """
    if cov is None:
        raise ValueError("BATUDE rank policy needs activation covariances; use --whiten-type other than none or provide --cov-cache")

    n_expert = int(config["num_experts"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    multiple = max(1, int(args.rank_multiple))
    large_dim = hidden_size
    min_rank = _round_up_to_multiple(int(args.batude_min_rank), multiple)
    min_rank = min(max(multiple, min_rank), _round_down_to_multiple(large_dim, multiple))
    max_rank = _round_down_to_multiple(large_dim, multiple)

    original_per_role = n_expert * hidden_size * intermediate_size
    target_total = int(len(layers) * len(ROLE_TO_PARAM) * original_per_role * (1.0 - args.ratio))

    def ranks_from_large(role: str, rank_large: int) -> Tuple[int, int, int]:
        rank_large = int(rank_large)
        if role in {"gate", "up"}:
            return (n_expert, intermediate_size, rank_large)
        return (n_expert, rank_large, intermediate_size)

    ranks_by_layer_role: Dict[int, Dict[str, Tuple[int, int, int]]] = {
        layer: {role: ranks_from_large(role, min_rank) for role in ROLE_TO_PARAM}
        for layer in layers
    }

    base_params = sum(
        _tucker_param_count(
            n_expert,
            intermediate_size if role in {"gate", "up"} else hidden_size,
            hidden_size if role in {"gate", "up"} else intermediate_size,
            ranks_by_layer_role[layer][role],
        )
        for layer in layers
        for role in ROLE_TO_PARAM
    )
    if base_params > target_total:
        raise ValueError(
            f"BATUDE min rank {min_rank} exceeds budget: base_params={base_params}, target={target_total}. "
            f"Lower --batude-min-rank."
        )

    hidden_cov = cov.get("input", {}).get("hidden", {}) if isinstance(cov, dict) else {}
    output_cov = cov.get("output", {}) if isinstance(cov, dict) else {}
    down_output_cov = output_cov.get("down", {}) if isinstance(output_cov, dict) else {}

    hidden_scores_by_layer: Dict[int, torch.Tensor] = {}
    down_scores_by_layer: Dict[int, torch.Tensor] = {}
    for layer in tqdm(layers, desc="batude spectra"):
        hidden_matrix = _dict_get_layer(hidden_cov, layer) if hidden_cov else None
        hidden_scores_by_layer[layer] = _batude_cov_spectrum(hidden_matrix, hidden_size, args)
        down_matrix = _dict_get_layer(down_output_cov, layer) if down_output_cov else hidden_matrix
        down_scores_by_layer[layer] = _batude_cov_spectrum(down_matrix, hidden_size, args)

    chunk_cost = multiple * (n_expert * intermediate_size + hidden_size)
    budget_chunks = max(0, (target_total - base_params) // chunk_cost)
    candidates: List[Tuple[float, int, str]] = []
    for layer in layers:
        for role in ROLE_TO_PARAM:
            scores = down_scores_by_layer[layer] if role == "down" else hidden_scores_by_layer[layer]
            for start in range(min_rank, max_rank, multiple):
                end = min(start + multiple, max_rank)
                benefit = float(scores[start:end].sum().item())
                benefit += 1e-12 * (10_000 - layer * 10 - {"gate": 0, "up": 1, "down": 2}[role])
                candidates.append((benefit, layer, role))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected = 0
    selected_by_role = {role: 0 for role in ROLE_TO_PARAM}
    if getattr(args, "batude_role_balanced", False):
        base_params_by_role = {
            role: sum(
                _tucker_param_count(
                    n_expert,
                    intermediate_size if role in {"gate", "up"} else hidden_size,
                    hidden_size if role in {"gate", "up"} else intermediate_size,
                    ranks_by_layer_role[layer][role],
                )
                for layer in layers
            )
            for role in ROLE_TO_PARAM
        }
        target_per_role = target_total // len(ROLE_TO_PARAM)
        budget_chunks_by_role = {
            role: max(0, (target_per_role - base_params_by_role[role]) // chunk_cost)
            for role in ROLE_TO_PARAM
        }
        for role in ROLE_TO_PARAM:
            role_selected = 0
            role_candidates = [item for item in candidates if item[2] == role]
            for _benefit, layer, role_name in role_candidates:
                if role_selected >= budget_chunks_by_role[role]:
                    break
                large_index = 2 if role_name in {"gate", "up"} else 1
                old_rank = ranks_by_layer_role[layer][role_name][large_index]
                if old_rank >= max_rank:
                    continue
                ranks_by_layer_role[layer][role_name] = ranks_from_large(role_name, old_rank + multiple)
                role_selected += 1
            selected_by_role[role] = int(role_selected)
        selected = sum(selected_by_role.values())
    else:
        for _benefit, layer, role in candidates:
            if selected >= budget_chunks:
                break
            large_index = 2 if role in {"gate", "up"} else 1
            old_rank = ranks_by_layer_role[layer][role][large_index]
            if old_rank >= max_rank:
                continue
            ranks_by_layer_role[layer][role] = ranks_from_large(role, old_rank + multiple)
            selected += 1
            selected_by_role[role] += 1

    total_params = sum(
        _tucker_param_count(
            n_expert,
            intermediate_size if role in {"gate", "up"} else hidden_size,
            hidden_size if role in {"gate", "up"} else intermediate_size,
            ranks_by_layer_role[layer][role],
        )
        for layer in layers
        for role in ROLE_TO_PARAM
    )
    original_total = len(layers) * len(ROLE_TO_PARAM) * original_per_role
    actual_ratio = 1.0 - total_params / max(1, original_total)

    role_stats = {}
    for role in ROLE_TO_PARAM:
        large_index = 2 if role in {"gate", "up"} else 1
        vals = [ranks_by_layer_role[layer][role][large_index] for layer in layers]
        role_stats[role] = {
            "min": int(min(vals)),
            "max": int(max(vals)),
            "mean": float(sum(vals) / len(vals)),
            "values": [int(v) for v in vals],
        }
    summary = {
        "policy": "batude",
        "ratio_requested": args.ratio,
        "actual_expert_compression": actual_ratio,
        "target_expert_params": target_total,
        "actual_expert_params": total_params,
        "original_expert_params": original_total,
        "rank_multiple": multiple,
        "min_rank": min_rank,
        "max_rank": max_rank,
        "selected_chunks": int(selected),
        "chunk_cost": int(chunk_cost),
        "normalize_scores": bool(args.batude_normalize_scores),
        "role_balanced": bool(getattr(args, "batude_role_balanced", False)),
        "selected_chunks_by_role": selected_by_role,
        "role_stats": role_stats,
    }
    print(
        "BATUDE dynamic ranks: "
        f"actual expert compression={actual_ratio:.6f}, params={total_params}/{original_total}"
    )
    for role, stats in role_stats.items():
        print(f"  {role}: min={stats['min']} max={stats['max']} mean={stats['mean']:.2f}")
    return ranks_by_layer_role, summary


def patch_modeling_for_dynamic_ranks(output_path: Path) -> None:
    modeling_path = output_path / "modeling_sdar_moe.py"
    if not modeling_path.exists():
        return
    text = modeling_path.read_text(encoding="utf-8")
    original_text = text
    text = text.replace("a candidate expert set of size 3 * top_k", "a candidate expert set of size 6 * top_k")
    text = text.replace("Route the centroid and select 3 * top_k candidate experts.", "Route the centroid and select 6 * top_k candidate experts.")
    text = text.replace("max(self.top_k, 3 * self.top_k)", "max(self.top_k, 6 * self.top_k)")
    if "td_moe_ranks_by_layer_role" in text:
        if text != original_text:
            modeling_path.write_text(text, encoding="utf-8")
            print(f"Patched cold-token candidate multiplier to 6*top_k: {modeling_path}")
        return
    original = """def _sdar_get_tucker_rank(config, role: str):
    ranks = getattr(config, "td_moe_ranks", None)
    if isinstance(ranks, dict) and role in ranks:
        return tuple(int(v) for v in ranks[role])
    raise ValueError(f"td_moe_ranks must define ranks for role '{role}'")
"""
    replacement = """def _sdar_get_tucker_rank(config, role: str, layer_idx: Optional[int] = None):
    layer_ranks = getattr(config, "td_moe_ranks_by_layer_role", None)
    if isinstance(layer_ranks, dict) and layer_idx is not None:
        layer_cfg = None
        for key in (str(layer_idx), layer_idx):
            if key in layer_ranks:
                layer_cfg = layer_ranks[key]
                break
        if isinstance(layer_cfg, dict) and role in layer_cfg:
            return tuple(int(v) for v in layer_cfg[role])
    ranks = getattr(config, "td_moe_ranks", None)
    if isinstance(ranks, dict) and role in ranks:
        return tuple(int(v) for v in ranks[role])
    raise ValueError(f"td_moe ranks must define ranks for role '{role}'")
"""
    if original not in text:
        raise RuntimeError("Could not locate _sdar_get_tucker_rank block for dynamic-rank patch")
    text = text.replace(original, replacement)
    text = text.replace(
"""class SDARMoeTuckerLinear(nn.Module):
    def __init__(self, config, role: str, d_in: int, d_out: int):
        super().__init__()
        rank_exp, rank_out, rank_in = _sdar_get_tucker_rank(config, role)
""",
"""class SDARMoeTuckerLinear(nn.Module):
    def __init__(self, config, role: str, d_in: int, d_out: int, layer_idx: Optional[int] = None):
        super().__init__()
        rank_exp, rank_out, rank_in = _sdar_get_tucker_rank(config, role, layer_idx=layer_idx)
""")
    text = text.replace(
"""class SDARMoeTuckerGateUp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = SDARMoeTuckerLinear(
            config,
            role="gate",
            d_in=config.hidden_size,
            d_out=config.moe_intermediate_size,
        )
        self.up_proj = SDARMoeTuckerLinear(
            config,
            role="up",
            d_in=config.hidden_size,
            d_out=config.moe_intermediate_size,
        )
""",
"""class SDARMoeTuckerGateUp(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.gate_proj = SDARMoeTuckerLinear(
            config,
            role="gate",
            d_in=config.hidden_size,
            d_out=config.moe_intermediate_size,
            layer_idx=layer_idx,
        )
        self.up_proj = SDARMoeTuckerLinear(
            config,
            role="up",
            d_in=config.hidden_size,
            d_out=config.moe_intermediate_size,
            layer_idx=layer_idx,
        )
""")
    text = text.replace(
"""            self.gate_up_proj = SDARMoeTuckerGateUp(config)
            self.down_proj = SDARMoeTuckerLinear(
                config,
                role="down",
                d_in=config.moe_intermediate_size,
                d_out=config.hidden_size,
            )
""",
"""            self.gate_up_proj = SDARMoeTuckerGateUp(config, layer_idx=layer_idx)
            self.down_proj = SDARMoeTuckerLinear(
                config,
                role="down",
                d_in=config.moe_intermediate_size,
                d_out=config.hidden_size,
                layer_idx=layer_idx,
            )
""")
    modeling_path.write_text(text, encoding="utf-8")
    print(f"Patched dynamic-rank modeling code: {modeling_path}")

def _read_jsonl_calib_texts(path: Path, limit: int | None = None) -> List[str]:
    texts: List[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = str(item.get("question") or item.get("problem") or item.get("prompt") or "").strip()
            a = str(item.get("answer") or item.get("solution") or item.get("target") or "").strip()
            if q:
                texts.append(f"Question: {q}\nAnswer: {a}" if a else q)
            elif a:
                texts.append(a)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def _synthetic_math_calib_texts(count: int, seed: int) -> List[str]:
    rng = random.Random(seed + 17)
    out: List[str] = []
    for _ in range(max(1, count)):
        a = rng.randint(2, 97); b = rng.randint(2, 97); c = rng.randint(2, 31)
        kind = rng.randint(0, 3)
        if kind == 0:
            out.append(f"A shop has {a} boxes with {b} pencils each. It gives away {c} pencils. Compute the remaining pencils step by step. Answer: {a*b-c}.")
        elif kind == 1:
            out.append(f"If a number is increased by {a} and then multiplied by {c}, the result is {(b+a)*c}. Find the original number. Answer: {b}.")
        elif kind == 2:
            out.append(f"There are {a} students split equally into {c} groups, with {b} extra books shared later. Reason about the total and give the final integer when applicable.")
        else:
            out.append(f"Solve carefully: ({a} + {b}) * {c} - {a}. Show the arithmetic and final answer {((a+b)*c-a)}.")
    return out


def make_calibration_batches(
    tokenizer,
    nsamples: int,
    seqlen: int,
    batch_size: int,
) -> List[torch.Tensor]:
    source = os.environ.get("TD_MOE_CALIB_SOURCE", "mixed").lower()
    seed = int(os.environ.get("TD_MOE_CALIB_SEED", "2026"))
    text_mult = max(4, int(os.environ.get("TD_MOE_CALIB_TEXT_MULT", "16")))
    rng = random.Random(seed)
    target_texts = max(64, nsamples * text_mult)
    texts: List[str] = []
    data_root = Path(__import__("os").environ["ITCMOE_GSM8K_DATA"])
    if source in {"mixed", "gsm8k", "gsm8k_train"}:
        texts.extend(_read_jsonl_calib_texts(data_root / "train.jsonl"))
        texts.extend(_read_jsonl_calib_texts(data_root / "train_socratic.jsonl"))
    if source in {"mixed", "synthetic", "math_synth"}:
        texts.extend(_synthetic_math_calib_texts(target_texts, seed))
    if os.environ.get("TD_MOE_USE_DATASETS", "0") == "1":
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts.extend([row["text"] for row in ds if row.get("text")])
        except Exception as exc:
            print(f"Could not load wikitext2 calibration data ({exc}); continuing with local calibration text.")
    if not texts:
        fallback = (
            "Mathematical reasoning requires identifying variables, deriving equations, "
            "checking constraints, and returning a concise answer. "
        )
        texts = [fallback] * max(128, target_texts)
    rng.shuffle(texts)
    texts = texts[: max(target_texts, nsamples)]
    token_stream: List[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_stream.extend(ids)
            if eos is not None:
                token_stream.append(int(eos))
        if len(token_stream) >= max(nsamples * seqlen * 4, seqlen + 1):
            break
    if len(token_stream) < seqlen:
        repeats = math.ceil(seqlen / max(1, len(token_stream)))
        token_stream = (token_stream * repeats)[:seqlen]
    if len(token_stream) < nsamples * seqlen:
        repeats = math.ceil((nsamples * seqlen) / max(1, len(token_stream)))
        token_stream = (token_stream * repeats)
    max_start = max(0, len(token_stream) - seqlen)
    samples = []
    for _ in range(nsamples):
        start = rng.randint(0, max_start) if max_start > 0 else 0
        samples.append(torch.tensor(token_stream[start : start + seqlen], dtype=torch.long))
    print(f"itcmoe calibration: source={source} seed={seed} texts={len(texts)} token_stream={len(token_stream)} nsamples={nsamples} seqlen={seqlen}", flush=True)
    batches = []
    for start in range(0, len(samples), batch_size):
        batches.append(torch.stack(samples[start : start + batch_size], dim=0))
    return batches


def _needs_input_cov(whiten_type: str) -> bool:
    return whiten_type in {"input", "both"}


def _needs_output_cov(whiten_type: str) -> bool:
    return whiten_type in {"output", "both"}


def _uses_output_cov(args: argparse.Namespace, role: str) -> bool:
    return _needs_output_cov(args.whiten_type) and role in set(args.output_whiten_roles or [])


def collect_activation_covariances(args: argparse.Namespace, config: dict, layers: List[int]) -> dict:
    default_name = "tdmoe_activation_covariances.pt"
    if args.whiten_type == "input":
        default_name = "tdmoe_input_covariances.pt"
    cache_path = args.cov_cache or (args.output_path / default_name)
    if cache_path.exists():
        print(f"Loading cached covariance: {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    if args.skip_covariance:
        raise FileNotFoundError(f"--skip-covariance was set but cache is missing: {cache_path}")

    print("Loading original model for covariance collection...")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    device = torch.device(args.device)

    collect_input = _needs_input_cov(args.whiten_type)
    collect_output = _needs_output_cov(args.whiten_type)
    output_roles = set(args.output_whiten_roles or []) if collect_output else set()

    hidden_cov = {
        layer: torch.zeros(hidden_size, hidden_size, device=device, dtype=torch.float32)
        for layer in layers
    } if collect_input else {}
    inter_cov = {
        layer: torch.zeros(intermediate_size, intermediate_size, device=device, dtype=torch.float32)
        for layer in layers
    } if collect_input else {}
    output_dims = {"gate": intermediate_size, "up": intermediate_size, "down": hidden_size}
    output_cov = {
        role: {
            layer: torch.zeros(output_dims[role], output_dims[role], device=device, dtype=torch.float32)
            for layer in layers
        }
        for role in output_roles
    }
    hidden_count = {layer: 0 for layer in layers} if collect_input else {}
    inter_count = {layer: 0 for layer in layers} if collect_input else {}
    output_count = {
        role: {layer: 0 for layer in layers}
        for role in output_roles
    }
    handles = []

    def make_mlp_hook(layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, hidden_size).float()
            hidden_cov[layer_idx].add_(x.T @ x)
            hidden_count[layer_idx] += x.shape[0]

        return hook

    def make_down_hook(layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, intermediate_size).float()
            inter_cov[layer_idx].add_(x.T @ x)
            inter_count[layer_idx] += x.shape[0]

        return hook

    def make_output_hook(layer_idx: int, role: str, dim: int):
        def hook(_module, _inputs, output):
            x = output.detach().reshape(-1, dim).float()
            output_cov[role][layer_idx].add_(x.T @ x)
            output_count[role][layer_idx] += x.shape[0]

        return hook

    for layer_idx in layers:
        mlp = model.model.layers[layer_idx].mlp
        if collect_input:
            handles.append(mlp.register_forward_pre_hook(make_mlp_hook(layer_idx)))
        for expert in mlp.experts:
            if collect_input:
                handles.append(expert.down_proj.register_forward_pre_hook(make_down_hook(layer_idx)))
            if "gate" in output_roles:
                handles.append(expert.gate_proj.register_forward_hook(make_output_hook(layer_idx, "gate", intermediate_size)))
            if "up" in output_roles:
                handles.append(expert.up_proj.register_forward_hook(make_output_hook(layer_idx, "up", intermediate_size)))
            if "down" in output_roles:
                handles.append(expert.down_proj.register_forward_hook(make_output_hook(layer_idx, "down", hidden_size)))

    batches = make_calibration_batches(
        tokenizer,
        nsamples=args.calib_samples,
        seqlen=args.calib_seq_len,
        batch_size=args.calib_batch_size,
    )
    print(f"Collecting input covariances on {len(batches)} batches...")
    with torch.no_grad():
        for batch in tqdm(batches, desc="covariance"):
            batch = batch.to(device)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)

    for handle in handles:
        handle.remove()

    cov = {
        "input": {"hidden": {}, "intermediate": {}},
        "output": {role: {} for role in output_roles},
        "counts": {
            "input_hidden": hidden_count,
            "input_intermediate": inter_count,
            "output": output_count,
        },
    }
    for layer_idx in layers:
        if collect_input:
            cov["input"]["hidden"][layer_idx] = (hidden_cov[layer_idx] / max(1, hidden_count[layer_idx])).cpu()
            cov["input"]["intermediate"][layer_idx] = (inter_cov[layer_idx] / max(1, inter_count[layer_idx])).cpu()
        if collect_output:
            for role in output_roles:
                cov["output"][role][layer_idx] = (
                    output_cov[role][layer_idx] / max(1, output_count[role][layer_idx])
                ).cpu()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cov, cache_path)
    print(f"Saved covariance cache: {cache_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cov


def ensure_positive_definite(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    matrix = (matrix.float() + matrix.float().T) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(eps)
    matrix = (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
    return (matrix + matrix.T) * 0.5


def randomized_left_basis(
    matrix: torch.Tensor,
    rank: int,
    oversample: int,
    n_iter: int,
) -> torch.Tensor:
    rows, cols = matrix.shape
    rank = min(rank, rows)
    if rank == rows:
        return torch.eye(rows, device=matrix.device, dtype=torch.float32)

    q = min(cols, rank + oversample)
    omega = torch.randn(cols, q, device=matrix.device, dtype=torch.float32)
    y = matrix.float() @ omega
    qmat, _ = torch.linalg.qr(y, mode="reduced")

    for _ in range(n_iter):
        z = matrix.float().T @ qmat
        qmat, _ = torch.linalg.qr(matrix.float() @ z, mode="reduced")

    small = qmat.T @ matrix.float()
    u_hat, _, _ = torch.linalg.svd(small, full_matrices=False)
    return (qmat @ u_hat[:, :rank]).contiguous()


def nmode_product_compress(tensor: torch.Tensor, basis: torch.Tensor, mode: int) -> torch.Tensor:
    result = torch.tensordot(tensor, basis, dims=([mode], [0]))
    return result.movedim(-1, mode).contiguous()


def left_whiten_weight(weight: torch.Tensor, factor: torch.Tensor, upper: bool) -> torch.Tensor:
    out_dim = weight.shape[1]
    flat = weight.permute(1, 0, 2).reshape(out_dim, -1)
    whitened = torch.linalg.solve_triangular(factor, flat, upper=upper)
    return whitened.reshape(out_dim, weight.shape[0], weight.shape[2]).permute(1, 0, 2).contiguous()


def left_color_weight(weight: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    out_dim = weight.shape[1]
    flat = weight.permute(1, 0, 2).reshape(out_dim, -1)
    colored = factor @ flat
    return colored.reshape(out_dim, weight.shape[0], weight.shape[2]).permute(1, 0, 2).contiguous()


def decompose_weight_tensor(
    weight: torch.Tensor,
    ranks: Tuple[int, int, int],
    input_cov: torch.Tensor | None,
    output_cov: torch.Tensor | None,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(args.device)
    weight = weight.to(device=device, dtype=torch.float32)
    input_chol = None
    output_chol = None

    if _needs_output_cov(args.whiten_type) and output_cov is not None:
        cov_pd = ensure_positive_definite(output_cov.to(device), eps=args.cholesky_eps)
        output_chol = torch.linalg.cholesky(cov_pd).contiguous()
        if args.output_whiten_mode == "inverse":
            # TD-MoE/SVD-LLM style when Sigma_out is a gradient covariance:
            # decompose S_out^{-1} @ W, then fuse S_out into U_out.
            weight = left_whiten_weight(weight, output_chol, upper=False)
        elif args.output_whiten_mode == "transpose":
            weight = left_color_weight(weight, output_chol.T)
        else:
            # Forward-output/residual covariance is safer as an output-error weight:
            # decompose S_out @ W, then fuse S_out^{-1} into U_out.
            weight = left_color_weight(weight, output_chol)

    if _needs_input_cov(args.whiten_type) and input_cov is not None:
        cov_pd = ensure_positive_definite(input_cov.to(device), eps=args.cholesky_eps)
        input_chol = torch.linalg.cholesky(cov_pd)
        weight = torch.matmul(weight, input_chol)

    rank_exp, rank_out, rank_in = ranks
    unfolded_exp = weight.reshape(weight.shape[0], -1)
    u_exp = randomized_left_basis(unfolded_exp, rank_exp, args.rand_oversample, args.rand_iters)
    del unfolded_exp

    unfolded_out = weight.permute(1, 0, 2).reshape(weight.shape[1], -1)
    u_out = randomized_left_basis(unfolded_out, rank_out, args.rand_oversample, args.rand_iters)
    del unfolded_out

    unfolded_in = weight.permute(2, 0, 1).reshape(weight.shape[2], -1)
    u_in = randomized_left_basis(unfolded_in, rank_in, args.rand_oversample, args.rand_iters)
    del unfolded_in

    core = nmode_product_compress(weight, u_exp, 0)
    core = nmode_product_compress(core, u_out, 1)
    core = nmode_product_compress(core, u_in, 2)

    if output_chol is not None:
        if args.output_whiten_mode == "inverse":
            u_out = output_chol @ u_out
        elif args.output_whiten_mode == "transpose":
            u_out = torch.linalg.solve_triangular(output_chol.T, u_out, upper=True)
        else:
            u_out = torch.linalg.solve_triangular(output_chol, u_out, upper=False)

    if input_chol is not None:
        u_in = torch.linalg.solve_triangular(input_chol.T, u_in, upper=True)

    out_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    return (
        core.to("cpu", dtype=out_dtype).contiguous(),
        u_exp.to("cpu", dtype=out_dtype).contiguous(),
        u_out.to("cpu", dtype=out_dtype).contiguous(),
        u_in.to("cpu", dtype=out_dtype).contiguous(),
    )


def expert_key(layer_idx: int, expert_idx: int, param: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{param}.weight"


def is_expert_weight(key: str) -> bool:
    return ".mlp.experts." in key and key.endswith(".weight")


def add_tucker_tensors(
    output: Dict[str, torch.Tensor],
    layer_idx: int,
    role: str,
    core: torch.Tensor,
    u_exp: torch.Tensor,
    u_out: torch.Tensor,
    u_in: torch.Tensor,
) -> None:
    if role in {"gate", "up"}:
        base = f"model.layers.{layer_idx}.mlp.gate_up_proj.{role}_proj"
    else:
        base = f"model.layers.{layer_idx}.mlp.down_proj"
    output[f"{base}.core"] = core
    output[f"{base}.u_exp"] = u_exp
    output[f"{base}.u_out"] = u_out
    output[f"{base}.u_in"] = u_in


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_model_side_files(model_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for item in model_path.iterdir():
        if item.name == "model.safetensors.index.json":
            continue
        if item.suffix == ".safetensors":
            if item.name == "others.safetensors":
                link_or_copy(item, output_path / item.name)
            continue
        if item.is_file():
            shutil.copy2(item, output_path / item.name)


def build_compressed_checkpoint(args: argparse.Namespace) -> None:
    config = load_json(args.model_path / "config.json")
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    layers = args.layers if args.layers else list(range(num_layers))

    copy_model_side_files(args.model_path, args.output_path)
    patch_modeling_for_dynamic_ranks(args.output_path)
    cov = collect_activation_covariances(args, config, layers) if args.whiten_type != "none" else None

    batude_summary = None
    if args.rank_plan is not None:
        plan = load_json(args.rank_plan)
        rank_section = plan.get("ranks_by_layer_role", plan)
        ranks_by_layer_role = {
            int(layer): {role: tuple(int(v) for v in rank_section[str(layer)][role]) for role in ROLE_TO_PARAM}
            for layer in layers
        }
        batude_summary = plan.get("summary", {"policy": "batude_trained_rank_plan", "rank_plan": str(args.rank_plan)})
        first_layer = layers[0]
        ranks_by_role = {role: ranks_by_layer_role[first_layer][role] for role in ROLE_TO_PARAM}
    elif args.rank_policy == "batude":
        ranks_by_layer_role, batude_summary = choose_batude_dynamic_ranks(config, layers, cov, args)
        first_layer = layers[0]
        ranks_by_role = {role: ranks_by_layer_role[first_layer][role] for role in ROLE_TO_PARAM}
    else:
        gate_ranks = choose_tucker_ranks(num_experts, intermediate_size, hidden_size, args)
        up_ranks = gate_ranks
        down_ranks = choose_tucker_ranks(num_experts, hidden_size, intermediate_size, args)
        ranks_by_role = {"gate": gate_ranks, "up": up_ranks, "down": down_ranks}
        ranks_by_layer_role = {layer: dict(ranks_by_role) for layer in layers}

    original_index = load_json(args.model_path / "model.safetensors.index.json")
    original_map = original_index["weight_map"]
    new_weight_map: Dict[str, str] = {}
    total_size = 0

    for key, filename in original_map.items():
        if filename == "others.safetensors":
            new_weight_map[key] = filename

    for layer_idx in tqdm(layers, desc="layers"):
        layer_file = args.model_path / f"layer-{layer_idx}-ep-0-of-1.safetensors"
        out_file_name = f"layer-{layer_idx}-tdmoe.safetensors"
        out_file = args.output_path / out_file_name

        with safe_open(layer_file, framework="pt", device="cpu") as f:
            layer_tensors = {key: f.get_tensor(key) for key in f.keys()}

        new_tensors = {
            key: tensor
            for key, tensor in layer_tensors.items()
            if not is_expert_weight(key)
        }

        for role, param in ROLE_TO_PARAM.items():
            print(f"Layer {layer_idx}: decomposing {role}")
            stacked = torch.stack(
                [layer_tensors[expert_key(layer_idx, expert_idx, param)] for expert_idx in range(num_experts)],
                dim=0,
            )
            input_cov = None
            output_cov = None
            if cov is not None:
                if _needs_input_cov(args.whiten_type):
                    if "input" in cov:
                        input_cov = cov["input"]["hidden"][layer_idx] if role in {"gate", "up"} else cov["input"]["intermediate"][layer_idx]
                    else:
                        input_cov = cov["hidden"][layer_idx] if role in {"gate", "up"} else cov["intermediate"][layer_idx]
                if _uses_output_cov(args, role):
                    output_cov = cov["output"][role][layer_idx]
            core, u_exp, u_out, u_in = decompose_weight_tensor(
                stacked,
                ranks_by_layer_role[layer_idx][role],
                input_cov,
                output_cov,
                args,
            )
            add_tucker_tensors(new_tensors, layer_idx, role, core, u_exp, u_out, u_in)
            del stacked, core, u_exp, u_out, u_in
            gc.collect()
            torch.cuda.empty_cache()

        save_file(new_tensors, out_file, metadata={"format": "pt"})
        for key, tensor in new_tensors.items():
            new_weight_map[key] = out_file_name
            total_size += tensor.numel() * tensor.element_size()
        del layer_tensors, new_tensors
        gc.collect()

    if (args.output_path / "others.safetensors").exists():
        total_size += (args.output_path / "others.safetensors").stat().st_size

    config["td_moe_enabled"] = True
    config["td_moe_kernel"] = args.kernel
    config["td_moe_operator"] = args.operator
    config["td_moe_chunk_size"] = args.chunk_size
    config["td_moe_whiten_type"] = args.whiten_type
    config["td_moe_output_whiten_mode"] = args.output_whiten_mode
    config["td_moe_rank_policy"] = args.rank_policy
    config["td_moe_output_whiten_roles"] = list(args.output_whiten_roles or [])
    config["td_moe_ratio"] = args.ratio
    config["td_moe_rank_expert"] = args.rank_expert
    config["td_moe_ranks"] = {key: list(value) for key, value in ranks_by_role.items()}
    config["td_moe_ranks_by_layer_role"] = {
        str(layer): {role: list(ranks_by_layer_role[layer][role]) for role in ROLE_TO_PARAM}
        for layer in layers
    }
    if batude_summary is not None:
        config["td_moe_batude_summary"] = batude_summary
    default_cov_name = "tdmoe_activation_covariances.pt"
    if args.whiten_type == "input":
        default_cov_name = "tdmoe_input_covariances.pt"
    config["td_moe_calibration"] = {
        "samples": args.calib_samples,
        "seq_len": args.calib_seq_len,
        "covariance": str((args.cov_cache or (args.output_path / default_cov_name)).name),
    }
    save_json(config, args.output_path / "config.json")
    save_json({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, args.output_path / "model.safetensors.index.json")

    print(f"Saved compressed itcmoe checkpoint to: {args.output_path}")


def main() -> None:
    args = parse_args()
    args.model_path = args.model_path.resolve()
    args.output_path = args.output_path.resolve()
    build_compressed_checkpoint(args)


if __name__ == "__main__":
    main()
