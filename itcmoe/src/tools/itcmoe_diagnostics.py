#!/usr/bin/env python3
"""Diagnostics for itcmoe SDAR checkpoints.

Checks per-layer Tucker reconstruction and sampled output error against original
expert weights, and numerical alignment of PyTorch and Triton runtime kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm


ROLE_TO_PARAM = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check itcmoe reconstruction and kernel numerical alignment")
    parser.add_argument("--original-path", required=True, type=Path)
    parser.add_argument("--compressed-path", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--roles", choices=["gate", "up", "down"], nargs="*", default=["gate", "up", "down"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--sample-tokens", type=int, default=128)
    parser.add_argument("--kernel-tokens", type=int, default=2048)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--skip-kernel", action="store_true")
    parser.add_argument("--check-activation-error", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=8)
    parser.add_argument("--calib-seq-len", type=int, default=256)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--max-tokens-per-layer", type=int, default=256)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def expert_key(layer_idx: int, expert_idx: int, param: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{param}.weight"


def tucker_base(layer_idx: int, role: str) -> str:
    if role in {"gate", "up"}:
        return f"model.layers.{layer_idx}.mlp.gate_up_proj.{role}_proj"
    return f"model.layers.{layer_idx}.mlp.down_proj"


def reconstruct_role(layer_tensors: dict[str, torch.Tensor], layer_idx: int, role: str, device: torch.device) -> torch.Tensor:
    base = tucker_base(layer_idx, role)
    core = layer_tensors[f"{base}.core"].to(device=device, dtype=torch.float32)
    u_exp = layer_tensors[f"{base}.u_exp"].to(device=device, dtype=torch.float32)
    u_out = layer_tensors[f"{base}.u_out"].to(device=device, dtype=torch.float32)
    u_in = layer_tensors[f"{base}.u_in"].to(device=device, dtype=torch.float32)
    effective = torch.einsum("er,roi->eoi", u_exp, core)
    return torch.einsum("oa,eab,ib->eoi", u_out, effective, u_in).contiguous()


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12))


def _read_jsonl_calib_texts(path: Path, limit: int | None = None) -> list[str]:
    texts: list[str] = []
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


def _synthetic_math_calib_texts(count: int, seed: int) -> list[str]:
    rng = random.Random(seed + 17)
    templates = []
    for _ in range(max(1, count)):
        a = rng.randint(2, 97)
        b = rng.randint(2, 97)
        c = rng.randint(2, 31)
        kind = rng.randint(0, 3)
        if kind == 0:
            templates.append(f"A shop has {a} boxes with {b} pencils each. It gives away {c} pencils. Compute the remaining pencils step by step. Answer: {a*b-c}.")
        elif kind == 1:
            templates.append(f"If a number is increased by {a} and then multiplied by {c}, the result is {(b+a)*c}. Find the original number. Answer: {b}.")
        elif kind == 2:
            templates.append(f"There are {a} students split equally into {c} groups, with {b} extra books shared later. Reason about the total and give the final integer when applicable.")
        else:
            templates.append(f"Solve carefully: ({a} + {b}) * {c} - {a}. Show the arithmetic and final answer {((a+b)*c-a)}.")
    return templates


def make_calibration_batches(tokenizer, nsamples: int, seqlen: int, batch_size: int) -> list[torch.Tensor]:
    """Build diverse calibration windows for rank/compensation training.

    Default is deliberately not the first GSM8K rows: we shuffle local training
    rows and sample random token windows.  Test files are never used.
    Env controls:
      TD_MOE_CALIB_SOURCE=mixed|gsm8k|synthetic|fallback
      TD_MOE_CALIB_SEED=2026
      TD_MOE_CALIB_TEXT_MULT=16   # candidate texts per requested sample
    """
    source = os.environ.get("TD_MOE_CALIB_SOURCE", "mixed").lower()
    seed = int(os.environ.get("TD_MOE_CALIB_SEED", "2026"))
    text_mult = max(4, int(os.environ.get("TD_MOE_CALIB_TEXT_MULT", "16")))
    rng = random.Random(seed)
    target_texts = max(64, nsamples * text_mult)
    texts: list[str] = []

    data_root = Path(__import__("os").environ["ITCMOE_GSM8K_DATA"])
    if source in {"mixed", "gsm8k", "gsm8k_train"}:
        texts.extend(_read_jsonl_calib_texts(data_root / "train.jsonl"))
        texts.extend(_read_jsonl_calib_texts(data_root / "train_socratic.jsonl"))

    if source in {"mixed", "synthetic", "math_synth"}:
        texts.extend(_synthetic_math_calib_texts(target_texts, seed))

    if not texts:
        fallback = (
            "Mathematical reasoning requires identifying variables, deriving equations, "
            "checking constraints, and returning a concise answer. "
        )
        texts = [fallback] * max(128, target_texts)

    rng.shuffle(texts)
    texts = texts[: max(target_texts, nsamples)]

    token_stream: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            token_stream.extend(ids)
            if eos is not None:
                token_stream.append(int(eos))
        if len(token_stream) >= max(nsamples * seqlen * 4, seqlen + 1):
            # Enough for random windows; keep memory bounded.
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
    return [torch.stack(samples[i : i + batch_size], dim=0) for i in range(0, len(samples), batch_size)]


def iter_layers(config: dict, requested: list[int] | None, max_layers: int | None) -> list[int]:
    if requested:
        layers = requested
    else:
        layers = list(range(int(config["num_hidden_layers"])))
    if max_layers is not None:
        layers = layers[:max_layers]
    return layers


def check_reconstruction(args: argparse.Namespace, config: dict, layers: Iterable[int]) -> None:
    device = torch.device(args.device)
    num_experts = int(config["num_experts"])
    print("== reconstruction ==")
    for layer_idx in layers:
        original_file = args.original_path / f"layer-{layer_idx}-ep-0-of-1.safetensors"
        compressed_file = args.compressed_path / f"layer-{layer_idx}-tdmoe.safetensors"
        with safe_open(original_file, framework="pt", device="cpu") as f:
            original_tensors = {key: f.get_tensor(key) for key in f.keys()}
        with safe_open(compressed_file, framework="pt", device="cpu") as f:
            compressed_tensors = {key: f.get_tensor(key) for key in f.keys()}

        for role in args.roles:
            param = ROLE_TO_PARAM[role]
            original = torch.stack(
                [original_tensors[expert_key(layer_idx, expert_idx, param)] for expert_idx in range(num_experts)],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            reconstructed = reconstruct_role(compressed_tensors, layer_idx, role, device)
            weight_rel = rel_error(original, reconstructed)

            sample_rel_values = []
            if args.sample_tokens > 0:
                d_in = original.shape[2]
                for expert_idx in range(num_experts):
                    x = torch.randn(args.sample_tokens, d_in, device=device, dtype=torch.float32)
                    y_ref = x @ original[expert_idx].T
                    y_new = x @ reconstructed[expert_idx].T
                    sample_rel_values.append(rel_error(y_ref, y_new))
                sample_rel = sum(sample_rel_values) / max(1, len(sample_rel_values))
                sample_max = max(sample_rel_values) if sample_rel_values else 0.0
                print(
                    f"layer={layer_idx:02d} role={role:4s} "
                    f"weight_rel={weight_rel:.6f} sampled_out_rel_mean={sample_rel:.6f} "
                    f"sampled_out_rel_max={sample_max:.6f}"
                )
            else:
                print(f"layer={layer_idx:02d} role={role:4s} weight_rel={weight_rel:.6f}")

            del original, reconstructed
            torch.cuda.empty_cache()


def get_tucker_module(model, layer_idx: int, role: str):
    mlp = model.model.layers[layer_idx].mlp
    if role == "gate":
        return mlp.gate_up_proj.gate_proj
    if role == "up":
        return mlp.gate_up_proj.up_proj
    return mlp.down_proj


def check_kernel(args: argparse.Namespace, config: dict, layers: Iterable[int]) -> None:
    if not torch.cuda.is_available() or args.device == "cpu":
        print("== kernel alignment skipped: CUDA is not available ==")
        return

    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    print("== torch vs triton kernel ==")
    model = AutoModelForCausalLM.from_pretrained(
        args.compressed_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    num_experts = int(config["num_experts"])
    for layer_idx in layers:
        for role in args.roles:
            module = get_tucker_module(model, layer_idx, role)
            d_in = module.d_in
            tokens = max(num_experts, args.kernel_tokens)
            per_expert = math.ceil(tokens / num_experts)
            expert_indices = torch.arange(num_experts, device=device).repeat_interleave(per_expert)[:tokens]
            x = torch.randn(tokens, d_in, device=device, dtype=dtype)

            old_kernel = module.kernel
            module.kernel = "torch"
            with torch.no_grad():
                y_torch = module(x, expert_indices)
            module.kernel = "triton"
            with torch.no_grad():
                y_triton = module(x, expert_indices)
            module.kernel = old_kernel

            diff = (y_torch.float() - y_triton.float()).abs()
            rel = rel_error(y_torch, y_triton)
            print(
                f"layer={layer_idx:02d} role={role:4s} "
                f"kernel_rel={rel:.8f} max_abs={float(diff.max()):.8f} "
                f"mean_abs={float(diff.mean()):.8f}"
            )

            del x, expert_indices, y_torch, y_triton
            torch.cuda.empty_cache()


def collect_routed_inputs(args: argparse.Namespace, config: dict, layers: Iterable[int]) -> dict[int, dict[str, torch.Tensor]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.original_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.original_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    wanted_layers = set(layers)
    buffers: dict[int, dict[str, list[torch.Tensor] | int]] = {
        layer: {"x": [], "experts": [], "weights": [], "count": 0}
        for layer in wanted_layers
    }
    handles = []

    def make_hook(layer_idx: int):
        def hook(module, inputs):
            slot = buffers[layer_idx]
            remaining = args.max_tokens_per_layer - int(slot["count"])
            if remaining <= 0:
                return
            x = inputs[0].detach().reshape(-1, int(config["hidden_size"]))
            take = min(remaining, x.shape[0])
            x_take = x[:take]
            with torch.no_grad():
                router_logits = module.gate(x_take)
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                topk_weights, selected_experts = torch.topk(routing_weights, module.top_k, dim=-1)
                if module.norm_topk_prob:
                    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            slot["x"].append(x_take.cpu())
            slot["experts"].append(selected_experts.cpu())
            slot["weights"].append(topk_weights.cpu())
            slot["count"] = int(slot["count"]) + take

        return hook

    for layer_idx in wanted_layers:
        handles.append(model.model.layers[layer_idx].mlp.register_forward_pre_hook(make_hook(layer_idx)))

    batches = make_calibration_batches(
        tokenizer,
        nsamples=args.calib_samples,
        seqlen=args.calib_seq_len,
        batch_size=args.calib_batch_size,
    )
    with torch.no_grad():
        for batch in tqdm(batches, desc="activation-error calibration"):
            batch = batch.to(device)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
            if all(int(slot["count"]) >= args.max_tokens_per_layer for slot in buffers.values()):
                break

    for handle in handles:
        handle.remove()

    out: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, slot in buffers.items():
        if not slot["x"]:
            continue
        out[layer_idx] = {
            "x": torch.cat(slot["x"], dim=0)[: args.max_tokens_per_layer],
            "experts": torch.cat(slot["experts"], dim=0)[: args.max_tokens_per_layer],
            "weights": torch.cat(slot["weights"], dim=0)[: args.max_tokens_per_layer],
        }

    del model
    torch.cuda.empty_cache()
    return out


def original_linear(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer_idx: int,
    role: str,
    num_experts: int,
) -> torch.Tensor:
    param = ROLE_TO_PARAM[role]
    weight = torch.stack(
        [weights[expert_key(layer_idx, expert_idx, param)] for expert_idx in range(num_experts)],
        dim=0,
    ).to(device=x.device, dtype=torch.float32)
    out = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=torch.float32)
    for expert_tensor in torch.unique(expert_indices):
        expert_idx = int(expert_tensor.item())
        mask = expert_indices == expert_idx
        out[mask] = x[mask].float() @ weight[expert_idx].T
    return out


def tucker_linear(x: torch.Tensor, expert_indices: torch.Tensor, tensors: dict[str, torch.Tensor], layer_idx: int, role: str) -> torch.Tensor:
    base = tucker_base(layer_idx, role)
    core = tensors[f"{base}.core"].to(device=x.device, dtype=torch.float32)
    u_exp = tensors[f"{base}.u_exp"].to(device=x.device, dtype=torch.float32)
    u_out = tensors[f"{base}.u_out"].to(device=x.device, dtype=torch.float32)
    u_in = tensors[f"{base}.u_in"].to(device=x.device, dtype=torch.float32)
    x_in = x.float() @ u_in
    effective = torch.einsum("er,roi->eoi", u_exp, core)
    mid = torch.empty((x.shape[0], effective.shape[1]), device=x.device, dtype=torch.float32)
    for expert_tensor in torch.unique(expert_indices):
        expert_idx = int(expert_tensor.item())
        mask = expert_indices == expert_idx
        mid[mask] = x_in[mask] @ effective[expert_idx].T
    return mid @ u_out.T


def original_moe_output(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer_idx: int,
    num_experts: int,
) -> torch.Tensor:
    flat_experts = selected_experts.reshape(-1).to(x.device)
    flat_weights = routing_weights.reshape(-1).to(x.device)
    flat_token_pos = (
        torch.arange(x.shape[0], device=x.device)
        .unsqueeze(1)
        .expand(-1, selected_experts.shape[1])
        .reshape(-1)
    )
    routed_x = x.index_select(0, flat_token_pos)
    gate = original_linear(routed_x, flat_experts, weights, layer_idx, "gate", num_experts)
    up = original_linear(routed_x, flat_experts, weights, layer_idx, "up", num_experts)
    inter = F.silu(gate) * up
    down = original_linear(inter, flat_experts, weights, layer_idx, "down", num_experts)
    out = torch.zeros((x.shape[0], down.shape[1]), device=x.device, dtype=torch.float32)
    out.index_add_(0, flat_token_pos, down * flat_weights[:, None])
    return out


def tucker_moe_output(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    flat_experts = selected_experts.reshape(-1).to(x.device)
    flat_weights = routing_weights.reshape(-1).to(x.device)
    flat_token_pos = (
        torch.arange(x.shape[0], device=x.device)
        .unsqueeze(1)
        .expand(-1, selected_experts.shape[1])
        .reshape(-1)
    )
    routed_x = x.index_select(0, flat_token_pos)
    gate = tucker_linear(routed_x, flat_experts, tensors, layer_idx, "gate")
    up = tucker_linear(routed_x, flat_experts, tensors, layer_idx, "up")
    inter = F.silu(gate) * up
    down = tucker_linear(inter, flat_experts, tensors, layer_idx, "down")
    out = torch.zeros((x.shape[0], down.shape[1]), device=x.device, dtype=torch.float32)
    out.index_add_(0, flat_token_pos, down * flat_weights[:, None])
    return out


def check_activation_error(args: argparse.Namespace, config: dict, layers: Iterable[int]) -> None:
    device = torch.device(args.device)
    num_experts = int(config["num_experts"])
    print("== routed activation MoE output error ==")
    routed = collect_routed_inputs(args, config, layers)
    for layer_idx in layers:
        if layer_idx not in routed:
            print(f"layer={layer_idx:02d} missing routed calibration samples")
            continue
        original_file = args.original_path / f"layer-{layer_idx}-ep-0-of-1.safetensors"
        compressed_file = args.compressed_path / f"layer-{layer_idx}-tdmoe.safetensors"
        with safe_open(original_file, framework="pt", device="cpu") as f:
            original_tensors = {key: f.get_tensor(key) for key in f.keys()}
        with safe_open(compressed_file, framework="pt", device="cpu") as f:
            compressed_tensors = {key: f.get_tensor(key) for key in f.keys()}

        x = routed[layer_idx]["x"].to(device=device, dtype=torch.float32)
        experts = routed[layer_idx]["experts"].to(device=device)
        weights = routed[layer_idx]["weights"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            y_ref = original_moe_output(x, experts, weights, original_tensors, layer_idx, num_experts)
            y_new = tucker_moe_output(x, experts, weights, compressed_tensors, layer_idx)
        diff = (y_ref - y_new).abs()
        print(
            f"layer={layer_idx:02d} tokens={x.shape[0]} "
            f"moe_out_rel={rel_error(y_ref, y_new):.6f} "
            f"max_abs={float(diff.max()):.6f} mean_abs={float(diff.mean()):.6f} "
            f"ref_rms={float(y_ref.float().pow(2).mean().sqrt()):.6f}"
        )
        del original_tensors, compressed_tensors, x, experts, weights, y_ref, y_new
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.original_path = args.original_path.resolve()
    args.compressed_path = args.compressed_path.resolve()
    config = load_json(args.compressed_path / "config.json")
    layers = iter_layers(config, args.layers, args.max_layers)
    if not args.skip_reconstruction:
        check_reconstruction(args, config, layers)
    if not args.skip_kernel:
        check_kernel(args, config, layers)
    if args.check_activation_error:
        check_activation_error(args, config, layers)


if __name__ == "__main__":
    main()
