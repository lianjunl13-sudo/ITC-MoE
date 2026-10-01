#!/usr/bin/env python3
"""itcmoe layer-wise rank-training helpers for SDAR.

Original 30B parameters are frozen. For each MoE layer, this script trains only
three continuous prefix-rank masks (gate/up/down) on routed calibration tokens.
The training objective is routed MoE output reconstruction loss plus a parameter
budget penalty. The learned masks are rounded to integer Tucker ranks and saved
as a rank plan consumable by itcmoe_decompose.py --rank-plan.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tqdm import tqdm

ROLE_TO_PARAM = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
ROLE_INDEX = {"gate": 0, "up": 1, "down": 2}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--cov-cache", required=True, type=Path)
    p.add_argument("--output-plan", required=True, type=Path)
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--rank-min", type=int, default=1024)
    p.add_argument("--rank-max", type=int, default=1984)
    p.add_argument("--rank-step", type=int, default=32)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--calib-samples", type=int, default=4)
    p.add_argument("--calib-seq-len", type=int, default=256)
    p.add_argument("--calib-batch-size", type=int, default=1)
    p.add_argument("--max-tokens-per-layer", type=int, default=128)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--budget-weight", type=float, default=10.0)
    p.add_argument("--rank-budget-mode", choices=["penalty", "normalized"], default="penalty",
                   help="penalty uses a soft budget penalty; normalized keeps the per-layer rank-sum budget fixed during training.")
    p.add_argument("--role-loss-weight", type=float, default=0.05)
    p.add_argument("--loss-type", choices=["relative", "direct", "mixed", "grad_direct", "grad_mixed"], default="relative",
                   help="Rank-training reconstruction loss: relative/direct/mixed MSE, or grad_direct/grad_mixed using output gradient covariance quadratic loss for down/MoE outputs.")
    p.add_argument("--direct-weight", type=float, default=1.0)
    p.add_argument("--relative-weight", type=float, default=0.05)
    p.add_argument("--reconstruction-loss-scale", type=float, default=1.0,
                   help="Multiplier applied to reconstruction losses before adding the budget penalty.")
    p.add_argument("--hot-token-weight", type=float, default=1.0,
                   help="Weight multiplier for state-aware calibration tokens during rank training. 1 disables weighting.")
    p.add_argument("--rank-token-state", choices=["all", "hot", "cold"], default="all",
                   help="Which token state receives larger loss weight. hot uses high router confidence; cold uses low router confidence.")
    p.add_argument("--temp-start", type=float, default=96.0)
    p.add_argument("--temp-end", type=float, default=8.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--whiten-type", choices=["input", "output", "both", "none"], default="both")
    p.add_argument("--output-whiten-mode", choices=["inverse", "direct"], default="direct")
    p.add_argument("--output-whiten-roles", choices=["gate", "up", "down"], nargs="*", default=["down"])
    p.add_argument("--rand-oversample", type=int, default=16)
    p.add_argument("--rand-iters", type=int, default=1)
    p.add_argument("--cholesky-eps", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def expert_key(layer_idx: int, expert_idx: int, param: str) -> str:
    return f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{param}.weight"


def rank_tuple(role: str, num_experts: int, hidden: int, inter: int, large_rank: int):
    if role in {"gate", "up"}:
        return [num_experts, inter, int(large_rank)]
    return [num_experts, int(large_rank), inter]


def pcount(role: str, num_experts: int, hidden: int, inter: int, large_rank: int, td) -> int:
    d_out = inter if role in {"gate", "up"} else hidden
    d_in = hidden if role in {"gate", "up"} else inter
    return td._tucker_param_count(num_experts, d_out, d_in, tuple(rank_tuple(role, num_experts, hidden, inter, large_rank)))


def routed_linear_dense(x: torch.Tensor, expert_indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    out = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=torch.float32)
    for expert_tensor in torch.unique(expert_indices):
        expert_idx = int(expert_tensor.item())
        mask = expert_indices == expert_idx
        out[mask] = x[mask].float() @ weight[expert_idx].float().T
    return out


def factor_linear(role: str, x: torch.Tensor, expert_indices: torch.Tensor, factors: dict, soft_mask: torch.Tensor) -> torch.Tensor:
    # factors are frozen; soft_mask is differentiable and controls the large Tucker mode.
    weight_dtype = factors["u_in"].dtype
    x_proj = x.to(weight_dtype) @ factors["u_in"]
    effective = factors["effective"]
    if role in {"gate", "up"}:
        x_proj = x_proj * soft_mask.to(weight_dtype)
        mid = torch.empty((x.shape[0], effective.shape[1]), device=x.device, dtype=weight_dtype)
        for expert_tensor in torch.unique(expert_indices):
            expert_idx = int(expert_tensor.item())
            sel = expert_indices == expert_idx
            mid[sel] = x_proj[sel] @ effective[expert_idx].transpose(0, 1)
        return (mid @ factors["u_out"].T).float()
    mid = torch.empty((x.shape[0], effective.shape[1]), device=x.device, dtype=weight_dtype)
    for expert_tensor in torch.unique(expert_indices):
        expert_idx = int(expert_tensor.item())
        sel = expert_indices == expert_idx
        mid[sel] = x_proj[sel] @ effective[expert_idx].transpose(0, 1)
    mid = mid * soft_mask.to(weight_dtype)
    return (mid @ factors["u_out"].T).float()


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return values.mean()
    w = sample_weight.to(device=values.device, dtype=torch.float32).reshape(-1)
    values = values.reshape(values.shape[0], -1).float().mean(dim=1)
    return (values * w).sum() / w.sum().clamp_min(1e-12)


def rel_mse(y_ref: torch.Tensor, y_new: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    num = _weighted_mean((y_ref.float() - y_new.float()).pow(2), sample_weight)
    den = _weighted_mean(y_ref.float().pow(2), sample_weight).clamp_min(1e-12)
    return num / den


def direct_mse(y_ref: torch.Tensor, y_new: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    return _weighted_mean((y_ref.float() - y_new.float()).pow(2), sample_weight)


def quadratic_mse(
    y_ref: torch.Tensor,
    y_new: torch.Tensor,
    metric: torch.Tensor | None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if metric is None:
        return direct_mse(y_ref, y_new, sample_weight)
    delta = (y_ref.float() - y_new.float())
    metric = metric.to(device=delta.device, dtype=torch.float32)
    # Trace-normalized covariance/Fisher metric: if metric == I, this equals direct MSE.
    q = ((delta @ metric) * delta).sum(dim=1) / max(1, delta.shape[-1])
    if sample_weight is None:
        return q.mean()
    w = sample_weight.to(device=q.device, dtype=torch.float32).reshape(-1)
    return (q * w).sum() / w.sum().clamp_min(1e-12)


def quadratic_rel_mse(
    y_ref: torch.Tensor,
    y_new: torch.Tensor,
    metric: torch.Tensor | None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if metric is None:
        return rel_mse(y_ref, y_new, sample_weight)
    num = quadratic_mse(y_ref, y_new, metric, sample_weight)
    den = quadratic_mse(torch.zeros_like(y_ref), y_ref, metric, sample_weight).clamp_min(1e-12)
    return num / den


def reconstruction_loss(
    y_ref: torch.Tensor,
    y_new: torch.Tensor,
    loss_type: str,
    sample_weight: torch.Tensor | None = None,
    direct_weight: float = 1.0,
    relative_weight: float = 0.05,
    metric: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_type == "relative":
        return rel_mse(y_ref, y_new, sample_weight)
    if loss_type == "direct":
        return direct_mse(y_ref, y_new, sample_weight)
    if loss_type == "mixed":
        return float(direct_weight) * direct_mse(y_ref, y_new, sample_weight) + float(relative_weight) * rel_mse(y_ref, y_new, sample_weight)
    if loss_type == "grad_direct":
        return quadratic_mse(y_ref, y_new, metric, sample_weight)
    if loss_type == "grad_mixed":
        return float(direct_weight) * quadratic_mse(y_ref, y_new, metric, sample_weight) + float(relative_weight) * quadratic_rel_mse(y_ref, y_new, metric, sample_weight)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def inv_sigmoid(x: float) -> float:
    x = min(max(x, 1e-5), 1.0 - 1e-5)
    return math.log(x / (1.0 - x))


def soft_prefix_mask(rho: torch.Tensor, rank_max: int, temp: float, device: torch.device) -> torch.Tensor:
    positions = torch.arange(1, rank_max + 1, device=device, dtype=torch.float32)
    return torch.sigmoid((rho.float() - positions) / temp)


def round_layer_ranks(rhos: Dict[str, float], rank_min: int, rank_max: int, rank_step: int, target_sum: float) -> Dict[str, int]:
    max_chunks = (rank_max - rank_min) // rank_step
    target_chunks = int(round((target_sum - 3 * rank_min) / rank_step))
    target_chunks = max(0, min(3 * max_chunks, target_chunks))
    desired = {}
    for role, rho in rhos.items():
        desired[role] = max(0.0, min(float(max_chunks), (rho - rank_min) / rank_step))
    chunks = {role: int(math.floor(v)) for role, v in desired.items()}
    # adjust to exact layer budget using learned fractional preference.
    while sum(chunks.values()) < target_chunks:
        candidates = [r for r in ROLE_TO_PARAM if chunks[r] < max_chunks]
        if not candidates:
            break
        role = max(candidates, key=lambda r: desired[r] - chunks[r])
        chunks[role] += 1
    while sum(chunks.values()) > target_chunks:
        candidates = [r for r in ROLE_TO_PARAM if chunks[r] > 0]
        if not candidates:
            break
        role = min(candidates, key=lambda r: desired[r] - (chunks[r] - 1))
        chunks[role] -= 1
    return {role: rank_min + chunks[role] * rank_step for role in ROLE_TO_PARAM}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parent
    td = import_file("itcmoe_decompose", root / "itcmoe_decompose.py")
    diag = import_file("itcmoe_diagnostics", root / "itcmoe_diagnostics.py")
    args.original_path = args.model_path

    config = load_json(args.model_path / "config.json")
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])
    hidden = int(config["hidden_size"])
    inter = int(config["moe_intermediate_size"])
    layers = args.layers if args.layers else list(range(num_layers))
    device = torch.device(args.device)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    step_param_delta = pcount("gate", num_experts, hidden, inter, args.rank_min + args.rank_step, td) - pcount("gate", num_experts, hidden, inter, args.rank_min, td)
    coeff_per_rank = step_param_delta / float(args.rank_step)
    const_per_role = pcount("gate", num_experts, hidden, inter, args.rank_min, td) - coeff_per_rank * args.rank_min
    orig_per_role = num_experts * hidden * inter
    target_layer_params = 3 * orig_per_role * (1.0 - args.ratio)
    target_rank_sum = (target_layer_params - 3 * const_per_role) / coeff_per_rank
    target_rank_sum = max(3 * args.rank_min, min(3 * args.rank_max, target_rank_sum))
    print(f"target rank sum per layer: {target_rank_sum:.3f}; avg={target_rank_sum/3:.3f}")

    plan = {"summary": {}, "ranks_by_layer_role": {}, "train_logs": {}}
    if args.resume and args.output_plan.exists():
        plan = load_json(args.output_plan)
        print(f"resuming from {args.output_plan}, done layers={len(plan.get('ranks_by_layer_role', {}))}")

    cov = torch.load(args.cov_cache, map_location="cpu")
    print("collecting routed calibration activations from frozen original model...")
    routed = diag.collect_routed_inputs(args, config, layers)

    for layer_idx in layers:
        if str(layer_idx) in plan.get("ranks_by_layer_role", {}) and args.resume:
            print(f"layer {layer_idx}: skip existing")
            continue
        print(f"=== train layer {layer_idx} ===")
        layer_file = args.model_path / f"layer-{layer_idx}-ep-0-of-1.safetensors"
        with safe_open(layer_file, framework="pt", device="cpu") as f:
            layer_tensors = {key: f.get_tensor(key) for key in f.keys()}

        x = routed[layer_idx]["x"].to(device=device, dtype=torch.float32)
        selected = routed[layer_idx]["experts"].to(device=device)
        route_w = routed[layer_idx]["weights"].to(device=device, dtype=torch.float32)
        flat_experts = selected.reshape(-1)
        flat_weights = route_w.reshape(-1)
        flat_pos = torch.arange(x.shape[0], device=device).unsqueeze(1).expand(-1, selected.shape[1]).reshape(-1)
        routed_x = x.index_select(0, flat_pos)

        # State-aware token weighting for BATUDE rank allocation. We use router
        # top-k confidence as a lightweight proxy for token state on calibration
        # data: high confidence -> hot, low confidence -> cold. This affects only
        # the rank-training loss; the checkpoint still stores one max-rank factorization.
        with torch.no_grad():
            hot_score = route_w.float().max(dim=-1).values
            hot_score = (hot_score - hot_score.min()) / (hot_score.max() - hot_score.min()).clamp_min(1e-12)
            if args.rank_token_state == "hot":
                state_score = hot_score
            elif args.rank_token_state == "cold":
                state_score = 1.0 - hot_score
            else:
                state_score = torch.zeros_like(hot_score)
            token_weight = 1.0 + (float(args.hot_token_weight) - 1.0) * state_score
            flat_token_weight = token_weight.index_select(0, flat_pos)

        original = {}
        for role, param in ROLE_TO_PARAM.items():
            original[role] = torch.stack([layer_tensors[expert_key(layer_idx, e, param)] for e in range(num_experts)], dim=0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            gate_ref = routed_linear_dense(routed_x, flat_experts, original["gate"])
            up_ref = routed_linear_dense(routed_x, flat_experts, original["up"])
            inter_ref = F.silu(gate_ref) * up_ref
            down_ref = routed_linear_dense(inter_ref, flat_experts, original["down"])
            out_ref = torch.zeros((x.shape[0], hidden), device=device, dtype=torch.float32)
            out_ref.index_add_(0, flat_pos, down_ref * flat_weights[:, None])

        factors = {}
        for role, param in ROLE_TO_PARAM.items():
            stacked = original[role].detach().cpu()
            input_cov = None
            output_cov = None
            if args.whiten_type in {"input", "both"}:
                input_cov = cov["input"]["hidden"][layer_idx] if role in {"gate", "up"} else cov["input"]["intermediate"][layer_idx]
            if args.whiten_type in {"output", "both"} and role in set(args.output_whiten_roles or []):
                output_cov = cov["output"][role][layer_idx]
            torch.manual_seed(args.seed + layer_idx * 31 + ROLE_INDEX[role])
            core, u_exp, u_out, u_in = td.decompose_weight_tensor(
                stacked,
                tuple(rank_tuple(role, num_experts, hidden, inter, args.rank_max)),
                input_cov,
                output_cov,
                args,
            )
            core = core.to(device=device, dtype=out_dtype)
            u_exp = u_exp.to(device=device, dtype=out_dtype)
            u_out = u_out.to(device=device, dtype=out_dtype)
            u_in = u_in.to(device=device, dtype=out_dtype)
            with torch.no_grad():
                effective = torch.einsum("er,roi->eoi", u_exp, core).contiguous()
            factors[role] = {"u_out": u_out, "u_in": u_in, "effective": effective}
            del stacked, core, u_exp
            gc.collect(); torch.cuda.empty_cache()

        output_metric = None
        if args.loss_type in {"grad_direct", "grad_mixed"}:
            output_metric = cov.get("output", {}).get("down", {}).get(layer_idx)
            if output_metric is None:
                print(f"warning: layer {layer_idx} has no output/down gradient covariance; falling back to direct MSE metric")
            else:
                output_metric = output_metric.to(device=device, dtype=torch.float32)
                output_metric = 0.5 * (output_metric + output_metric.T)

        init_rank = min(max(target_rank_sum / 3.0, args.rank_min), args.rank_max)
        init_theta = inv_sigmoid((init_rank - args.rank_min) / max(1e-6, args.rank_max - args.rank_min))
        if args.rank_budget_mode == "normalized":
            # Keep the per-layer rank budget fixed and train only the role allocation.
            # With three equal logits this initializes at the same average rank as the
            # penalty formulation, but direct-MSE gradients cannot push all ranks upward.
            theta = torch.nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))
            target_extra_sum = torch.tensor(
                max(0.0, target_rank_sum - 3 * args.rank_min),
                device=device,
                dtype=torch.float32,
            )
        else:
            theta = torch.nn.Parameter(torch.tensor([init_theta, init_theta, init_theta], device=device, dtype=torch.float32))
            target_extra_sum = None
        opt = torch.optim.Adam([theta], lr=args.lr)
        layer_log = []
        best_loss = float("inf")
        best_step = -1
        best_rho_t = None
        for step in range(args.steps):
            frac = step / max(1, args.steps - 1)
            temp = args.temp_start * ((args.temp_end / args.temp_start) ** frac)
            if args.rank_budget_mode == "normalized":
                alloc = torch.softmax(theta, dim=0) * target_extra_sum
                rho_unclamped = args.rank_min + alloc
                cap_excess = torch.relu(rho_unclamped - args.rank_max)
                rho = rho_unclamped.clamp(max=args.rank_max)
            else:
                rho = args.rank_min + (args.rank_max - args.rank_min) * torch.sigmoid(theta)
                cap_excess = torch.zeros_like(rho)
            masks = {role: soft_prefix_mask(rho[ROLE_INDEX[role]], args.rank_max, temp, device) for role in ROLE_TO_PARAM}
            gate = factor_linear("gate", routed_x, flat_experts, factors["gate"], masks["gate"])
            up = factor_linear("up", routed_x, flat_experts, factors["up"], masks["up"])
            inter_new = F.silu(gate) * up
            down = factor_linear("down", inter_new, flat_experts, factors["down"], masks["down"])
            out_new = torch.zeros((x.shape[0], hidden), device=device, dtype=torch.float32)
            out_new.index_add_(0, flat_pos, down * flat_weights[:, None])
            main_loss = reconstruction_loss(out_ref, out_new, args.loss_type, token_weight, args.direct_weight, args.relative_weight, output_metric)
            role_loss = (
                reconstruction_loss(gate_ref, gate, args.loss_type, flat_token_weight, args.direct_weight, args.relative_weight, None)
                + reconstruction_loss(up_ref, up, args.loss_type, flat_token_weight, args.direct_weight, args.relative_weight, None)
                + reconstruction_loss(down_ref, down, args.loss_type, flat_token_weight, args.direct_weight, args.relative_weight, output_metric)
            )
            soft_rank_sum = masks["gate"].sum() + masks["up"].sum() + masks["down"].sum()
            if args.rank_budget_mode == "normalized":
                budget_loss = (cap_excess / max(1.0, float(args.rank_max - args.rank_min))).pow(2).mean()
            else:
                budget_loss = ((soft_rank_sum - target_rank_sum) / target_rank_sum) ** 2
            reconstruction_term = args.reconstruction_loss_scale * (main_loss + args.role_loss_weight * role_loss)
            loss = reconstruction_term + args.budget_weight * budget_loss
            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                best_rho_t = rho.detach().clone()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % max(1, args.steps // 6) == 0 or step == args.steps - 1:
                print(
                    f"layer={layer_idx:02d} step={step:03d} loss={float(loss):.6f} "
                    f"moe={float(main_loss):.6f} role={float(role_loss):.6f} "
                    f"soft_rank_sum={float(soft_rank_sum):.2f} "
                    f"rho={[round(float(v),1) for v in rho.detach().cpu()]}"
                )
            layer_log.append({
                "step": step,
                "loss": loss_value,
                "moe_loss": float(main_loss.detach().cpu()),
                "role_loss": float(role_loss.detach().cpu()),
                "soft_rank_sum": float(soft_rank_sum.detach().cpu()),
                "rho": [float(v) for v in rho.detach().cpu()],
            })

        if best_rho_t is not None:
            final_rho_t = best_rho_t
        elif args.rank_budget_mode == "normalized":
            final_alloc = torch.softmax(theta.detach(), dim=0) * torch.tensor(
                max(0.0, target_rank_sum - 3 * args.rank_min),
                device=device,
                dtype=torch.float32,
            )
            final_rho_t = (args.rank_min + final_alloc).clamp(max=args.rank_max)
        else:
            final_rho_t = args.rank_min + (args.rank_max - args.rank_min) * torch.sigmoid(theta.detach())
        final_rhos = {role: float(final_rho_t[ROLE_INDEX[role]].cpu()) for role in ROLE_TO_PARAM}
        int_ranks = round_layer_ranks(final_rhos, args.rank_min, args.rank_max, args.rank_step, target_rank_sum)
        print(f"layer={layer_idx:02d} final_rho={final_rhos} int_ranks={int_ranks}")
        plan["ranks_by_layer_role"][str(layer_idx)] = {
            role: rank_tuple(role, num_experts, hidden, inter, int_ranks[role]) for role in ROLE_TO_PARAM
        }
        plan["train_logs"][str(layer_idx)] = {
            "final_rho": final_rhos,
            "best_step": best_step,
            "best_loss": best_loss,
            "int_ranks": int_ranks,
            "log_tail": layer_log[-10:],
        }
        role_stats = {}
        for role in ROLE_TO_PARAM:
            vals = [plan["ranks_by_layer_role"][str(l)][role][2 if role in {"gate", "up"} else 1]
                    for l in layers if str(l) in plan["ranks_by_layer_role"]]
            if vals:
                role_stats[role] = {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "values": vals}
        actual_params = 0
        done_groups = 0
        for l in layers:
            if str(l) not in plan["ranks_by_layer_role"]:
                continue
            for role in ROLE_TO_PARAM:
                r = plan["ranks_by_layer_role"][str(l)][role][2 if role in {"gate", "up"} else 1]
                actual_params += pcount(role, num_experts, hidden, inter, r, td)
                done_groups += 1
        plan["summary"] = {
            "policy": "batude_layerwise_frozen_soft_rank_training",
            "description": "Sequential layer-wise training of continuous Tucker rank masks. Original 30B weights and Tucker bases are frozen; only rank logits are optimized with routed MoE calibration loss and a budget constraint/penalty.",
            "ratio_requested": args.ratio,
            "rank_min": args.rank_min,
            "rank_max": args.rank_max,
            "rank_step": args.rank_step,
            "steps": args.steps,
            "calib_samples": args.calib_samples,
            "calib_seq_len": args.calib_seq_len,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "done_layers": len(plan["ranks_by_layer_role"]),
            "loss_type": args.loss_type,
            "direct_weight": args.direct_weight,
            "relative_weight": args.relative_weight,
            "role_loss_weight": args.role_loss_weight,
            "budget_weight": args.budget_weight,
            "loss_metric": "output_loss_gradient_covariance" if args.loss_type in {"grad_direct", "grad_mixed"} else "euclidean",
            "whiten_type": args.whiten_type,
            "output_whiten_mode": args.output_whiten_mode,
            "output_whiten_roles": args.output_whiten_roles,
            "rank_budget_mode": args.rank_budget_mode,
            "reconstruction_loss_scale": args.reconstruction_loss_scale,
            "hot_token_weight": args.hot_token_weight,
            "rank_token_state": args.rank_token_state,
            "role_stats_so_far": role_stats,
        }
        save_json(plan, args.output_plan)

        del layer_tensors, x, selected, route_w, flat_experts, flat_weights, flat_pos, routed_x, token_weight, flat_token_weight
        del original, gate_ref, up_ref, inter_ref, down_ref, out_ref, factors, theta, opt, output_metric
        gc.collect(); torch.cuda.empty_cache()

    # final full summary
    orig_total = len(layers) * 3 * num_experts * hidden * inter
    actual_total = 0
    role_stats = {}
    for role in ROLE_TO_PARAM:
        vals = []
        for l in layers:
            r = plan["ranks_by_layer_role"][str(l)][role][2 if role in {"gate", "up"} else 1]
            vals.append(r)
            actual_total += pcount(role, num_experts, hidden, inter, r, td)
        role_stats[role] = {"min": min(vals), "max": max(vals), "mean": sum(vals)/len(vals), "values": vals}
    plan["summary"].update({
        "done_layers": len(layers),
        "actual_expert_params": actual_total,
        "original_expert_params": orig_total,
        "actual_expert_compression": 1.0 - actual_total / orig_total,
        "role_stats": role_stats,
    })
    save_json(plan, args.output_plan)
    print(f"saved final rank plan: {args.output_plan}")
    print(json.dumps(plan["summary"], indent=2))


if __name__ == "__main__":
    main()
