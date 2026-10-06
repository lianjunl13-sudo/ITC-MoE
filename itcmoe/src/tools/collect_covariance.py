#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from covariance_shrinkage import shrink_covariance, shrinkage_metadata, validate_shrinkage_cache
from transformers import AutoModelForCausalLM, AutoTokenizer


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect itcmoe input/output covariances from GSM8K and MBPP training buckets"
    )
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--calib-jsonl", type=Path, required=True,
                   help="Two-bucket calibration JSONL with 128 GSM8K train and 128 MBPP train records")
    p.add_argument("--calib-balance-mode", type=str, default="two_bucket_equal_interleaved")
    p.add_argument("--sampling-manifest", type=Path, default=None,
                   help="Optional path to write the loader sampling manifest.")
    p.add_argument("--output-cov-cache", type=Path, required=True)
    p.add_argument("--layers", type=int, nargs="*")
    p.add_argument("--calib-samples", type=int, default=64)
    p.add_argument("--calib-seq-len", type=int, default=256)
    p.add_argument("--calib-batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--loss-scale", type=float, default=32.0,
                   help="Multiplies loss before backward to avoid tiny fp16 gradients; covariance is trace-normalized afterwards.")
    p.add_argument("--normalize-trace", action="store_true", default=True)
    p.add_argument("--no-normalize-trace", dest="normalize_trace", action="store_false")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=13037)
    p.add_argument("--shrinkage-eta", type=float, default=0.25)
    p.add_argument("--min-loss-scale", type=float, default=1.0 / 1024.0)
    args = p.parse_args()
    if not (math.isfinite(args.loss_scale) and math.isfinite(args.min_loss_scale)
            and 0 < args.min_loss_scale <= args.loss_scale):
        p.error("Loss scales must be finite and 0 < min-loss-scale <= loss-scale")
    shrinkage_metadata(args.shrinkage_eta)
    if not args.normalize_trace:
        p.error("Shrinkage requires trace normalization")
    return args


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_cov(cov: dict, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(cov, tmp)
    tmp.replace(path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sym_trace_normalize(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mat = (mat.float() + mat.float().T) * 0.5
    dim = mat.shape[0]
    tr = torch.trace(mat).float()
    if not torch.isfinite(tr) or float(tr.abs()) <= eps:
        return torch.eye(dim, dtype=torch.float32)
    mat = mat * (dim / tr.clamp_min(eps))
    return (mat + mat.T) * 0.5



def accumulate_if_finite(totals, batch_matrices, total_counts, batch_counts):
    if not all(torch.isfinite(matrix).all() for matrix in batch_matrices):
        return False
    for total, matrix in zip(totals, batch_matrices):
        total.add_(matrix)
    for i, count in enumerate(batch_counts):
        total_counts[i] += count
    if not all(torch.isfinite(matrix).all() for matrix in totals):
        raise FloatingPointError("Cumulative covariance overflow")
    return True


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    root = Path(__file__).resolve().parent
    loader = import_file(
        "itcmoe_calibration_loader",
        root / "itcmoe_calibration_loader.py",
    )

    config = load_json(args.model_path / "config.json")
    num_layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    layers: List[int] = args.layers if args.layers else list(range(num_layers))

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("Loading the frozen original teacher model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print("Building interleaved GSM8K and MBPP calibration batches...", flush=True)
    batches, sampling_info = loader.build_balanced_batches(
        jsonl_path=str(args.calib_jsonl),
        tokenizer=tokenizer,
        nsamples=int(args.calib_samples),
        seqlen=int(args.calib_seq_len),
        batch_size=int(args.calib_batch_size),
        seed=int(args.seed),
        balance_mode=args.calib_balance_mode,
    )
    if args.sampling_manifest is not None:
        manifest_out = {k: v for k, v in sampling_info.items() if k != "sequence_buckets"}
        Path(args.sampling_manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.sampling_manifest).write_text(
            json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"GSM8K+MBPP batches={len(batches)} seqlen={args.calib_seq_len} "
        f"balance={args.calib_balance_mode} tokens={sampling_info['bucket_token_counts']}",
        flush=True,
    )

    
    if args.output_cov_cache.exists():
        if not args.resume:
            raise FileExistsError(
                f"Covariance cache already exists; refusing to overwrite: {args.output_cov_cache}; "
                "pass --resume to continue"
            )
        cov = torch.load(args.output_cov_cache, map_location="cpu", weights_only=False)
        validate_shrinkage_cache(cov, args.shrinkage_eta)
        old_hash = cov.get("metadata", {}).get("sequence_input_ids_sha256")
        if old_hash is not None and old_hash != sampling_info["sequence_input_ids_sha256"]:
            raise RuntimeError("The cached sampling-sequence hash differs from the current two-bucket sample")
        print(f"Resuming covariance cache: {args.output_cov_cache}", flush=True)
    else:
        cov = {
            "input": {"hidden": {}, "intermediate": {}},
            "output": {role: {} for role in ("gate", "up", "down")},
            "counts": {"input_hidden": {}, "input_intermediate": {}, "output": {role: {} for role in ("gate", "up", "down")}},
            "metadata": {},
        }
    cov["metadata"].update(shrinkage_metadata(args.shrinkage_eta))
    cov["metadata"].update(
        {
            "calibration_jsonl_sha256": sampling_info["calibration_jsonl_sha256"],
            "sequence_input_ids_sha256": sampling_info["sequence_input_ids_sha256"],
            "bucket_effective_token_counts": sampling_info["bucket_token_counts"],
        }
    )

    for layer_idx in layers:
        already_complete = all(
            layer_idx in section
            for section in (
                cov["input"]["hidden"],
                cov["input"]["intermediate"],
                *(cov["output"][role] for role in ("gate", "up", "down")),
                cov["counts"]["input_hidden"],
                cov["counts"]["input_intermediate"],
                *(cov["counts"]["output"][role] for role in ("gate", "up", "down")),
            )
        )
        if args.resume and already_complete:
            matrices = (
                cov["input"]["hidden"][layer_idx],
                cov["input"]["intermediate"][layer_idx],
                *(cov["output"][role][layer_idx] for role in ("gate", "up", "down")),
            )
            if not all(torch.isfinite(matrix).all() for matrix in matrices):
                raise FloatingPointError(f"Existing cache for layer {layer_idx} contains NaN/Inf")
            print(f"Layer {layer_idx} cache is complete; skipping on resume", flush=True)
            continue
        print(f"=== Collecting covariance for layer {layer_idx} ===", flush=True)
        hidden_cov = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
        inter_cov = torch.zeros(intermediate, intermediate, device=device, dtype=torch.float32)
        output_dims = {"gate": intermediate, "up": intermediate, "down": hidden}
        output_cov = {role: torch.zeros(dim, dim, device=device, dtype=torch.float32) for role, dim in output_dims.items()}
        hidden_count = {"n": 0}
        inter_count = {"n": 0}
        output_count = {role: 0 for role in output_dims}
        handles = []
        total_matrices = [torch.zeros_like(m) for m in [hidden_cov, inter_cov, *output_cov.values()]]
        total_counts = [0] * len(total_matrices)
        layer_scales = []
        retry_count = 0
        active_scale = float(args.loss_scale)

        def make_mlp_pre():
            def hook(_m, inp):
                x = inp[0].detach().reshape(-1, hidden).float()
                hidden_cov.add_(x.T @ x)
                hidden_count["n"] += int(x.shape[0])

            return hook

        def make_down_pre():
            def hook(_m, inp):
                x = inp[0].detach().reshape(-1, intermediate).float()
                inter_cov.add_(x.T @ x)
                inter_count["n"] += int(x.shape[0])

            return hook

        def make_output_grad(role):
            def hook(_m, _inp, output):
                y = output if output.requires_grad else output.detach().requires_grad_(True)

                def grad_hook(grad):
                    g = grad.detach().reshape(-1, output_dims[role]).float()
                    g = g / active_scale
                    output_cov[role].add_(g.T @ g)
                    output_count[role] += int(g.shape[0])

                y.register_hook(grad_hook)
                return y

            return hook

        mlp = model.model.layers[layer_idx].mlp
        handles.append(mlp.register_forward_pre_hook(make_mlp_pre()))
        for expert in mlp.experts:
            handles.append(expert.down_proj.register_forward_pre_hook(make_down_pre()))
            for role in output_dims:
                handles.append(getattr(expert, role + "_proj").register_forward_hook(make_output_grad(role)))

        for batch_index, batch in enumerate(tqdm(batches, desc=f"cov L{layer_idx:02d}")):
            batch = batch.to(device)
            attention_mask = torch.ones_like(batch, device=device)
            labels = batch.clone()
            rng_cpu = torch.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            while True:
                for matrix in [hidden_cov, inter_cov, *output_cov.values()]:
                    matrix.zero_()
                hidden_count["n"] = 0
                inter_count["n"] = 0
                for role in output_count:
                    output_count[role] = 0
                torch.set_rng_state(rng_cpu)
                if rng_cuda is not None:
                    torch.cuda.set_rng_state(rng_cuda, device)
                model.zero_grad(set_to_none=True)
                result = model(input_ids=batch, attention_mask=attention_mask, labels=labels, use_cache=False)
                out = result[0] if isinstance(result, tuple) else result
                loss = out.loss
                if loss is None:
                    raise RuntimeError("The model did not return a calibration loss")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite forward loss at layer {layer_idx}, batch {batch_index}")
                (loss * active_scale).backward()
                matrices = [hidden_cov, inter_cov, *output_cov.values()]
                counts = [hidden_count["n"], inter_count["n"], *output_count.values()]
                valid = accumulate_if_finite(total_matrices, matrices, total_counts, counts)
                del result, out, loss, matrices
                if valid:
                    if any(count <= 0 for count in counts):
                        raise RuntimeError(f"Missing statistics at layer {layer_idx}, batch {batch_index}")
                    layer_scales.append(active_scale)
                    break
                if active_scale <= args.min_loss_scale:
                    raise FloatingPointError(f"Non-finite gradients at minimum loss scale, layer {layer_idx}, batch {batch_index}")
                active_scale = max(float(args.min_loss_scale), active_scale / 2.0)
                retry_count += 1
                print(f"Retry layer {layer_idx} batch {batch_index} with loss scale {active_scale}", flush=True)
            del batch, attention_mask, labels
            torch.cuda.empty_cache()

        hidden_cov, inter_cov = total_matrices[:2]
        output_cov = dict(zip(output_dims, total_matrices[2:]))
        hidden_count["n"], inter_count["n"] = total_counts[:2]
        output_count = dict(zip(output_dims, total_counts[2:]))
        cov.setdefault("numerics", {})[layer_idx] = dict(
            initial_loss_scale=float(args.loss_scale), min_loss_scale=float(args.min_loss_scale),
            effective_loss_scales=layer_scales, retries=retry_count, accepted_batches=len(layer_scales))

        for h in handles:
            h.remove()

        nh = max(1, hidden_count["n"])
        ni = max(1, inter_count["n"])
        cov["input"]["hidden"][layer_idx] = (hidden_cov / nh).detach().cpu().contiguous()
        cov["input"]["intermediate"][layer_idx] = (inter_cov / ni).detach().cpu().contiguous()
        for role in output_dims:
            if output_count[role] <= 0:
                raise RuntimeError(f"No output gradients collected for layer {layer_idx} role {role}")
            matrix = (output_cov[role] / output_count[role]).detach().cpu()
            cov["output"][role][layer_idx] = shrink_covariance(matrix, args.shrinkage_eta).contiguous()
            cov["counts"]["output"][role][layer_idx] = output_count[role]
        matrices = [cov["input"][name][layer_idx] for name in ("hidden", "intermediate")]
        matrices.extend(cov["output"][role][layer_idx] for role in output_dims)
        if not all(torch.isfinite(matrix).all() for matrix in matrices):
            raise FloatingPointError(f"Layer {layer_idx} covariance contains NaN/Inf")
        cov["counts"]["input_hidden"][layer_idx] = int(hidden_count["n"])
        cov["counts"]["input_intermediate"][layer_idx] = int(inter_count["n"])

        save_cov(cov, args.output_cov_cache)
        print(
            f"layer {layer_idx}: in_hidden_tok={hidden_count['n']} in_inter_tok={inter_count['n']} output_tok={output_count} saved={args.output_cov_cache}",
            flush=True,
        )
        del hidden_cov, inter_cov, output_cov, matrix, matrices, total_matrices
        gc.collect()
        torch.cuda.empty_cache()

    
    manifest_path = Path(args.calib_jsonl).parent / "manifest.json"
    manifest_sha = sha256_file(str(manifest_path)) if manifest_path.exists() else None
    cov["metadata"].update(
        {
            "calibration_source": "gsm8k_train_plus_mbpp_train_only",
            "calibration_jsonl_path": str(args.calib_jsonl),
            "calibration_jsonl_sha256": sampling_info["calibration_jsonl_sha256"],
            "calibration_manifest_path": str(manifest_path),
            "calibration_manifest_sha256": manifest_sha,
            "calibration_balance_mode": args.calib_balance_mode,
            "calibration_seed": int(args.seed),
            "calibration_samples": int(args.calib_samples),
            "calibration_seq_len": int(args.calib_seq_len),
            "calibration_batch_size": int(args.calib_batch_size),
            "bucket_sequence_counts": sampling_info["bucket_sequence_counts"],
            "bucket_effective_token_counts": sampling_info["bucket_token_counts"],
            "gsm8k_effective_token_count": sampling_info["gsm8k_effective_token_count"],
            "mbpp_effective_token_count": sampling_info["mbpp_effective_token_count"],
            "input_covariance_kind": "forward_activation_data_covariance",
            "output_covariance_kind": "loss_gradient_covariance",
            "output_covariance_roles": ["gate", "up", "down"],
            "gradient_covariance_loss": "causal_lm_labels=input_ids",
            "gradient_covariance_trace_normalized": bool(args.normalize_trace),
            **shrinkage_metadata(args.shrinkage_eta),
            "created_at": datetime.datetime.now().isoformat(),
            "source_script_sha256": sha256_file(str(root / "collect_covariance.py")),
            "loader_script_sha256": sha256_file(str(root / "itcmoe_calibration_loader.py")),
            "sequence_input_ids_sha256": sampling_info["sequence_input_ids_sha256"],
        }
    )
    save_cov(cov, args.output_cov_cache)
    print(f"Saved GSM8K+MBPP covariance cache: {args.output_cov_cache}", flush=True)


if __name__ == "__main__":
    main()
