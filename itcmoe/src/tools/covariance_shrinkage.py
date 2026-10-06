import math
import torch

SCHEMA = "trace_identity_shrinkage_all_roles_v2"


def shrink_covariance(matrix, eta=0.25):
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("shrinkage eta must be finite and in [0, 1]")
    matrix = matrix.float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Covariance must be square")
    if not torch.isfinite(matrix).all():
        raise ValueError("Covariance contains non-finite values")
    matrix = (matrix + matrix.T) * 0.5
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    mean = torch.trace(matrix) / matrix.shape[0]
    if float(mean) < 0:
        raise ValueError("Covariance trace must be nonnegative")
    normalized = matrix / mean if float(mean) > 0 else identity
    return (1.0 - eta) * identity + eta * normalized


def shrinkage_metadata(eta):
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("shrinkage eta must be finite and in [0, 1]")
    return dict(shrinkage_schema=SCHEMA, shrinkage_applied=True,
                shrinkage_eta=float(eta), shrinkage_alpha=float(eta),
                gradient_covariance_trace_normalized=True,
                shrinkage_formula="(1-eta)*I + eta*Sigma_g/(trace(Sigma_g)/d)")


def validate_shrinkage_cache(cov, eta):
    expected = shrinkage_metadata(eta)
    metadata = cov.get("metadata", {})
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Covariance shrinkage configuration mismatch; use a new covariance cache")
