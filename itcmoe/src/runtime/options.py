"""Control operators, candidate restriction and Hot compensation independently without changing weights."""
def configure_runtime(model, operator=True, hot=True, candidate=True, candidate_size=48):
    blocks = [m for m in model.modules() if m.__class__.__name__ == 'SDARMoeSparseMoeBlock']
    if not blocks:
        raise RuntimeError('No SDAR MoE layers found')
    compressed = 0
    for block in blocks:
        if not hasattr(block, 'shared_projection_enabled'):
            raise RuntimeError('The model uses an older runtime; run scripts/install_runtime.py first')
        if not block.top_k <= candidate_size <= block.num_experts:
            raise ValueError('Candidate size is outside the valid range')
        block.candidate_enabled = bool(candidate)
        block.candidate_size = candidate_size
        block.shared_projection_enabled = bool(operator)
        if not block.td_moe_enabled:
            continue
        compressed += 1
        for projection in (block.gate_up_proj.gate_proj, block.gate_up_proj.up_proj, block.down_proj):
            projection.operator = 'hybrid_down_effective' if operator else 'plain'
            projection.kernel = 'triton' if operator else 'torch'
            projection.hot_svd_disabled = not hot
    return dict(blocks=len(blocks), compressed_blocks=compressed, operator=operator,
                hot=hot, candidate=candidate, candidate_size=candidate_size)
