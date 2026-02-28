# Copyright 2026 Bytedance Ltd.

from typing import Dict, Optional, Tuple

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def compute_rollout_rejection_mask(
    coverage_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    correctness: torch.Tensor,
    max_turns: int = 1,
    coverage_rs: str = "turn",
    coverage_rs_threshold: Optional[float] = None,
    coverage_rs_factor: Optional[float] = None,
    speedup: Optional[torch.Tensor] = None,
    speedup_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute response-mask rejection from coverage statistics."""
    device = response_mask.device
    batch_size = response_mask.shape[0] // max_turns
    seq_length = response_mask.shape[1]

    if coverage_ratio.dim() == 1:
        coverage_ratio = coverage_ratio.reshape(batch_size, max_turns)

    correctness_2d = correctness.reshape(batch_size, max_turns)

    if coverage_rs == "geometric":
        valid_turn_mask = response_mask.reshape(batch_size, max_turns, seq_length).any(dim=-1)
        log_coverage = torch.log(coverage_ratio.clamp(min=1e-8))
        masked_log = log_coverage.masked_fill(~valid_turn_mask, 0)
        num_valid = valid_turn_mask.sum(dim=1, keepdim=True).clamp(min=1)
        geometric = torch.exp(masked_log.sum(dim=1, keepdim=True) / num_valid)
        rollout_is_weights = geometric.expand(-1, max_turns)
    else:
        rollout_is_weights = coverage_ratio

    rollout_is_weights_speedup = None
    if speedup is not None and speedup_threshold is not None:
        if speedup.dim() == 1:
            rollout_is_weights_speedup = speedup.reshape(batch_size, max_turns)
        else:
            rollout_is_weights_speedup = speedup

    if coverage_rs_threshold is None:
        coverage_rs_threshold = 0.3

    if coverage_rs_factor is None or coverage_rs_factor == 0:
        mask_prob = (rollout_is_weights >= coverage_rs_threshold).float()
    else:
        mask_prob = (rollout_is_weights - coverage_rs_threshold) / coverage_rs_factor
        mask_prob = mask_prob.clamp(min=0, max=1)

    # Only correct samples are eligible for rejection.
    mask_prob = torch.where(correctness_2d, mask_prob, torch.ones_like(mask_prob))

    if rollout_is_weights_speedup is not None:
        speedup_mask = (rollout_is_weights_speedup >= speedup_threshold).float()
        mask_prob = torch.maximum(mask_prob, speedup_mask)

    turn_level_mask = torch.bernoulli(mask_prob).float()

    token_level_mask = turn_level_mask.reshape(-1, 1).expand(-1, seq_length)
    modified_response_mask = response_mask * token_level_mask

    metrics: Dict[str, float] = {}
    valid_mask = response_mask > 0
    if valid_mask.any():
        metrics["coverage/coverage_rs_masked_fraction"] = verl_F.masked_mean(1 - token_level_mask, valid_mask).item()

        if coverage_rs == "turn":
            seq_has_masked = (turn_level_mask.reshape(batch_size, max_turns) == 0).any(dim=1)
            metrics["coverage/coverage_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
        else:
            first_turn_mask = turn_level_mask.reshape(batch_size, max_turns)[:, 0]
            metrics["coverage/coverage_rs_seq_masked_fraction"] = (first_turn_mask == 0).float().mean().item()

        correct_mask = correctness.bool() & valid_mask.any(dim=-1)
        if correct_mask.any():
            correct_token_mask = correct_mask.unsqueeze(-1).expand_as(valid_mask) & valid_mask
            metrics["coverage/coverage_rs_correct_only_masked_fraction"] = verl_F.masked_mean(
                1 - token_level_mask, correct_token_mask
            ).item()

        metrics["coverage/coverage_rs_mean_coverage"] = rollout_is_weights.mean().item()
        metrics["coverage/coverage_rs_min_coverage"] = rollout_is_weights.min().item()
        metrics["coverage/coverage_rs_max_coverage"] = rollout_is_weights.max().item()

    return modified_response_mask, metrics


def compute_coverage_rejection_mask(
    time_coverage: torch.Tensor,
    num_coverage: torch.Tensor,
    response_mask: torch.Tensor,
    correctness: torch.Tensor,
    max_turns: int = 1,
    coverage_rs: str = "turn",
    coverage_rs_key: str = "time_coverage",
    coverage_rs_threshold: Optional[float] = 0.3,
    coverage_rs_factor: Optional[float] = 0.1,
    speedup: Optional[torch.Tensor] = None,
    speedup_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Select coverage metric and apply rejection sampling."""
    if coverage_rs_key not in {"time_coverage", "num_coverage"}:
        raise ValueError(f"Invalid coverage_rs_key: {coverage_rs_key}")
    if coverage_rs not in {"turn", "geometric"}:
        raise ValueError(f"Invalid coverage_rs: {coverage_rs}")

    if speedup_threshold is not None and speedup_threshold < 0:
        speedup_threshold = None

    coverage_ratio = time_coverage if coverage_rs_key == "time_coverage" else num_coverage

    modified_response_mask, metrics = compute_rollout_rejection_mask(
        coverage_ratio=coverage_ratio,
        response_mask=response_mask,
        correctness=correctness,
        max_turns=max_turns,
        coverage_rs=coverage_rs,
        coverage_rs_threshold=coverage_rs_threshold,
        coverage_rs_factor=coverage_rs_factor,
        speedup=speedup,
        speedup_threshold=speedup_threshold,
    )
    return modified_response_mask, metrics


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _get_kernel_cfg(config):
    reward_cfg = _cfg_get(config, "reward", None)
    if reward_cfg is not None:
        kernel_cfg = _cfg_get(reward_cfg, "kernel", None)
        if kernel_cfg is not None:
            return kernel_cfg

    legacy_reward_model = _cfg_get(config, "reward_model", None)
    if legacy_reward_model is not None:
        return legacy_reward_model

    return {}


def _extract_metric_array(batch, key: str, default: float, alias_key: str | None = None) -> np.ndarray:
    batch_size = int(batch.batch["response_mask"].shape[0])

    if key in batch.non_tensor_batch:
        arr = np.asarray(batch.non_tensor_batch[key])
        if arr.shape[0] == batch_size:
            return arr
    if alias_key is not None and alias_key in batch.non_tensor_batch:
        arr = np.asarray(batch.non_tensor_batch[alias_key])
        if arr.shape[0] == batch_size:
            return arr

    if "reward_extra_info" in batch.non_tensor_batch:
        info_arr = np.asarray(batch.non_tensor_batch["reward_extra_info"], dtype=object)
        if info_arr.shape[0] == batch_size:
            vals = []
            for item in info_arr:
                if isinstance(item, dict):
                    value = item.get(key, None)
                    if value is None and alias_key is not None:
                        value = item.get(alias_key, None)
                    if value is None:
                        value = default
                    vals.append(value)
                else:
                    vals.append(default)
            return np.asarray(vals)

    return np.full((batch_size,), default)


def _infer_max_turns(batch, config) -> int:
    batch_size = int(batch.batch["response_mask"].shape[0])
    algorithm_cfg = _cfg_get(config, "algorithm", {})
    max_turns = int(
        _cfg_get(_cfg_get(algorithm_cfg, "turn_level_loss", {}), "max_turns", 0) or _cfg_get(algorithm_cfg, "max_turns", 0) or 0
    )
    if max_turns > 0 and batch_size % max_turns == 0:
        return max_turns

    if "turn_indices" in batch.batch:
        turn_indices = batch.batch["turn_indices"]
        valid = turn_indices[turn_indices >= 0]
        if valid.numel() > 0:
            inferred = int(valid.max().item()) + 1
            if inferred > 0 and batch_size % inferred == 0:
                return inferred

    if "__num_turns__" in batch.non_tensor_batch:
        arr = np.asarray(batch.non_tensor_batch["__num_turns__"])
        if arr.size > 0:
            inferred = int(np.max(arr))
            if inferred > 0 and batch_size % inferred == 0:
                return inferred

    return 1


def apply_coverage_rejection_to_batch(batch, trainer_config):
    """Apply DrKernel coverage-based rejection sampling on top of native rollout correction."""
    kernel_cfg = _get_kernel_cfg(trainer_config)
    coverage_rs = _cfg_get(kernel_cfg, "coverage_rs", None)
    if coverage_rs in (None, "null", "None"):
        return batch, {}

    response_mask = batch.batch["response_mask"]
    device = response_mask.device
    max_turns = _infer_max_turns(batch, trainer_config)

    time_coverage = torch.tensor(_extract_metric_array(batch, "time_coverage", 0.0), device=device, dtype=torch.float32)
    num_coverage = torch.tensor(_extract_metric_array(batch, "num_coverage", 0.0), device=device, dtype=torch.float32)

    correctness = torch.tensor(_extract_metric_array(batch, "correctness", 0.0), device=device, dtype=torch.bool)
    is_decoy = torch.tensor(
        _extract_metric_array(batch, "is_decoy_kernel", 0.0, alias_key="decoy_kernel"),
        device=device,
        dtype=torch.bool,
    )
    correctness = correctness & (~is_decoy)

    performance = torch.tensor(
        _extract_metric_array(batch, "performance", 0.0, alias_key="speedup"),
        device=device,
        dtype=torch.float32,
    )

    speedup_threshold = _cfg_get(kernel_cfg, "speedup_threshold", None)
    if speedup_threshold in (None, "null", "None", ""):
        speedup_threshold = None
    else:
        speedup_threshold = float(speedup_threshold)

    modified_response_mask, metrics = compute_coverage_rejection_mask(
        time_coverage=time_coverage,
        num_coverage=num_coverage,
        response_mask=response_mask,
        correctness=correctness,
        max_turns=max_turns,
        coverage_rs=str(coverage_rs),
        coverage_rs_key=str(_cfg_get(kernel_cfg, "coverage_rs_key", "time_coverage")),
        coverage_rs_threshold=float(_cfg_get(kernel_cfg, "coverage_rs_threshold", 0.3)),
        coverage_rs_factor=float(_cfg_get(kernel_cfg, "coverage_rs_factor", 0.1)),
        speedup=performance,
        speedup_threshold=speedup_threshold,
    )

    batch.batch["response_mask"] = modified_response_mask
    prefixed_metrics = {f"rollout_corr/{k}": v for k, v in metrics.items()}
    return batch, prefixed_metrics
