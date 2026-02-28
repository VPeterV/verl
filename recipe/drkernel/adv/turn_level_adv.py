# Copyright 2026 Bytedance Ltd.

from collections import defaultdict
from typing import Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo import core_algos


def _to_estimator_name(adv_estimator) -> str:
    if hasattr(adv_estimator, "value"):
        return str(adv_estimator.value)
    return str(adv_estimator)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _ensure_response_mask(data: DataProto) -> None:
    if "response_mask" in data.batch.keys():
        return

    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    data.batch["response_mask"] = attention_mask[:, -response_length:]


def _get_loss_mask(data: DataProto) -> torch.Tensor:
    if "loss_mask" in data.batch:
        return data.batch["loss_mask"].bool()
    return torch.ones(data.batch["response_mask"].shape[0], dtype=torch.bool, device=data.batch["response_mask"].device)


def _get_turn_level_cfg(config) -> dict:
    if config is None:
        return {}

    cfg = _cfg_get(config, "turn_level_loss", {})
    if cfg is None:
        return {}
    return cfg


def _get_algo_value(config, key: str, default=None):
    value = _cfg_get(config, key, None)
    if value is not None:
        return value
    return _cfg_get(_get_turn_level_cfg(config), key, default)


def _get_algo_bool(config, key: str, default: bool = False) -> bool:
    return bool(_get_algo_value(config, key, default))


def _is_turn_level_enabled(config) -> bool:
    cfg = _get_turn_level_cfg(config)
    return bool(_cfg_get(cfg, "enable", False))


def _infer_max_turns(data: DataProto, config) -> int:
    cfg = _get_turn_level_cfg(config)
    max_turns = int(_cfg_get(cfg, "max_turns", 0) or _cfg_get(config, "max_turns", 0) or 0)
    if max_turns > 0:
        return max_turns

    if "__num_turns__" in data.non_tensor_batch:
        arr = np.asarray(data.non_tensor_batch["__num_turns__"])
        if arr.size > 0:
            inferred = int(np.max(arr))
            if inferred > 0:
                return inferred

    if "turn_indices" not in data.batch:
        return 1

    turn_indices = data.batch["turn_indices"]
    valid = turn_indices[turn_indices >= 0]
    if valid.numel() == 0:
        return 1
    return int(valid.max().item()) + 1


def _shape_rewards(rewards: torch.Tensor, max_turns: int, gamma: float, unbiased: bool = False) -> torch.Tensor:
    with torch.no_grad():
        rewards_to_use = rewards.reshape(-1, max_turns)
        if unbiased:
            rewards_to_use[:, -1] = rewards_to_use[:, -1] - gamma * rewards_to_use[:, -1]

        shaped_rewards = torch.zeros_like(rewards_to_use)
        for i in range(max_turns):
            if i == 0:
                shaped_rewards[:, i] = rewards_to_use[:, i]
            else:
                shaped_rewards[:, i] = rewards_to_use[:, i] - rewards_to_use[:, i - 1]

        shaped_rewards = shaped_rewards.reshape(-1)
    return shaped_rewards


def _compute_multi_turn_returns(scores: torch.Tensor, gamma: float, max_turns: int) -> torch.Tensor:
    with torch.no_grad():
        shaped_scores = scores.reshape(-1, max_turns)
        returns = torch.zeros_like(shaped_scores)

        for i in reversed(range(max_turns)):
            if i == max_turns - 1:
                returns[:, i] = shaped_scores[:, i]
            else:
                returns[:, i] = shaped_scores[:, i] + gamma * returns[:, i + 1]

        returns = returns.reshape(-1)
    return returns


def _compute_multi_turn_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    turn_indices: torch.Tensor,
    index: np.ndarray,
    max_turns: int,
    gamma: float,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    returns = _compute_multi_turn_returns(scores, gamma, max_turns)

    id2return = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = returns.shape[0]
        advantages = torch.zeros_like(returns)

        for i in range(bsz):
            if turn_indices[i].item() == -1 or not loss_mask[i]:
                continue
            idx = (index[i], turn_indices[i].item())
            id2return[idx].append(returns[i])

        for idx in id2return:
            if len(id2return[idx]) == 1:
                id2mean[idx] = returns.new_tensor(0.0)
            elif len(id2return[idx]) > 1:
                id2mean[idx] = torch.stack(id2return[idx]).mean()
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if turn_indices[i].item() == -1 or not loss_mask[i]:
                continue
            idx = (index[i], turn_indices[i].item())
            response_num = len(id2return[idx])
            if response_num > 1:
                advantages[i] = returns[i] * response_num / (response_num - 1) - id2mean[idx] * response_num / (
                    response_num - 1
                )
            else:
                advantages[i] = returns[i]

        advantages = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
        returns = returns.unsqueeze(-1).tile([1, response_length]) * response_mask

    return advantages, returns


def _check_uniform_within_sequences(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    epsilon: float = 1e-6,
) -> bool:
    batch_size = advantages.shape[0]

    for i in range(batch_size):
        seq_mask = response_mask[i].bool()
        valid_seq_length = seq_mask.sum().item()
        if valid_seq_length <= 1:
            continue

        seq_advantages = advantages[i][seq_mask]
        seq_variance = torch.var(seq_advantages).item()
        if seq_variance > epsilon:
            return False

    return True


def _apply_batch_standardization(advantages: torch.Tensor, response_mask: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    with torch.no_grad():
        if advantages.numel() == 0 or response_mask.sum() == 0:
            return advantages

        is_uniform_within_sequences = _check_uniform_within_sequences(advantages, response_mask, epsilon)

        if is_uniform_within_sequences:
            sequence_lengths = response_mask.sum(dim=-1)
            valid_sequences = sequence_lengths > 0
            if not valid_sequences.any():
                return advantages

            sequence_advantages = verl_F.masked_mean(advantages, response_mask, axis=-1)
            valid_seq_advantages = sequence_advantages[valid_sequences]

            seq_mean = torch.mean(valid_seq_advantages)
            seq_std = torch.std(valid_seq_advantages)
            if seq_std < epsilon:
                return advantages

            standardized_seq_advantages = (sequence_advantages - seq_mean) / seq_std
            standardized_advantages = standardized_seq_advantages.unsqueeze(-1) * response_mask
            return standardized_advantages

        valid_mask = response_mask.bool()
        valid_advantages = advantages[valid_mask]
        token_mean = torch.mean(valid_advantages)
        token_std = torch.std(valid_advantages)
        if token_std < epsilon:
            return advantages

        standardized_advantages = torch.zeros_like(advantages)
        standardized_advantages[valid_mask] = (valid_advantages - token_mean) / token_std
        return standardized_advantages


def maybe_apply_final_reward(data: DataProto, config) -> None:
    """Mimic DrKernel `use_final_reward=True`: only keep rewards on last turn."""
    if not _get_algo_bool(config, "use_final_reward", True):
        return

    max_turns = _infer_max_turns(data, config)
    if max_turns <= 1:
        return

    if "token_level_scores" not in data.batch and "token_level_rewards" not in data.batch:
        return

    for key in ("token_level_scores", "token_level_rewards"):
        if key not in data.batch:
            continue
        tensor = data.batch[key]
        if tensor.ndim != 2:
            continue
        if tensor.shape[0] % max_turns != 0:
            continue
        reshaped = tensor.reshape(-1, max_turns, tensor.shape[-1]).clone()
        reshaped[:, :-1, :] = 0.0
        data.batch[key] = reshaped.reshape(-1, tensor.shape[-1])


def maybe_apply_loss_mask_to_masks(data: DataProto) -> None:
    """Mimic DrKernel `apply_loss_mask_to_masks`."""
    if "loss_mask" not in data.batch:
        return
    if "response_mask" not in data.batch or "attention_mask" not in data.batch:
        return

    loss_mask = data.batch["loss_mask"]
    if loss_mask.dim() != 1:
        raise ValueError(f"Expected `loss_mask` to be 1-D, got shape {tuple(loss_mask.shape)}")
    mask = loss_mask.unsqueeze(1)

    data.batch["response_mask"] = data.batch["response_mask"] * mask.to(data.batch["response_mask"].dtype)
    data.batch["attention_mask"] = data.batch["attention_mask"] * mask.to(data.batch["attention_mask"].dtype)


def maybe_apply_loss_mask_to_rewards(data: DataProto) -> None:
    """Mimic DrKernel `apply_loss_mask_to_rewards`."""
    if "loss_mask" not in data.batch:
        return

    loss_mask = data.batch["loss_mask"]
    if loss_mask.dim() != 1:
        raise ValueError(f"Expected `loss_mask` to be 1-D, got shape {tuple(loss_mask.shape)}")
    mask = loss_mask.unsqueeze(1)

    for key in ("token_level_scores", "token_level_rewards"):
        if key not in data.batch:
            continue
        tensor = data.batch[key]
        data.batch[key] = tensor * mask.to(tensor.dtype)


def _maybe_apply_extra_reduction(data: DataProto, config) -> None:
    use_multi_prompt_mvu = _get_algo_bool(config, "use_multi_prompt_mvu", False)
    if not use_multi_prompt_mvu:
        return

    try:
        from verl_patch.trainer.code.ppo.variance_reduction import apply_variance_reduction
    except Exception as exc:
        raise RuntimeError(
            "`algorithm.use_multi_prompt_mvu=True` requires `verl_patch` variance_reduction module."
        ) from exc

    modified_advantages, variance_info = apply_variance_reduction(
        data=data,
        use_batch_reweighting=False,
        use_multi_prompt_mvu=True,
    )
    data.batch["advantages"] = modified_advantages
    if variance_info is not None:
        data.meta_info["variance_reduction_info"] = variance_info


def _maybe_apply_batch_std(data: DataProto, config) -> None:
    if not _get_algo_bool(config, "batch_std", False):
        return

    if "attention_mask" in data.batch:
        response_length = data.batch["advantages"].shape[-1]
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    else:
        response_mask = data.batch["response_mask"]
    data.batch["advantages"] = _apply_batch_standardization(data.batch["advantages"], response_mask)


def maybe_compute_advantage_by_last_turn(
    data: DataProto,
    adv_estimator,
    gamma: float,
    lam: float,
    num_repeat: int,
    norm_adv_by_std_in_grpo: bool,
    config,
    fallback_compute_advantage,
) -> Optional[DataProto]:
    """Emulate DrKernel `adv_by_last_turn` behavior: compute on final turn then broadcast."""
    if "turn_indices" not in data.batch:
        return None

    adv_by_last_turn = _get_algo_bool(config, "adv_by_last_turn", True)
    if not adv_by_last_turn:
        return None

    _ensure_response_mask(data)

    batch_size = data.batch["response_mask"].shape[0]
    max_turns = _infer_max_turns(data, config)
    if max_turns <= 1 or batch_size % max_turns != 0:
        return None

    last_turn_indices = torch.arange(max_turns - 1, batch_size, max_turns, dtype=torch.long)
    last_turn_batch = data.select_idxs(last_turn_indices)

    last_turn_batch = fallback_compute_advantage(
        data=last_turn_batch,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    full_response_mask = data.batch["response_mask"]
    seq_count = batch_size // max_turns

    for key in ("advantages", "returns"):
        if key not in last_turn_batch.batch:
            continue

        last_turn_tensor = last_turn_batch.batch[key]
        if last_turn_tensor.ndim != 2:
            continue

        if "response_mask" in last_turn_batch.batch:
            mask = last_turn_batch.batch["response_mask"].to(last_turn_tensor.dtype)
            max_diff = (((last_turn_tensor - last_turn_tensor[:, :1]) * mask).abs()).max().item()
            if max_diff > 1e-10:
                raise ValueError(
                    f"Cannot broadcast `{key}` from last turn because token values differ within sequence."
                )

        scalar_values = last_turn_tensor[:, 0]
        expanded_mask = full_response_mask.reshape(seq_count, max_turns, -1).to(last_turn_tensor.dtype)
        broadcast_tensor = (expanded_mask * scalar_values.view(seq_count, 1, 1)).reshape(batch_size, -1)
        data.batch[key] = broadcast_tensor

    if "advantages" in data.batch and "returns" in data.batch:
        with torch.no_grad():
            data.batch["turn_level_advantages"] = verl_F.masked_mean(data.batch["advantages"], full_response_mask, axis=-1)
            data.batch["turn_level_returns"] = verl_F.masked_mean(data.batch["returns"], full_response_mask, axis=-1)

    data.meta_info.update(last_turn_batch.meta_info)
    return data


def maybe_compute_turn_level_advantage(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> Optional[DataProto]:
    """DrKernel-equivalent multi-turn advantage implementation for selected estimators."""
    _ = num_repeat
    _ = norm_adv_by_std_in_grpo

    if not _is_turn_level_enabled(config):
        return None

    if "turn_indices" not in data.batch:
        return None

    if _get_algo_bool(config, "adv_by_last_turn", True):
        # Handled by maybe_compute_advantage_by_last_turn.
        return None

    _ensure_response_mask(data)

    estimator_name = _to_estimator_name(adv_estimator)
    enabled_estimators = set(
        _cfg_get(
            _get_turn_level_cfg(config),
            "estimators",
            ["trloo", "erloo", "erloo_norm", "grpo", "turn_independent_grpo", "reinforce", "egae"],
        )
    )
    if estimator_name not in enabled_estimators:
        return None

    max_turns = _infer_max_turns(data, config)
    if max_turns <= 0 or data.batch["response_mask"].shape[0] % max_turns != 0:
        return None

    reward_shaping = _get_algo_bool(config, "reward_shaping", False)
    unbiased_shaping = _get_algo_bool(config, "unbiased_shaping", False)
    loss_mask = _get_loss_mask(data)

    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    response_length = token_level_rewards.shape[-1]

    uid = np.asarray(data.non_tensor_batch.get("uid", np.arange(token_level_rewards.shape[0], dtype=object)), dtype=object)
    turn_indices = data.batch["turn_indices"]

    if estimator_name == "grpo":
        turn_rewards = token_level_rewards.sum(dim=-1)
        if reward_shaping:
            turn_rewards = _shape_rewards(turn_rewards, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = _compute_multi_turn_returns(turn_rewards, gamma, max_turns)

        with torch.no_grad():
            prompt_groups = defaultdict(list)
            batch_size = turn_scores.shape[0]
            for i in range(batch_size):
                if turn_indices[i].item() == -1 or not loss_mask[i]:
                    continue
                prompt_groups[(uid[i], turn_indices[i].item())].append(i)

            baselines = torch.zeros_like(turn_scores)
            for _, trajectory_indices in prompt_groups.items():
                traj_idx = torch.tensor(trajectory_indices, device=turn_scores.device)
                if len(trajectory_indices) == 1:
                    baselines[traj_idx[0]] = 0.0
                else:
                    baselines[traj_idx] = turn_scores[traj_idx].mean().to(baselines.dtype)

            advantages = turn_scores - baselines

        data.batch["advantages"] = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
        data.batch["returns"] = turn_scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        data.meta_info["compare_mtrloo"] = True

    elif estimator_name == "turn_independent_grpo":
        turn_rewards = token_level_rewards.sum(dim=-1)
        if reward_shaping:
            turn_rewards = _shape_rewards(turn_rewards, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = turn_rewards

        with torch.no_grad():
            prompt_groups = defaultdict(list)
            batch_size = turn_scores.shape[0]
            for i in range(batch_size):
                if turn_indices[i].item() == -1 or not loss_mask[i]:
                    continue
                prompt_groups[uid[i]].append(i)

            baselines = torch.zeros_like(turn_scores)
            for _, trajectory_indices in prompt_groups.items():
                traj_idx = torch.tensor(trajectory_indices, device=turn_scores.device)
                if len(trajectory_indices) == 1:
                    baselines[traj_idx[0]] = 0.0
                else:
                    baselines[traj_idx] = turn_scores[traj_idx].mean().to(baselines.dtype)

            advantages = turn_scores - baselines

        data.batch["advantages"] = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
        data.batch["returns"] = turn_scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        data.meta_info["compare_mtrloo"] = True

    elif estimator_name == "egae":
        turn_rewards = token_level_rewards.sum(dim=-1)
        if reward_shaping:
            turn_rewards = _shape_rewards(turn_rewards, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = turn_rewards

        turn_scores = _compute_multi_turn_returns(turn_rewards, gamma, max_turns)

        response_lengths = response_mask.sum(dim=-1).to(torch.long)
        valid_rows = response_lengths > 0
        row_indices = torch.arange(len(response_lengths), device=token_level_rewards.device)[valid_rows]
        col_indices = response_lengths[valid_rows] - 1

        token_level_rewards[row_indices, col_indices] = turn_scores[valid_rows]

        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=data.batch["values"],
            response_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    elif estimator_name == "reinforce":
        scores = token_level_rewards.sum(dim=-1)
        if reward_shaping:
            scores = _shape_rewards(scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = scores

        returns = _compute_multi_turn_returns(scores, gamma, max_turns)

        with torch.no_grad():
            valid_returns = returns[loss_mask]
            if len(valid_returns) == 1:
                return_mean = returns.new_tensor(0.0)
                return_std = returns.new_tensor(1.0)
            elif len(valid_returns) > 1:
                return_mean = torch.mean(valid_returns)
                return_std = torch.std(valid_returns)
            else:
                raise ValueError("No valid returns to compute advantages.")

            advantages = (returns - return_mean) / (return_std + 1e-6)
            data.batch["advantages"] = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
            data.batch["returns"] = returns.unsqueeze(-1).tile([1, response_length]) * response_mask
            data.meta_info["compare_mtrloo"] = True

    elif estimator_name == "erloo":
        scores = token_level_rewards.sum(dim=-1)
        if reward_shaping:
            scores = _shape_rewards(scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = scores

        returns = _compute_multi_turn_returns(scores, gamma, max_turns)

        id2return = defaultdict(list)
        id2mean = {}

        with torch.no_grad():
            bsz = returns.shape[0]
            advantages = torch.zeros_like(returns)

            for i in range(bsz):
                if not loss_mask[i]:
                    continue
                id2return[uid[i]].append(returns[i])

            for idx in id2return:
                if len(id2return[idx]) == 1:
                    id2mean[idx] = returns.new_tensor(0.0)
                elif len(id2return[idx]) > 1:
                    id2mean[idx] = torch.stack(id2return[idx]).mean()
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            for i in range(bsz):
                if not loss_mask[i]:
                    continue
                idx = uid[i]
                response_num = len(id2return[idx])
                if response_num > 1:
                    advantages[i] = returns[i] * response_num / (response_num - 1) - id2mean[idx] * response_num / (
                        response_num - 1
                    )
                else:
                    advantages[i] = returns[i]

            data.batch["advantages"] = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
            data.batch["returns"] = returns.unsqueeze(-1).tile([1, response_length]) * response_mask
            data.meta_info["compare_mtrloo"] = True

    elif estimator_name == "erloo_norm":
        scores = token_level_rewards.sum(dim=-1)
        if reward_shaping:
            scores = _shape_rewards(scores, max_turns, gamma, unbiased_shaping)
            data.batch["shaped_turn_rewards"] = scores

        returns = _compute_multi_turn_returns(scores, gamma, max_turns)

        with torch.no_grad():
            prompt_groups = defaultdict(list)
            batch_size = returns.shape[0]
            for i in range(batch_size):
                if not loss_mask[i]:
                    continue
                prompt_groups[uid[i]].append(i)

            advantages = torch.zeros_like(returns)
            for _, prompt_indices in prompt_groups.items():
                N = len(prompt_indices)
                group_returns = returns[prompt_indices]

                if N == 1:
                    loo_mean = returns.new_tensor(0.0)
                    loo_std = returns.new_tensor(1.0)
                elif N == 2:
                    loo_mean = group_returns.flip(dims=[-1])
                    loo_std = returns.new_tensor(1.0)
                else:
                    total_sum = group_returns.sum()
                    loo_mean = (total_sum - group_returns) / (N - 1)

                    group_returns_repeat = group_returns.unsqueeze(0).repeat(N, 1)
                    loo_mask = torch.ones_like(group_returns_repeat, dtype=torch.bool)
                    loo_mask[torch.arange(N), torch.arange(N)] = False
                    loo_group_returns = group_returns_repeat[loo_mask].reshape(-1, N - 1)
                    loo_std = torch.std(loo_group_returns, dim=-1)

                advantages[prompt_indices] = (group_returns - loo_mean) / (loo_std + 1e-8)

            data.batch["advantages"] = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask
            data.batch["returns"] = returns.unsqueeze(-1).tile([1, response_length]) * response_mask

    elif estimator_name == "trloo":
        if reward_shaping:
            raise NotImplementedError("Reward shaping is not supported for trloo.")

        advantages, returns = _compute_multi_turn_rloo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            loss_mask=loss_mask,
            turn_indices=turn_indices,
            index=uid,
            max_turns=max_turns,
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.meta_info["compare_mtrloo"] = True

    else:
        return None

    _maybe_apply_extra_reduction(data, config)
    _maybe_apply_batch_std(data, config)

    with torch.no_grad():
        data.batch["turn_level_advantages"] = verl_F.masked_mean(data.batch["advantages"], response_mask, axis=-1)
        data.batch["turn_level_returns"] = verl_F.masked_mean(data.batch["returns"], response_mask, axis=-1)

    return data
