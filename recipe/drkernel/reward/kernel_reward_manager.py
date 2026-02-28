# Copyright 2026 Bytedance Ltd.

import inspect
from typing import Any

import numpy as np

from recipe.drkernel.reward.kernel_reward_fn import compute_kernel_reward
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


def _first_scalar(value: Any, default=None):
    if value is None:
        return default

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0]

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return value[0]

    return value


class DrKernelRewardManager(RewardManagerBase):
    """Reward manager that preserves DrKernel reward fields on top of reward-loop API."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        _ = kwargs
        self.compute_score = compute_score or compute_kernel_reward
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.is_batch_reward_score = False
        try:
            params = set(inspect.signature(self.compute_score).parameters.keys())
            self.is_batch_reward_score = "solution_strs" in params or "ground_truths" in params
        except Exception:
            self.is_batch_reward_score = False

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]

        attention_tail = data_item.batch["attention_mask"][-response_length:]
        valid_response_length = int(attention_tail.sum().item())
        valid_response_length = max(valid_response_length, 1)
        valid_response_ids = response_ids[:valid_response_length]

        data_source = _first_scalar(data_item.non_tensor_batch.get("data_source"), "kernel")
        reward_model = _first_scalar(data_item.non_tensor_batch.get("reward_model"), {})
        if reward_model is None:
            reward_model = {}

        ground_truth = reward_model.get("ground_truth", "")
        entry_point = reward_model.get("entry_point", _first_scalar(data_item.non_tensor_batch.get("entry_point"), ""))
        uuid = _first_scalar(data_item.non_tensor_batch.get("uid"), "")

        extra_info = _first_scalar(data_item.non_tensor_batch.get("extra_info"), {})
        if extra_info is None:
            extra_info = {}
        tool_extra_fields = _first_scalar(data_item.non_tensor_batch.get("tool_extra_fields"), None)
        if isinstance(tool_extra_fields, dict):
            extra_info.update(tool_extra_fields)

        extra_info["entry_point"] = entry_point
        extra_info["uuid"] = uuid

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        is_valid = bool(self.config.reward.get("kernel", {}).get("is_valid", False))

        single_compute_kwargs = {
            "data_source": data_source,
            "solution_str": response_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "entry_point": entry_point,
            "uuid": uuid,
            "reward_config": self.config,
            "is_valid": is_valid,
        }
        batch_compute_kwargs = {
            "solution_strs": [response_str],
            "ground_truths": [ground_truth],
            "entry_points": [entry_point],
            "uuids": [uuid],
            "reward_config": self.config,
            "is_valid": is_valid,
        }

        async def _run_async_compute():
            if self.is_batch_reward_score:
                return await self.compute_score(**batch_compute_kwargs)
            try:
                return await self.compute_score(**single_compute_kwargs)
            except TypeError:
                return await self.compute_score(**batch_compute_kwargs)

        def _run_sync_compute():
            if self.is_batch_reward_score:
                return self.compute_score(**batch_compute_kwargs)
            try:
                return self.compute_score(**single_compute_kwargs)
            except TypeError:
                return self.compute_score(**batch_compute_kwargs)

        if self.is_async_reward_score:
            result = await _run_async_compute()
        else:
            result = await self.loop.run_in_executor(None, _run_sync_compute)

        if isinstance(result, list):
            result = result[0] if result else {"score": 0.0}

        if not isinstance(result, dict):
            result = {"score": float(result)}

        reward_score = float(result.get("score", result.get("reward", 0.0)))
        reward_extra_info = dict(result)
        reward_extra_info.pop("score", None)
        reward_extra_info.pop("reward", None)

        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }
