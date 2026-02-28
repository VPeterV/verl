# Copyright 2026 Bytedance Ltd.

from contextlib import contextmanager
from typing import Iterator

from recipe.drkernel.adv.turn_level_adv import (
    maybe_apply_final_reward,
    maybe_apply_loss_mask_to_masks,
    maybe_apply_loss_mask_to_rewards,
    maybe_compute_advantage_by_last_turn,
    maybe_compute_turn_level_advantage,
)
from recipe.drkernel.reward.coverage_rs import apply_coverage_rejection_to_batch
from verl.trainer.ppo import ray_trainer as base_ray_trainer
from verl.trainer.ppo import rollout_corr_helper


@contextmanager
def _patch_compute_advantage(trainer_config) -> Iterator[None]:
    original_compute_advantage = base_ray_trainer.compute_advantage
    algorithm_config = trainer_config.algorithm

    def wrapped_compute_advantage(
        data,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        algo_cfg = config if config is not None else algorithm_config

        maybe_apply_loss_mask_to_masks(data)
        maybe_apply_final_reward(data, algo_cfg)
        maybe_apply_loss_mask_to_rewards(data)

        last_turn_data = maybe_compute_advantage_by_last_turn(
            data=data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=algo_cfg,
            fallback_compute_advantage=original_compute_advantage,
        )
        if last_turn_data is not None:
            return last_turn_data

        turn_level_data = maybe_compute_turn_level_advantage(
            data=data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=algo_cfg,
        )
        if turn_level_data is not None:
            return turn_level_data

        return original_compute_advantage(
            data=data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=algo_cfg,
        )

    base_ray_trainer.compute_advantage = wrapped_compute_advantage
    try:
        yield
    finally:
        base_ray_trainer.compute_advantage = original_compute_advantage


@contextmanager
def _patch_rollout_correction(trainer_config) -> Iterator[None]:
    original_rollout_correction = rollout_corr_helper.compute_rollout_correction_and_add_to_batch

    def wrapped_rollout_correction(batch, rollout_corr_config):
        maybe_apply_loss_mask_to_masks(batch)
        batch, metrics = original_rollout_correction(batch, rollout_corr_config)
        batch, coverage_metrics = apply_coverage_rejection_to_batch(batch, trainer_config)
        if coverage_metrics:
            metrics.update(coverage_metrics)
        return batch, metrics

    rollout_corr_helper.compute_rollout_correction_and_add_to_batch = wrapped_rollout_correction
    try:
        yield
    finally:
        rollout_corr_helper.compute_rollout_correction_and_add_to_batch = original_rollout_correction


@contextmanager
def _patch_apply_kl_penalty(trainer_config) -> Iterator[None]:
    original_apply_kl_penalty = base_ray_trainer.apply_kl_penalty
    algorithm_config = trainer_config.algorithm

    def wrapped_apply_kl_penalty(data, kl_ctrl, kl_penalty="kl"):
        maybe_apply_final_reward(data, algorithm_config)
        maybe_apply_loss_mask_to_rewards(data)
        return original_apply_kl_penalty(data, kl_ctrl=kl_ctrl, kl_penalty=kl_penalty)

    base_ray_trainer.apply_kl_penalty = wrapped_apply_kl_penalty
    try:
        yield
    finally:
        base_ray_trainer.apply_kl_penalty = original_apply_kl_penalty


class DrKernelRayPPOTrainer(base_ray_trainer.RayPPOTrainer):
    """PPO trainer extension for DrKernel migration."""

    def fit(self):
        with (
            _patch_compute_advantage(self.config),
            _patch_rollout_correction(self.config),
            _patch_apply_kl_penalty(self.config),
        ):
            return super().fit()
