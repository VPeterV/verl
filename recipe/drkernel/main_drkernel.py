# Copyright 2026 Bytedance Ltd.

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from recipe.drkernel.task_runner import DrKernelTaskRunner
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


def _migrate_legacy_kernel_reward_fields(config):
    """Preserve DrKernel-specific legacy reward_model fields under reward.kernel."""
    legacy_reward_model = OmegaConf.to_container(config.get("reward_model", {}), resolve=False)
    config = migrate_legacy_reward_impl(config)

    if not isinstance(legacy_reward_model, dict):
        return config

    consumed_legacy_keys = {
        "num_workers",
        "reward_manager",
        "enable",
        "enable_resource_pool",
        "n_gpus_per_node",
        "nnodes",
        "reward_loop_source",
        "reward_loop_module_path",
        "reward_loop_class_name",
        "model",
        "rollout",
        "reward_kwargs",
    }

    with open_dict(config):
        if "kernel" not in config.reward or config.reward.kernel is None:
            config.reward.kernel = {}

        for key, value in legacy_reward_model.items():
            if key in consumed_legacy_keys or value is None:
                continue
            if key not in config.reward.kernel or config.reward.kernel[key] is None:
                config.reward.kernel[key] = value

    return config


@hydra.main(config_path="config", config_name="drkernel_trainer", version_base=None)
def main(config):
    """Entry point for DrKernel recipe on top of the latest verl PPO pipeline."""
    auto_set_device(config)
    config = _migrate_legacy_kernel_reward_fields(config)
    task_runner_class = ray.remote(num_cpus=1)(DrKernelTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
