# Copyright 2026 Bytedance Ltd.

from recipe.drkernel.trainer.drkernel_ray_trainer import DrKernelRayPPOTrainer
from verl.trainer import main_ppo


class DrKernelTaskRunner(main_ppo.TaskRunner):
    """Task runner that swaps the trainer with DrKernel extensions."""

    def run(self, config):
        original_cls = main_ppo.RayPPOTrainer
        main_ppo.RayPPOTrainer = DrKernelRayPPOTrainer
        try:
            super().run(config)
        finally:
            main_ppo.RayPPOTrainer = original_cls
