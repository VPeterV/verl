import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.drkernel.reward.coverage_rs import apply_coverage_rejection_to_batch
from verl import DataProto


def test_coverage_rejection_masks_only_low_coverage_correct_samples():
    batch = TensorDict(
        {
            "response_mask": torch.ones((4, 3), dtype=torch.float32),
        },
        batch_size=[4],
    )
    non_tensor = {
        "time_coverage": np.array([0.1, 0.7, 0.2, 0.9]),
        "num_coverage": np.array([0.1, 0.7, 0.2, 0.9]),
        "correctness": np.array([True, True, False, True]),
        "is_decoy_kernel": np.array([False, False, False, False]),
        "performance": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    data = DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={})

    cfg = OmegaConf.create(
        {
            "reward": {
                "kernel": {
                    "coverage_rs": "turn",
                    "coverage_rs_key": "time_coverage",
                    "coverage_rs_threshold": 0.5,
                    "coverage_rs_factor": 0.0,
                    "speedup_threshold": None,
                }
            },
            "algorithm": {"turn_level_loss": {"max_turns": 1}},
        }
    )

    out, metrics = apply_coverage_rejection_to_batch(data, cfg)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(out.batch["response_mask"], expected)

    assert "rollout_corr/coverage/coverage_rs_masked_fraction" in metrics
    assert metrics["rollout_corr/coverage/coverage_rs_masked_fraction"] == 0.25
