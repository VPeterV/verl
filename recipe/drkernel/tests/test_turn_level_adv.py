import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.drkernel.adv.turn_level_adv import (
    maybe_apply_final_reward,
    maybe_apply_loss_mask_to_masks,
    maybe_apply_loss_mask_to_rewards,
    maybe_compute_advantage_by_last_turn,
    maybe_compute_turn_level_advantage,
)
from verl import DataProto


def _build_base_data() -> DataProto:
    batch = TensorDict(
        {
            "token_level_rewards": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "token_level_scores": torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "response_mask": torch.ones((4, 3), dtype=torch.float32),
            "attention_mask": torch.ones((4, 6), dtype=torch.float32),
            "turn_indices": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            "loss_mask": torch.tensor([1, 1, 1, 1], dtype=torch.bool),
        },
        batch_size=[4],
    )
    non_tensor = {
        "uid": np.array(["p0", "p0", "p0", "p0"], dtype=object),
        "__num_turns__": np.array([2, 2, 2, 2], dtype=np.int32),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={})


def test_turn_level_trloo_advantage_matches_expected_values():
    data = _build_base_data()
    cfg = OmegaConf.create(
        {
            "turn_level_loss": {
                "enable": True,
                "max_turns": 2,
                "estimators": ["trloo"],
                "epsilon": 1e-6,
            },
            "adv_by_last_turn": False,
        }
    )

    out = maybe_compute_turn_level_advantage(
        data=data,
        adv_estimator="trloo",
        gamma=1.0,
        config=cfg,
    )

    assert out is not None

    expected_turn_returns = torch.tensor([3.0, 2.0, 7.0, 4.0])
    expected_turn_adv = torch.tensor([-4.0, -2.0, 4.0, 2.0])
    assert torch.allclose(out.batch["turn_level_returns"], expected_turn_returns)
    assert torch.allclose(out.batch["turn_level_advantages"], expected_turn_adv)

    expected_token_adv = expected_turn_adv.unsqueeze(-1).expand(-1, 3)
    expected_token_ret = expected_turn_returns.unsqueeze(-1).expand(-1, 3)
    assert torch.allclose(out.batch["advantages"], expected_token_adv)
    assert torch.allclose(out.batch["returns"], expected_token_ret)


def test_adv_by_last_turn_broadcast_matches_drkernel_behavior():
    data = _build_base_data()
    cfg = OmegaConf.create(
        {
            "adv_by_last_turn": True,
            "turn_level_loss": {"max_turns": 2},
        }
    )

    def fallback_compute_advantage(**kwargs):
        last_turn_data = kwargs["data"]
        last_turn_data.batch["advantages"] = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]], dtype=torch.float32)
        last_turn_data.batch["returns"] = torch.tensor([[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]], dtype=torch.float32)
        last_turn_data.meta_info["fallback_called"] = True
        return last_turn_data

    out = maybe_compute_advantage_by_last_turn(
        data=data,
        adv_estimator="grpo",
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=cfg,
        fallback_compute_advantage=fallback_compute_advantage,
    )

    assert out is not None
    expected_adv = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    expected_ret = torch.tensor(
        [
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
            [6.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(out.batch["advantages"], expected_adv)
    assert torch.allclose(out.batch["returns"], expected_ret)
    assert out.meta_info["fallback_called"] is True


def test_final_reward_and_loss_mask_helpers_match_legacy_intent():
    data = _build_base_data()
    data.batch["loss_mask"] = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    cfg = OmegaConf.create({"use_final_reward": True, "turn_level_loss": {"max_turns": 2}})

    maybe_apply_final_reward(data, cfg)
    maybe_apply_loss_mask_to_rewards(data)
    maybe_apply_loss_mask_to_masks(data)

    expected_scores = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(data.batch["token_level_scores"], expected_scores)
    assert torch.allclose(data.batch["token_level_rewards"], expected_scores)
    assert torch.allclose(data.batch["response_mask"], expected_mask)
