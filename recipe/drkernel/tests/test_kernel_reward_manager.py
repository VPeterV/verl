import math
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from recipe.drkernel.reward.kernel_reward_manager import DrKernelRewardManager
from verl import DataProto


class _DummyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        _ = skip_special_tokens
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(str(int(x)) for x in token_ids)


def test_kernel_reward_manager_run_single_contract():
    config = OmegaConf.create({"reward": {"kernel": {"is_valid": False}}})

    def compute_score(**kwargs):
        assert kwargs["entry_point"] == "kernel_main"
        assert kwargs["uuid"] == "uid-1"
        return {
            "score": 1.25,
            "correctness": True,
            "performance": 2.0,
            "time_coverage": 0.5,
        }

    manager = DrKernelRewardManager(
        config=config,
        tokenizer=_DummyTokenizer(),
        compute_score=compute_score,
    )

    batch = TensorDict(
        {
            "responses": torch.tensor([[101, 102, 103]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.int64),
        },
        batch_size=[1],
    )
    non_tensor = {
        "data_source": np.array(["kernel"], dtype=object),
        "reward_model": np.array(
            [
                {
                    "ground_truth": "def kernel_main(x): return x",
                    "entry_point": "kernel_main",
                }
            ],
            dtype=object,
        ),
        "uid": np.array(["uid-1"], dtype=object),
        "extra_info": np.array([{"task": "unit"}], dtype=object),
    }

    data = DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={})
    out = manager.loop.run_until_complete(manager.run_single(data))

    assert math.isclose(out["reward_score"], 1.25, rel_tol=1e-6)
    assert out["reward_extra_info"]["correctness"] is True
    assert math.isclose(out["reward_extra_info"]["performance"], 2.0, rel_tol=1e-6)


def test_kernel_reward_manager_supports_batch_style_reward_fn():
    config = OmegaConf.create({"reward": {"kernel": {"is_valid": False}}})

    def compute_score(solution_strs, ground_truths, entry_points, uuids, **kwargs):
        assert len(solution_strs) == 1
        assert ground_truths[0].startswith("def kernel_main")
        assert entry_points[0] == "kernel_main"
        assert uuids[0] == "uid-1"
        assert kwargs["is_valid"] is False
        return [
            {
                "reward": 0.75,
                "correctness": False,
                "performance": 1.2,
            }
        ]

    manager = DrKernelRewardManager(
        config=config,
        tokenizer=_DummyTokenizer(),
        compute_score=compute_score,
    )

    batch = TensorDict(
        {
            "responses": torch.tensor([[101, 102, 103]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.int64),
        },
        batch_size=[1],
    )
    non_tensor = {
        "data_source": np.array(["kernel"], dtype=object),
        "reward_model": np.array(
            [
                {
                    "ground_truth": "def kernel_main(x): return x",
                    "entry_point": "kernel_main",
                }
            ],
            dtype=object,
        ),
        "uid": np.array(["uid-1"], dtype=object),
    }

    data = DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info={})
    out = manager.loop.run_until_complete(manager.run_single(data))
    assert math.isclose(out["reward_score"], 0.75, rel_tol=1e-6)
    assert out["reward_extra_info"]["correctness"] is False
