from omegaconf import OmegaConf

from recipe.drkernel.reward.kernel_reward_fn import compute_kernel_reward, compute_kernel_reward_batch


def _build_cfg():
    return OmegaConf.create(
        {
            "reward": {
                "kernel": {
                    "server_url": None,
                    "speedup_eps": 0.01,
                    "reward_policy": {"penalties": {"penalty_score": -0.2}},
                }
            }
        }
    )


def test_kernel_reward_batch_returns_penalty_when_server_unavailable():
    cfg = _build_cfg()
    out = compute_kernel_reward_batch(
        solution_strs=["def k(x): return x"],
        ground_truths=["def k(x): return x"],
        entry_points=["k"],
        uuids=["u-1"],
        reward_config=cfg,
    )
    assert len(out) == 1
    assert out[0]["score"] == -0.2
    assert out[0]["compilation"] is False
    assert out[0]["status"] == "error"


def test_kernel_reward_single_output_contains_legacy_compat_fields():
    cfg = _build_cfg()
    out = compute_kernel_reward(
        data_source="kernel",
        solution_str="def k(x): return x",
        ground_truth="def k(x): return x",
        entry_point="k",
        uuid="u-1",
        reward_config=cfg,
    )
    assert out["score"] == -0.2
    assert out["reward"] == -0.2
    assert "compilation" in out
    assert "is_speedup_positive" in out
    assert "num_coverage" in out
    assert "time_coverage" in out
