# Copyright 2026 Bytedance Ltd.

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_CLIENT = None
_CLIENT_SERVER_URL = None


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _round2(value: float) -> float:
    return float(f"{value:.2f}")


def _extract_kernel_cfg(reward_config):
    if reward_config is None:
        return {}

    reward_cfg = _cfg_get(reward_config, "reward", None)
    if reward_cfg is not None:
        kernel_cfg = _cfg_get(reward_cfg, "kernel", None)
        if kernel_cfg is not None:
            return kernel_cfg

    reward_model_cfg = _cfg_get(reward_config, "reward_model", None)
    if reward_model_cfg is not None:
        return reward_model_cfg

    return reward_config


def _penalty_score(kernel_cfg) -> float:
    policy = _cfg_get(kernel_cfg, "reward_policy", {})
    penalties = _cfg_get(policy, "penalties", {})
    return _to_float(_cfg_get(penalties, "penalty_score", 0.0), 0.0)


def _extract_kernel_code(solution_str: str) -> str:
    patterns = [
        r"# Kernel Implementation\\s*\\n(.*?)(?=# End|$)",
        r"```python\\s*# Kernel\\s*\\n(.*?)```",
        r"# Your implementation:\\s*\\n(.*?)(?=# End|$)",
        r"# Generated kernel:\\s*\\n(.*?)(?=# End|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    code_blocks = re.findall(r"```(?:\\w+)?\\s*\\n?(.*?)```", solution_str, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    return solution_str


def _build_error_results(num_items: int, kernel_cfg, error: Exception | str) -> list[dict[str, Any]]:
    penalty = _penalty_score(kernel_cfg)
    err_str = str(error)
    return [
        {
            "score": penalty,
            "reward": penalty,
            "correctness": False,
            "success": False,
            "compiled": False,
            "compilation": False,
            "speedup": 0.0,
            "performance": 0.0,
            "is_speedup_positive": False,
            "is_decoy_kernel": False,
            "num_custom_kernel": 0,
            "num_total_kernels": 0,
            "num_coverage": 0.0,
            "custom_kernel_cuda_time_in_profiling_us": 0.0,
            "total_kernel_run_time_in_profiling_us": 0.0,
            "time_coverage": 0.0,
            "status": "error",
            "error": err_str,
        }
        for _ in range(num_items)
    ]


def _run_coro(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        isolated = asyncio.new_event_loop()
        try:
            return isolated.run_until_complete(coro)
        finally:
            isolated.close()

    return loop.run_until_complete(coro)


def _get_client(kernel_cfg):
    global _CLIENT
    global _CLIENT_SERVER_URL

    server_url = _cfg_get(kernel_cfg, "server_url", None)
    if not server_url:
        return None

    try:
        from kernel.rewards.reward_client import KernelRewardClient
    except Exception as exc:
        logger.warning("kernel.rewards.reward_client is unavailable, fallback reward path will be used: %s", exc)
        return None

    if _CLIENT is None or _CLIENT_SERVER_URL != server_url:
        _CLIENT = KernelRewardClient(reward_config=kernel_cfg)
        _CLIENT_SERVER_URL = server_url

    return _CLIENT


def compute_kernel_reward_batch(
    solution_strs: list[str],
    ground_truths: list[str],
    entry_points: list[str] | None = None,
    uuids: list[str] | None = None,
    reward_config=None,
    is_valid: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """Batch kernel reward computation against KernelServer with graceful fallback."""
    kernel_cfg = _extract_kernel_cfg(reward_config)
    if entry_points is None:
        entry_points = [""] * len(solution_strs)

    if uuids is None:
        uuids = [""] * len(solution_strs)

    client = _get_client(kernel_cfg)
    if client is None:
        return _build_error_results(len(solution_strs), kernel_cfg, "KernelRewardClient unavailable or server_url missing")

    num_perf_trials = _cfg_get(kernel_cfg, "num_perf_trials", None)
    num_correct_trials = _cfg_get(kernel_cfg, "num_correct_trials", None)
    enable_profiling = _cfg_get(kernel_cfg, "enable_profiling", None)
    verbose_errors = _cfg_get(kernel_cfg, "verbose_errors", None)
    detect_decoy_kernel = _cfg_get(kernel_cfg, "detect_decoy_kernel", None)
    reference_backend = _cfg_get(kernel_cfg, "reference_backend", None)
    task_timeout = _cfg_get(kernel_cfg, "task_timeout", None)
    task_timeout_in_client = _cfg_get(kernel_cfg, "task_timeout_in_client", None)

    tasks = []
    for i, solution_str in enumerate(solution_strs):
        tasks.append(
            {
                "reference_code": ground_truths[i],
                "kernel_code": _extract_kernel_code(solution_str),
                "entry_point": entry_points[i],
                "use_reference_cache": False,
                "uuid": uuids[i],
                "is_valid": is_valid,
                "task_timeout": task_timeout,
                "task_timeout_in_client": task_timeout_in_client,
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "enable_profiling": enable_profiling,
                "verbose_errors": verbose_errors,
                "detect_decoy_kernel": detect_decoy_kernel,
                "reference_backend": reference_backend,
            }
        )

    try:
        return _run_coro(
            client.compute_batch_rewards(
                tasks,
                use_reference_cache=False,
                is_valid=is_valid,
                task_timeout=task_timeout,
                task_timeout_in_client=task_timeout_in_client,
            )
        )
    except Exception as exc:
        logger.exception("compute_kernel_reward_batch failed: %s", exc)
        return _build_error_results(len(solution_strs), kernel_cfg, exc)


def compute_kernel_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    entry_point: str | None = None,
    uuid: str | None = None,
    reward_config=None,
    is_valid: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Single-sample reward entrypoint compatible with verl reward manager contract."""
    _ = data_source

    extra_info = extra_info or {}
    kernel_cfg = _extract_kernel_cfg(reward_config)

    if entry_point is None:
        entry_point = str(extra_info.get("entry_point", ""))
    if uuid is None:
        uuid = str(extra_info.get("uuid", extra_info.get("task_id", "")))

    results = compute_kernel_reward_batch(
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        entry_points=[entry_point],
        uuids=[uuid],
        reward_config=reward_config,
        is_valid=is_valid,
        **kwargs,
    )

    if not results:
        results = _build_error_results(1, kernel_cfg, "empty reward result")

    result = results[0]
    score = _to_float(result.get("score", result.get("reward", _penalty_score(kernel_cfg))), _penalty_score(kernel_cfg))

    num_custom_kernel = _to_float(result.get("num_custom_kernel", 0.0), 0.0)
    num_total_kernels = _to_float(result.get("num_total_kernels", 0.0), 0.0)
    custom_time = _to_float(result.get("custom_kernel_cuda_time_in_profiling_us", 0.0), 0.0)
    total_time = _to_float(result.get("total_kernel_run_time_in_profiling_us", 0.0), 0.0)

    num_coverage = _to_float(result.get("num_coverage", 0.0), 0.0)
    if num_total_kernels > 0:
        num_coverage = num_custom_kernel / num_total_kernels

    time_coverage = _to_float(result.get("time_coverage", 0.0), 0.0)
    if total_time > 0:
        time_coverage = custom_time / total_time

    speedup = _to_float(result.get("speedup", result.get("performance", 0.0)), 0.0)
    speedup_eps = _to_float(_cfg_get(kernel_cfg, "speedup_eps", 0.01), 0.01)
    compiled = bool(result.get("compiled", result.get("compilation", False)))
    decoy_raw = result.get("decoy_kernel", result.get("is_decoy_kernel", False))

    return {
        "score": score,
        "reward": score,
        "success": bool(result.get("success", False)),
        "compiled": compiled,
        "compilation": compiled,
        "correctness": bool(result.get("correctness", False)),
        "speedup": speedup,
        "performance": speedup,
        "is_speedup_positive": bool(speedup >= 1.0 + speedup_eps),
        "is_decoy_kernel": decoy_raw,
        "num_custom_kernel": num_custom_kernel,
        "num_total_kernels": num_total_kernels,
        "num_coverage": _round2(num_coverage),
        "custom_kernel_cuda_time_in_profiling_us": custom_time,
        "total_kernel_run_time_in_profiling_us": total_time,
        "time_coverage": _round2(time_coverage),
        "status": result.get("status", "unknown"),
        "error": result.get("error", None),
    }


compute_kernel_reward_batch.__drkernel_batch_fn__ = True
