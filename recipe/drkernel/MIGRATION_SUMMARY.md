# DrKernel -> 新版 verl 迁移交付总结（可上线测试版）

## 1. 交付目标
- 在不 fork `verl` 主干核心逻辑的前提下，把 `drkernel` recipe 迁移到新版 `verl`。
- 对齐旧版关键语义，优先保证训练结果行为一致：
  - `adv_by_last_turn`
  - turn-level multi-turn advantage（`trloo/erloo/erloo_norm/grpo/turn_independent_grpo/reinforce/egae`）
  - `use_final_reward`
  - `loss_mask` 对 mask/reward 的屏蔽
  - coverage rejection sampling（`coverage_rs`）
- 结构遵循新版设计：能力尽量放在 `recipe/drkernel`，复用原生 `verl`。

## 2. 本次已完成的实现

### 2.1 训练链路对齐（核心）
- 文件：`recipe/drkernel/trainer/drkernel_ray_trainer.py`
- 完成项：
  - patch `compute_advantage`，并按旧版顺序执行：
    1. `loss_mask` -> `response_mask/attention_mask` 屏蔽
    2. `use_final_reward`
    3. `loss_mask` -> `token_level_scores/token_level_rewards` 屏蔽
    4. `adv_by_last_turn` 分支
    5. turn-level 分支
    6. fallback 到新版原生 `compute_advantage`
  - patch `rollout_correction`，保留原生 rollout correction 后叠加 coverage rejection。
  - patch `apply_kl_penalty`，在 `use_kl_in_reward=true` 下仍执行 DrKernel 风格的 final-reward + reward-mask 预处理。

### 2.2 turn-level 优势估计严格映射
- 文件：`recipe/drkernel/adv/turn_level_adv.py`
- 完成项：
  - 对齐旧版公式：
    - `compute_multi_turn_returns`
    - `trloo`（按 `(uid, turn_idx)` 分组 LOO）
    - `erloo` / `erloo_norm`
    - `grpo` / `turn_independent_grpo`
    - `reinforce`
    - `egae`
  - 对齐旧版增强项：
    - `adv_by_last_turn`（最后一轮计算 + 广播）
    - `reward_shaping` / `unbiased_shaping`
    - `batch_std`（智能标准化）
    - `use_multi_prompt_mvu`（若启用但缺模块则抛错）
  - 新增 DrKernel 兼容 helper：
    - `maybe_apply_final_reward`
    - `maybe_apply_loss_mask_to_masks`
    - `maybe_apply_loss_mask_to_rewards`
  - 产出 token 级 `advantages/returns` 的同时，补充 `turn_level_advantages/turn_level_returns` 便于对拍。

### 2.3 Reward 路径对齐
- 文件：`recipe/drkernel/reward/kernel_reward_fn.py`
- 完成项：
  - 保留 batch + single 两入口：
    - `compute_kernel_reward_batch`
    - `compute_kernel_reward`
  - 与旧版字段对齐：
    - `reward/score`
    - `compiled/compilation`
    - `speedup/performance`
    - `is_speedup_positive`
    - `is_decoy_kernel`
    - `num_coverage/time_coverage`（保留旧版 2 位小数风格）
  - 无法连接 KernelServer 时返回 penalty fallback（可训练不中断）。

- 文件：`recipe/drkernel/reward/kernel_reward_manager.py`
- 完成项：
  - 同时兼容两类自定义 reward 函数签名：
    - single 风格：`solution_str/ground_truth/...`
    - batch 风格：`solution_strs/ground_truths/...`
  - 自动把 reward 结果规范化成 reward-loop 需要的 `reward_score + reward_extra_info`。

### 2.4 Coverage rejection 对齐
- 文件：`recipe/drkernel/reward/coverage_rs.py`
- 完成项：
  - 兼容 `turn/geometric` 两种 `coverage_rs`。
  - 支持 `time_coverage/num_coverage`、`speedup_threshold` 以及正确性过滤。
  - 兼容字段别名（如 `speedup` vs `performance`、`decoy_kernel` vs `is_decoy_kernel`）。
  - `max_turns` 推断增强（`turn_level_loss.max_turns` -> `turn_indices` -> `__num_turns__`）。

### 2.5 配置与入口迁移对齐
- 文件：`recipe/drkernel/config/drkernel_trainer.yaml`
- 完成项：
  - 补齐旧版关键算法开关：
    - `adv_by_last_turn`
    - `use_final_reward`
    - `reward_shaping`
    - `unbiased_shaping`
    - `batch_std`
    - `use_multi_prompt_mvu`
  - `turn_level_loss.max_turns` 与 `multi_turn.max_user_turns` 对齐。
  - `reward.kernel.*` 补齐 kernel server 常用字段。

- 文件：`recipe/drkernel/main_drkernel.py`
- 完成项：
  - 在 `migrate_legacy_reward_impl` 基础上追加迁移：
    - 旧 `reward_model` 中 DrKernel 自定义字段 -> 新 `reward.kernel`
  - 避免 legacy 配置字段在迁移后静默丢失。

## 3. 新旧实现映射（关键路径）
- 旧：`drkernel/kernel/kernel_trainer.py`  
  新：`recipe/drkernel/trainer/drkernel_ray_trainer.py` + `recipe/drkernel/adv/turn_level_adv.py`

- 旧：`drkernel/kernel/rewards/kernel_reward.py` + `kernel_async.py`  
  新：`recipe/drkernel/reward/kernel_reward_fn.py` + `recipe/drkernel/reward/kernel_reward_manager.py`

- 旧：`drkernel/kernel/rewards/coverage_helper.py`  
  新：`recipe/drkernel/reward/coverage_rs.py`

- 旧：`main_kernel.py` 入口与自定义装配  
  新：`recipe/drkernel/main_drkernel.py` + `recipe/drkernel/task_runner.py`

## 4. 新增/更新测试
- `recipe/drkernel/tests/test_turn_level_adv.py`
  - `trloo` 公式对拍
  - `adv_by_last_turn` 广播对拍
  - `use_final_reward + loss_mask` helper 行为对拍

- `recipe/drkernel/tests/test_kernel_reward_manager.py`
  - single 签名 reward 函数
  - batch 签名 reward 函数

- `recipe/drkernel/tests/test_kernel_reward_fn.py`
  - 无 server 时 penalty fallback
  - 旧字段兼容性检查

- `recipe/drkernel/tests/test_coverage_rs.py`
  - coverage rejection 关键行为校验

## 5. 本机验证结果
- 已执行：`python -m compileall recipe/drkernel`  
  - 结果：通过。

- 未执行成功：`pytest recipe/drkernel/tests -q`  
  - 原因：当前机器缺少 `pytest`（以及你之前提到的运行依赖不完整）。

## 6. 目标环境上线前建议验收命令
1. `pip install -r requirements.txt`（或你们线上标准依赖安装方式）
2. `pytest recipe/drkernel/tests -q`
3. 使用小规模固定 seed 对拍旧版与新版：
   - 对比 `advantages/returns`（抽样 batch）
   - 对比 `coverage_rs_masked_fraction` 等关键指标
   - 对比 reward 关键字段（`score/performance/time_coverage/num_coverage`）
4. 再做一轮短程训练 smoke test（几十到几百 step）确认曲线行为一致。
