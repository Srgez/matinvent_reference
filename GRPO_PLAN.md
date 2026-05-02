# GRPO 实现计划（matinvent_reference）

> 记录于 2026-05-02，根据聊天讨论整理。  
> 计划分三个阶段迭代：GRPO-lite → 严格 GRPO → Branch GRPO。

---

## 背景与动机

**MatInvent** 原有的 RL 微调框架基于 reward-weighted likelihood，对所有通过过滤的样本赋予相同正向权重，缺乏对"相对质量"的刻画。

**GRPO（Group-based Reward Policy Optimization）** 的改进点：
1. **在线生成**：每步 rollout 使用当前策略采样，而非固定 buffer。
2. **Advantage（优势）替代 reward 直接加权**：对同一批次内的 reward 做归一化，突出"比平均更好"的样本。
3. **KL 正则化解耦**：将 KL 损失从 advantage 权重中分离，单独控制权重，使两者不相互干扰。

---

## 阶段一：GRPO-lite（已完成 ✅）

### 核心思想

- **不改动采样链路**（不需要逐步 logprob）
- Advantage = `(r - μ_all) / (σ_all + ε)`，其中 `μ_all / σ_all` 基于**全部成功样本**（非仅 top-k）
- 使用 advantage 代替 reward 作为 ft_step 的权重
- KL 正则化：使用固定权重 `sigma`（`kl_weight_mode=fixed`）

### 数据流

```
rollout (N 个样本)
    ↓ 全部成功样本 → 计算 baseline (μ_all, σ_all)
    ↓ top-k 样本  → advantage_i = (r_i - μ_all) / (σ_all + ε)
    ↓ replay 样本 → 同一步 baseline 归一化（近似 off-policy）
    ↓ ft_step(batch, advantages=...)
        loss = -mean(advantage * log_p) + sigma * KL
```

### 关键超参（`configs/pipeline/mat_invent.yaml` → `grpo` 块）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `lite` | 控制 GRPO 模式（`none`/`lite`/`strict`） |
| `adv_eps` | `1e-8` | 优势归一化分母平滑项 |
| `adv_clip` | `5.0` | 优势截断上限（训练早期建议 3.0） |
| `kl_weight_mode` | `fixed` | KL 权重控制方式 |
| `clip_eps` | `0.2` | （Phase 2 用）PPO-style ratio clip |
| `strict_logprob` | `false` | （Phase 2 用）是否计算严格 logprob |

### 新增/修改文件

- `pipeline/grpo_utils.py` — `compute_advantage()`、`log_advantage_stats()` 等工具函数
- `pipeline/mat_invent.py` — `rl_step()` 加入 advantage 计算；`ft_step()` 接受 `advantages` 参数
- `models/mattergen/dataset.py` — `from_samples()` 支持 `extra_properties`（存 advantage 等）
- `models/suite/mattergen.py` — `get_dataloader()` 传递 `extra_properties`
- `configs/pipeline/mat_invent.yaml` — 新增 `grpo` 配置块

### Smoke 验证结果（2026-05-01）

```
reward mean=0.8157 std=0.1244
GRPO reward baseline: baseline_reward_mean=0.8157, train_reward_mean=0.9796
Advantage stats [train]: mean=1.3182, std=0.1637, min=1.1545, max=1.4820
Epoch 0: loss=0.6561, loss_diff=0.6559, loss_kl=0.0086
```

✅ 优势统计、loss、KL 均正常，smoke 通过。

---

## 阶段二：严格 GRPO（待实现 ⬜）

### 核心思想

在 GRPO-lite 基础上引入：
1. **逐步 logprob 计算**：在采样时记录每一步 `logπ(x_{t-1}|x_t)`
2. **Policy ratio**：`ratio = exp(logp_new - logp_old)`
3. **PPO-style clipping**：`L_clip = min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)`
4. **KL 完全解耦**：ratio 用 logp_new/logp_old 计算，KL 用单独 sigma 控制

### Logprob 组成

MatterGen 的每步去噪涉及三个字段，logprob 需要分别计算再求和：

| 字段 | 分布类型 | logprob 计算方式 |
|------|----------|-----------------|
| `pos`（原子位置） | 连续高斯 | `Normal(μ, σ).log_prob(x).sum()` |
| `cell`（晶胞参数） | 连续高斯 | `Normal(μ, σ).log_prob(x).sum()` |
| `atomic_numbers`（原子种类） | 离散 Categorical | `log_softmax(logit)[选中类别].sum()` |

**Corrector 步骤**：第一版可暂不计算 corrector 的 logprob（保留接口但置零），对 ratio 影响可接受。

### 新增文件（Vendor 策略）

将 MatterGen 采样相关代码复制到本项目并扩展，避免修改外部包：

- `models/mattergen/grpo_predictors.py`
  - `GRPOAncestralPredictor`：扩展 AncestralPredictor，`update_given_score_with_logprob()` 返回 `(x_next, logp)`
  - `GRPOD3PMPredictor`：同上针对 D3PM（离散扩散）
  - `_aggregate_logp(pos_logp, cell_logp, atom_logp) → float`

- `models/mattergen/grpo_sampler.py`
  - `sample_with_rollout()`：采样同时记录每步 `(x_t, x_next, t, logp_old)`，返回 `FlatRolloutBuffer`
  - `compute_logp_new(buffer, model)`：用当前策略重新计算 logp

- `models/mattergen/grpo_rollout.py`
  - `TransitionRecord`：单步转换记录
  - `SampleRollout`：单条轨迹的所有转换
  - `FlatRolloutBuffer`：多条轨迹展平后的 buffer（供 PPO 更新）

### 训练循环修改

```python
# ft_step (strict mode)
logp_new = compute_logp_new(buffer, model)   # 从 buffer 重算 logp
ratio = exp(logp_new - buffer.logp_old)      # policy ratio
L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
loss = -L.mean() + sigma * KL
```

---

## 阶段三：Branch GRPO（待实现 ⬜）

### 分支策略

```
batchsize = 4 条基础轨迹

step 0   ──→ step 300   ──4 branches──→ step 500   ──4 branches──→ step 800   ──2 branches──→ step 1000
```

| 区间 | 分支数 | 最终粒子数 | 奖励来源 | Advantage 计算范围 |
|------|--------|-----------|---------|-------------------|
| 800→1000 | 2 | 2 | 最终奖励 | 每条基础轨迹的 2 个末端分支间 |
| 500→800 | 4 | 4/分支前 | 下游 2 个分支奖励的均值 | 4 个 500 步分支间 |
| 300→500 | 4 | 16/轨迹 | 下游 8 个结果奖励的均值 | 4 个 300 步分支间 |
| 0→300 | — | 32（4×8） | 下游 32 个奖励的均值 | 4 条基础轨迹间 |

### 核心设计原则

1. **先均值后优势**：对某个区间，先对下游分支奖励求均值（代表该分支节点价值），再在**同级节点**间计算 advantage。
2. **局部 KL 正则**：每个区间的更新仅对该区间步骤计算 KL，不跨区间累积。
3. **800→1000 步**：可引入局部奖励（每步根据当前 x0 预测计算 intermediate reward），结合最终奖励优化末段。

### 数据结构

- `BranchNode`：节点 `(x_t, t, logp, children=[], reward=None)`
- `BranchRolloutTree`：树形结构，支持从叶到根的奖励反向传播
  - `propagate_rewards()`：从叶子向上均值传播
  - `compute_sibling_advantages(node_level)`：在同级节点中计算 advantage

---

## 代码管理策略

- **单一 branch**：全部改动在 `matinvent_reference` 主仓，不开新 worktree
- **兼容开关**：`pipeline.grpo.mode` 控制行为（`none`/`lite`/`strict`/`branch`），旧 smoke 脚本设 `mode=none` 即可回退
- **Vendor 隔离**：MatterGen 内部逻辑复制到 `models/mattergen/grpo_*.py`，不修改外部包
- **Phase 2/3 骨架**：`grpo_predictors.py`、`grpo_sampler.py`、`grpo_rollout.py` 已建立 skeleton，待 Phase 2 填充实现

---

## 运行脚本

| 脚本 | 说明 |
|------|------|
| `run_on_h/smoke_grpo_lite.sh` | GRPO-lite 最小 smoke（2 GPU，1 epoch） |
| `run_on_h/run_grpo_lite_ehull.sh` | GRPO-lite 正式训练，ehull 奖励，8 GPU，120 epoch |
| `run_on_h/run_grpo_lite_bandgap_ehull.sh` | GRPO-lite 正式训练，bandgap+ehull 联合奖励，8 GPU |

### 正式训练关键参数（ehull）

```bash
EVAL_SIZE=64          # 每步 rollout 结构数
SAMPLE_BATCH_SIZE=32  # MatterGen 生成 batch
SAMPLE_NUM_BATCHES=3  # → 96 candidates，保留 64
FT_TIMESTEPS=300      # 随机 diffusion 步覆盖范围
FT_EPOCHS=3
ACCUM_STEPS=50
LR=5e-6
SIGMA=0.025           # KL 权重
TOPK_RATIO=0.5        # 前 50% 用于训练
BUFFER_SIZE=300       # 经验回放 buffer
REWARD_CUTOFF=0.65    # 入 buffer 的最低 reward 门槛
GRPO_ADV_CLIP=3.0     # 训练早期较紧，避免梯度爆炸
```

### 已知注意事项

1. **LMDB 冲突**：`OptFilter` 和 `MatterSim` ehull reward 都会打开同一 LMDB 文件，同一进程只能打开一次。解决方案：正式训练脚本使用 `~sample_cfg.filter` 禁用 `OptFilter`。
2. **HuggingFace Hub**：集群节点无网络，需设置 `HF_HUB_OFFLINE=1` 并提供本地 reference 路径。
3. **WandB 离线**：设置 `WANDB_MODE=offline`，训练后用 `wandb sync <run_dir>` 上传。

---

## 参考文献

- `reference/flow_grpo/`：Flow-based GRPO 参考实现
- `reference/flow_grpo_fast/`：加速版本
- `reference/TempFlow-GRPO/`：TempFlow 中的 GRPO 应用
