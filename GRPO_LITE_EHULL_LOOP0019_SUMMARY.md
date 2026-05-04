# GRPO-lite ehull 阶段总结：`loop_0019`

本文档总结当前阶段已跑出的 GRPO-lite checkpoint：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my/models/loop_0019
```

该 checkpoint 是 `grpo_lite_ehull_v1_my` 实验中第 20 个 RL loop 结束后保存的模型，用于后续 resume 或作为 strict/full GRPO 的参考起点。

## 1. 当前方案定位

当前阶段实现的是 **GRPO-lite**，不是严格 PPO/GRPO 版本。

核心特点：

- 无条件生成模型：`my_train_mattergen_base`
- reward：`ehull`
- 训练目标：用当前 rollout 中的 reward advantage 替代原始 reward 作为 diffusion finetune 权重
- 不计算逐步 `logp_old / logp_new`
- 不使用 policy ratio
- 不做 PPO-style clipping
- KL 正则与 advantage 解耦，使用固定权重

简化理解：

```text
采样 64 个结构
  -> MatterSim 计算 ehull
  -> reward = descending(ehull)
  -> 用全部成功样本计算 baseline mean/std
  -> top-k 样本计算 advantage
  -> top-k + replay 样本用于 finetune
  -> loss = advantage * diffusion_loss + sigma * KL
```

## 2. 关键配置

实验目录：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my
```

checkpoint：

```text
models/loop_0019/last.ckpt
models/loop_0019/config.yaml
```

主要超参：

| 项目 | 值 |
|---|---:|
| `rl_epoch` | 120 |
| `save_freq` | 10 |
| `eval_size` | 64 |
| `sample batch_size` | 32 |
| `sample num_batches` | 3 |
| `finetune batch_size` | 32 |
| `topk_ratio` | 0.5 |
| `top-k train samples` | 16 |
| `ft_timesteps` | 300 |
| `ft_epochs` | 3 |
| `lr` | 5e-6 |
| `sigma` | 0.025 |
| `GRPO mode` | `lite` |
| `adv_clip` | 3.0 |
| `kl_weight_mode` | `fixed` |
| `replay` | true |
| `replay buffer_size` | 300 |
| `replay sample_size` | 16 |
| `replay reward_cutoff` | 0.65 |
| `div_filter` | true |
| `div_filter tol/buff` | 3 / 6 |

reward 配置：

```text
reward = ehull
MatterSim potential:
/mnt/shared-storage-user/zhangsizhe/mattergen_steering_2/checkpoints/mattersim/mattersim-v1.0.0-1M.pth

reference dataset:
/mnt/shared-storage-user/zhangsizhe/mattergen/data-release/alex-mp/reference_MP2020correction.gz
```

注意：正式脚本中禁用了 `sample_cfg.filter`，避免 `OptFilter` 和 MatterSim 同时打开同一个 LMDB reference 导致冲突。

## 3. Loop / Step / Epoch 含义

在当前 MatInvent 代码里：

- **Loop / Step**：一次完整 RL 迭代，包括采样、打分、GRPO advantage、finetune。
- **Epoch**：每个 loop 内部的 finetune epoch。当前每个 loop 有 3 个 finetune epoch。
- **Checkpoint loop_0019**：表示 `step=19` 的 RL loop 完成后保存，也就是已经完成 loop 0 到 loop 19，共 20 个 RL loops。

本 checkpoint 对应状态：

```text
完成 loop: 0-19
已保存 checkpoint: loop_0009, loop_0019
当前 checkpoint: loop_0019
```

## 4. `loop_0019` 当轮日志结果

`loop_0019` 从日志看是正常完成的：

```text
LOOP 19 START  : 2026-05-02 03:52:21
LOOP 19 FINISH : 2026-05-02 03:57:15
loop time      : 4.89 min
```

采样与 reward：

| 指标 | 值 |
|---|---:|
| successfully rewarded samples | 64 |
| evaluation costs to date | 1280 |
| reward mean | 0.8471 |
| reward std | 0.1488 |
| ehull mean | 0.0714 |
| ehull std | 0.0812 |
| generated crystals so far | 1280 |
| unique components | 1266 |
| div ratio | 0.9891 |
| burden | 1.5366 |

Replay / diversity：

| 指标 | 值 |
|---|---:|
| diversity tol_n | 0 |
| diversity buff_n | 0 |
| replay buffer size | 300 |
| replay buffer reward mean | 0.9810 |

GRPO advantage：

| 指标 | 值 |
|---|---:|
| baseline reward mean | 0.8471 |
| baseline reward std | 0.1488 |
| top-k train reward mean | 0.9927 |
| top-k train reward std | 0.0111 |
| `n_all_success` | 64 |
| `n_train` | 16 |
| advantage mean | 0.9010 |
| advantage std | 0.1643 |
| advantage min | 0.4048 |
| advantage max | 1.0277 |
| positive advantage fraction | 1.0000 |

Finetune loss：

| finetune epoch | loss | loss_diff | loss_kl |
|---:|---:|---:|---:|
| 0 | 0.2169 | 0.2066 | 0.4119 |
| 1 | 0.2128 | 0.2014 | 0.4547 |
| 2 | 0.2094 | 0.1970 | 0.4934 |

观察：

- 当轮 64 个样本全部成功获得 reward。
- `ehull mean=0.0714`，比早期 loop 的 0.10 左右更低，说明当前 rollout 已明显偏向低 ehull 区域。
- replay buffer 已满，且 buffer reward mean 达到 0.9810，说明 buffer 中积累的都是高 reward 样本。
- top-k train reward mean 接近 1.0，advantage 全为正，当前训练样本集中在比该轮平均更好的区域。
- `loss_diff` 从 epoch 0 到 epoch 2 下降，训练过程没有出现数值异常。
- `loss_kl` 在 0.41 到 0.49，说明 agent 已经开始偏离 prior，但当前使用固定 `sigma=0.025` 进行约束，整体 loss 仍稳定。

## 5. 当前方案相比原 MatInvent 的变化

原始 MatInvent 主要是 reward-weighted diffusion finetune。

当前 GRPO-lite 的关键变化是：

1. **使用 group baseline**

   每个 RL loop 中，使用全部成功样本的 reward mean/std 作为基线。

2. **训练 top-k，但 baseline 来自全部样本**

   训练只用 top-k 以及 replay 样本，但 advantage 的归一化不是只看 top-k，而是看当前 rollout 的全部成功样本。

3. **advantage 替代 raw reward**

   finetune 中使用 `advantage * sample_loss`，让模型更关注“相对当前 batch 更好”的样本。

4. **KL 与 advantage 解耦**

   GRPO-lite 模式下 KL 使用固定权重 `sigma=0.025`，不是原始的 reward-weighted KL。

5. **仍保留 replay**

   replay 样本使用当前 rollout baseline 重新归一化，属于近似 off-policy 使用。

## 6. 当前阶段的局限

这个 checkpoint 仍属于 GRPO-lite，局限包括：

- 没有逐步 logprob 缓存。
- 没有 `logp_old / logp_new`。
- 没有 policy ratio。
- 没有 PPO-style clipped surrogate。
- replay 是近似 off-policy，没有 importance ratio 修正。
- 对 diffusion trajectory 的每一步仍使用 sample loss 作为 proxy，不是严格的 action logprob。
- 当前是无条件生成，不涉及条件模型 adapter 或 CFG 条件分支优化。

因此，`loop_0019` 适合作为：

- GRPO-lite 的阶段性 checkpoint
- resume 训练的起点
- 与原始 MatterGen / my_train_mattergen_base 做生成质量对比的 checkpoint
- 后续 strict/full GRPO 开发的行为参考

但它还不是“完整 GRPO”。

## 7. 后续建议

### 继续 GRPO-lite 训练

如果继续当前实验，应从：

```text
models/loop_0019
```

加载，并设置：

```text
pipeline.start_loop=20
```

原因是 `loop_0019` 表示 loop 0-19 已完成，下一轮应该从 loop 20 开始。

### 开发 strict/full GRPO

建议在新 worktree 中开发：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference_grpo_strict
```

目标是实现：

- 采样时记录每个 denoising step 的 `logp_old`
- finetune 时重算 `logp_new`
- 计算 `ratio = exp(logp_new - logp_old)`
- 使用 PPO-style clipped objective
- 保留 KL fixed weight
- 暂不计算 corrector step logprob，但保留接口

### 与条件生成分开推进

当前 `loop_0019` 是无条件 GRPO-lite checkpoint。条件生成 adapter-only GRPO 应作为另一条线推进，避免和 strict/full GRPO 的无条件实现混在同一个开发分支中。

## 8. 相关文件

训练输出：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my
```

checkpoint：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my/models/loop_0019/last.ckpt
```

checkpoint config：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my/models/loop_0019/config.yaml
```

训练日志：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my/train.log
```

主配置记录：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/exp_res/grpo_lite_ehull_v1_my/hparams.yaml
```

相关实现：

```text
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/pipeline/mat_invent.py
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/pipeline/grpo_utils.py
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/models/mattergen/dataset.py
/mnt/shared-storage-user/zhangsizhe/matinvent_reference/models/suite/mattergen.py
```
