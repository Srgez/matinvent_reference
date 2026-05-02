"""
GRPO utility functions for MatInvent.

Data policy (phase 1 - GRPO-lite):
  all_success  : all samples that obtained a reward this step (used to compute baseline)
  top-k        : highest-reward subset of all_success (used for training updates)
  replay       : historical high-reward samples (off-policy; disabled by default for GRPO)

Advantage computation:
  advantage_i = (reward_i - mean(all_success_rewards)) / (std(all_success_rewards) + eps)
  clipped to [-adv_clip, adv_clip]

The baseline is always the FULL current-rollout pool (all_success), not the top-k subset.
This ensures top-k samples have a meaningful reference even when they are all high-scoring.
"""

import logging
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────
# Core advantage computation
# ──────────────────────────────────────────────────────────────

def compute_advantage(
    rewards_train: np.ndarray,
    rewards_all: np.ndarray,
    adv_eps: float = 1e-8,
    adv_clip: Optional[float] = 5.0,
) -> np.ndarray:
    """Compute per-sample GRPO advantage.

    Normalises rewards_train using the mean/std of the full current-rollout
    reward pool (rewards_all), which is typically larger than rewards_train.

    Args:
        rewards_train: rewards for the training subset (e.g. top-k), shape (N_train,).
        rewards_all:   rewards for ALL successfully scored samples this step, shape (N_all,).
        adv_eps:       small epsilon added to std to avoid division by zero.
        adv_clip:      symmetric clip magnitude; None disables clipping.

    Returns:
        advantage array of the same shape as rewards_train.
    """
    n_all = len(rewards_all)
    if n_all < 2:
        logging.warning(
            f"compute_advantage: only {n_all} success sample(s) this step. "
            "Returning zero advantage."
        )
        return np.zeros_like(rewards_train, dtype=float)

    mean_all = float(rewards_all.mean())
    std_all = float(rewards_all.std())

    advantage = (rewards_train.astype(float) - mean_all) / (std_all + adv_eps)

    if adv_clip is not None:
        advantage = np.clip(advantage, -adv_clip, adv_clip)

    return advantage


# ──────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────

def log_advantage_stats(advantage: np.ndarray, prefix: str = "train") -> dict:
    """Compute and log descriptive statistics of an advantage array."""
    stats = {
        f"adv_{prefix}_mean": float(advantage.mean()),
        f"adv_{prefix}_std": float(advantage.std()),
        f"adv_{prefix}_min": float(advantage.min()),
        f"adv_{prefix}_max": float(advantage.max()),
        f"adv_{prefix}_pos_frac": float((advantage > 0).mean()),
    }
    log_str = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
    logging.info(f"Advantage stats [{prefix}]: {log_str}")
    return stats


def log_reward_baseline_stats(
    rewards_all: np.ndarray,
    rewards_train: np.ndarray,
) -> dict:
    """Log baseline vs train-subset reward statistics."""
    stats = {
        "baseline_reward_mean": float(rewards_all.mean()),
        "baseline_reward_std": float(rewards_all.std()),
        "train_reward_mean": float(rewards_train.mean()),
        "train_reward_std": float(rewards_train.std()),
        "n_all_success": len(rewards_all),
        "n_train": len(rewards_train),
    }
    log_str = ", ".join(f"{k}={v}" for k, v in stats.items())
    logging.info(f"GRPO reward baseline: {log_str}")
    return stats
