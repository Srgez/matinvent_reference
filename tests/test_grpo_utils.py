"""
Unit tests for GRPO-lite utilities.

Tests cover:
  1. compute_advantage() correctness
  2. log_advantage_stats() / log_reward_baseline_stats() run without error
  3. MatterGenDataset.from_samples() with extra_properties
  4. BranchRolloutTree reward propagation and advantage computation

Run with:
    cd /mnt/shared-storage-user/zhangsizhe/matinvent_reference
    python -m pytest tests/test_grpo_utils.py -v
  or without pytest:
    python tests/test_grpo_utils.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf


# ─────────────────────────────────────────────────────────────────────────────
# Tests for pipeline/grpo_utils.py
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_advantage_basic():
    """Normalised advantages should have approximately zero mean and unit std."""
    rewards_all = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    rewards_train = np.array([0.5, 0.7, 0.9])  # top-3

    from pipeline.grpo_utils import compute_advantage
    adv = compute_advantage(rewards_train, rewards_all, adv_eps=1e-8, adv_clip=None)

    assert adv.shape == rewards_train.shape
    # train samples are above the mean (0.5) so their advantages should be >=0
    assert (adv >= 0).all(), f"Expected non-negative advantages for top-k, got {adv}"
    print(f"[PASS] test_compute_advantage_basic: adv = {adv}")


def test_compute_advantage_normalisation():
    """All-success baseline mean/std should match expected values."""
    np.random.seed(42)
    rewards_all = np.random.uniform(0, 1, size=20)
    rewards_train = rewards_all[:5]

    from pipeline.grpo_utils import compute_advantage
    adv = compute_advantage(rewards_train, rewards_all, adv_eps=0.0, adv_clip=None)

    expected = (rewards_train - rewards_all.mean()) / rewards_all.std()
    np.testing.assert_allclose(adv, expected, atol=1e-6)
    print(f"[PASS] test_compute_advantage_normalisation")


def test_compute_advantage_clip():
    """Advantages outside [-clip, clip] should be clipped."""
    rewards_all = np.array([0.0, 0.0, 0.0, 0.0, 100.0])  # extreme outlier
    rewards_train = np.array([100.0])

    from pipeline.grpo_utils import compute_advantage
    adv = compute_advantage(rewards_train, rewards_all, adv_eps=1e-8, adv_clip=3.0)

    assert adv[0] <= 3.0, f"Expected clip at 3.0, got {adv[0]}"
    print(f"[PASS] test_compute_advantage_clip: adv = {adv}")


def test_compute_advantage_small_group():
    """With < 2 samples, should return zeros with a warning."""
    from pipeline.grpo_utils import compute_advantage
    adv = compute_advantage(
        rewards_train=np.array([0.5]),
        rewards_all=np.array([0.5]),   # n_all=1 < 2
        adv_eps=1e-8,
        adv_clip=5.0,
    )
    np.testing.assert_array_equal(adv, np.zeros(1))
    print(f"[PASS] test_compute_advantage_small_group")


def test_log_functions_run():
    """Logging helpers should execute without error and return dicts."""
    from pipeline.grpo_utils import log_advantage_stats, log_reward_baseline_stats
    adv = np.array([-1.0, 0.5, 1.5, -0.3])
    stats = log_advantage_stats(adv, prefix="train")
    assert "adv_train_mean" in stats

    baseline_stats = log_reward_baseline_stats(
        rewards_all=np.linspace(0.1, 0.9, 16),
        rewards_train=np.array([0.7, 0.8, 0.9]),
    )
    assert "baseline_reward_mean" in baseline_stats
    print(f"[PASS] test_log_functions_run")


def test_resolve_finetune_timestep_window_default():
    """Default window should match the current 0..timesteps-1 behavior."""
    from pipeline.timestep_window import resolve_finetune_timestep_window

    cfg = OmegaConf.create({
        "timesteps": 300,
        "timestep_start": 0,
        "timestep_end": None,
    })
    window = resolve_finetune_timestep_window(cfg)

    assert len(window) == 300
    assert window[0] == 0
    assert window[-1] == 299
    print(f"[PASS] test_resolve_finetune_timestep_window_default")


def test_resolve_finetune_timestep_window_middle_range():
    """Explicit middle window should train the requested global step range."""
    from pipeline.timestep_window import resolve_finetune_timestep_window

    cfg = OmegaConf.create({
        "timesteps": 300,
        "timestep_start": 300,
        "timestep_end": 700,
    })
    window = resolve_finetune_timestep_window(cfg)

    assert len(window) == 400
    assert window[0] == 300
    assert window[-1] == 699
    print(f"[PASS] test_resolve_finetune_timestep_window_middle_range")


def test_resolve_finetune_timestep_window_invalid_ranges():
    """Invalid timestep windows should raise clear errors."""
    from pipeline.timestep_window import resolve_finetune_timestep_window

    bad_cfgs = [
        {"timesteps": 300, "timestep_start": -1, "timestep_end": None},
        {"timesteps": 300, "timestep_start": 700, "timestep_end": 700},
        {"timesteps": 300, "timestep_start": 700, "timestep_end": 1001},
    ]

    for bad_cfg in bad_cfgs:
        cfg = OmegaConf.create(bad_cfg)
        try:
            resolve_finetune_timestep_window(cfg)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for config {bad_cfg}")

    print(f"[PASS] test_resolve_finetune_timestep_window_invalid_ranges")


def test_resolve_finetune_timestep_window_schedule_segments():
    """Schedule mode should switch windows by RL loop and reuse the last segment."""
    from pipeline.timestep_window import (
        resolve_finetune_schedule_segment,
        resolve_finetune_timestep_window,
    )

    cfg = OmegaConf.create({
        "timesteps": 300,
        "timestep_start": 0,
        "timestep_end": None,
        "timestep_schedule": [
            {"loop_start": 0, "loop_end": 30, "timestep_start": 0, "timestep_end": 300},
            {"loop_start": 30, "loop_end": 60, "timestep_start": 300, "timestep_end": 700},
        ],
    })

    seg0 = resolve_finetune_schedule_segment(cfg, loop_idx=0)
    seg29 = resolve_finetune_schedule_segment(cfg, loop_idx=29)
    seg30 = resolve_finetune_schedule_segment(cfg, loop_idx=30)
    seg59 = resolve_finetune_schedule_segment(cfg, loop_idx=59)
    seg60 = resolve_finetune_schedule_segment(cfg, loop_idx=60)

    assert seg0["timestep_start"] == 0 and seg0["timestep_end"] == 300
    assert seg29["timestep_start"] == 0 and seg29["timestep_end"] == 300
    assert seg30["timestep_start"] == 300 and seg30["timestep_end"] == 700
    assert seg59["timestep_start"] == 300 and seg59["timestep_end"] == 700
    assert seg60["timestep_start"] == 300 and seg60["timestep_end"] == 700

    win0 = resolve_finetune_timestep_window(cfg, loop_idx=0)
    win30 = resolve_finetune_timestep_window(cfg, loop_idx=30)
    win60 = resolve_finetune_timestep_window(cfg, loop_idx=60)

    assert win0[0] == 0 and win0[-1] == 299
    assert win30[0] == 300 and win30[-1] == 699
    assert win60[0] == 300 and win60[-1] == 699
    print(f"[PASS] test_resolve_finetune_timestep_window_schedule_segments")


def test_resolve_finetune_timestep_window_schedule_invalid_ranges():
    """Invalid schedule segments should raise clear errors."""
    from pipeline.timestep_window import resolve_finetune_schedule_segment

    bad_cfgs = [
        {
            "timesteps": 300,
            "timestep_schedule": [
                {"loop_start": 0, "loop_end": 30, "timestep_start": 0, "timestep_end": 300},
                {"loop_start": 20, "loop_end": 40, "timestep_start": 300, "timestep_end": 700},
            ],
        },
        {
            "timesteps": 300,
            "timestep_schedule": [
                {"loop_start": 0, "loop_end": 30, "timestep_start": 700, "timestep_end": 1001},
            ],
        },
        {
            "timesteps": 300,
            "timestep_schedule": [
                {"loop_start": 10, "loop_end": 30, "timestep_start": 0, "timestep_end": 300},
            ],
        },
    ]

    for bad_cfg in bad_cfgs:
        cfg = OmegaConf.create(bad_cfg)
        try:
            resolve_finetune_schedule_segment(cfg, loop_idx=0)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for schedule config {bad_cfg}")

    print(f"[PASS] test_resolve_finetune_timestep_window_schedule_invalid_ranges")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for models/mattergen/grpo_rollout.py
# ─────────────────────────────────────────────────────────────────────────────

def test_branch_rollout_tree_propagation():
    """BranchRolloutTree should propagate leaf rewards bottom-up correctly."""
    from models.mattergen.grpo_rollout import BranchRolloutTree

    tree = BranchRolloutTree(branch_steps=[300, 500, 800], branch_factors=[4, 4, 2])

    # Build a tiny 1-root, 2-branch, 2-leaf tree for testing
    root = tree.new_node(parent_id=None, branch_depth=0, window_start=0, window_end=300, root_idx=0)
    child_a = tree.new_node(parent_id=root.node_id, branch_depth=1, window_start=300, window_end=800, root_idx=0)
    child_b = tree.new_node(parent_id=root.node_id, branch_depth=1, window_start=300, window_end=800, root_idx=0)
    leaf_a1 = tree.new_node(parent_id=child_a.node_id, branch_depth=2, window_start=800, window_end=1000, root_idx=0)
    leaf_a2 = tree.new_node(parent_id=child_a.node_id, branch_depth=2, window_start=800, window_end=1000, root_idx=0)
    leaf_b1 = tree.new_node(parent_id=child_b.node_id, branch_depth=2, window_start=800, window_end=1000, root_idx=0)
    leaf_b2 = tree.new_node(parent_id=child_b.node_id, branch_depth=2, window_start=800, window_end=1000, root_idx=0)

    leaf_rewards = {
        leaf_a1.node_id: 0.8,
        leaf_a2.node_id: 0.6,
        leaf_b1.node_id: 0.2,
        leaf_b2.node_id: 0.4,
    }
    tree.propagate_rewards(leaf_rewards)

    # child_a branch_return = mean(0.8, 0.6) = 0.7
    assert abs(tree.nodes[child_a.node_id].branch_return - 0.7) < 1e-6
    # child_b branch_return = mean(0.2, 0.4) = 0.3
    assert abs(tree.nodes[child_b.node_id].branch_return - 0.3) < 1e-6
    # root branch_return = mean(0.8, 0.6, 0.2, 0.4) = 0.5
    assert abs(tree.nodes[root.node_id].branch_return - 0.5) < 1e-6

    print(f"[PASS] test_branch_rollout_tree_propagation")


def test_branch_rollout_tree_advantages():
    """Sibling group advantages should be zero-mean normalised."""
    from models.mattergen.grpo_rollout import BranchRolloutTree

    tree = BranchRolloutTree(branch_steps=[300], branch_factors=[4])
    root = tree.new_node(parent_id=None, branch_depth=0, window_start=0, window_end=300, root_idx=0)
    rewards = [0.2, 0.4, 0.6, 0.8]
    children = []
    for r in rewards:
        c = tree.new_node(parent_id=root.node_id, branch_depth=1, window_start=300, window_end=1000, root_idx=0)
        tree.nodes[c.node_id].final_rewards = [r]
        tree.nodes[c.node_id].branch_return = r
        children.append(c)

    tree.compute_advantages(adv_eps=1e-8, adv_clip=None)

    child_advantages = [tree.nodes[c.node_id].advantage for c in children]
    mean_adv = np.mean(child_advantages)
    assert abs(mean_adv) < 1e-6, f"Expected zero-mean advantages, got {mean_adv}"
    # higher reward -> higher advantage
    assert child_advantages[0] < child_advantages[-1]
    print(f"[PASS] test_branch_rollout_tree_advantages: {child_advantages}")


def test_flat_rollout_buffer():
    """FlatRolloutBuffer assign and retrieve should work correctly."""
    from models.mattergen.grpo_rollout import FlatRolloutBuffer, SampleRollout

    buf = FlatRolloutBuffer()
    for i in range(4):
        buf.add_rollout(SampleRollout(sample_idx=i))

    rewards = np.array([0.2, 0.5, 0.7, 0.9])
    buf.assign_rewards(rewards)
    np.testing.assert_allclose(buf.get_rewards(), rewards)

    advantages = np.array([-1.0, 0.0, 0.5, 1.5])
    buf.assign_advantages(advantages)
    assert buf.rollouts[0].advantage == -1.0
    assert buf.rollouts[3].advantage == 1.5

    buf.clear()
    assert len(buf) == 0
    print(f"[PASS] test_flat_rollout_buffer")


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight integration test (no MatterGen model needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_advantage_data_flow():
    """End-to-end advantage flow: rewards -> compute_advantage -> numpy array.

    Simulates what rl_step() does without needing a real MatterGen model.
    """
    from pipeline.grpo_utils import compute_advantage

    # Simulate reward_step output: 16 success samples
    np.random.seed(0)
    all_success_rewards = np.random.uniform(0.1, 0.9, size=16)

    # top-k selection (top 8)
    sort_idx = np.argsort(all_success_rewards)[::-1]
    topk_rewards = all_success_rewards[sort_idx[:8]]

    # GRPO advantage
    advantages = compute_advantage(
        rewards_train=topk_rewards,
        rewards_all=all_success_rewards,
        adv_eps=1e-8,
        adv_clip=5.0,
    )

    assert advantages.shape == (8,)
    # All top-k rewards are >= median, so most advantages should be >= 0
    assert (advantages >= 0).sum() >= 4
    # Advantage std should be > 0
    assert advantages.std() > 0

    print(f"[PASS] test_advantage_data_flow")
    print(f"       all_success mean={all_success_rewards.mean():.3f}")
    print(f"       topk_rewards: {topk_rewards[:4]}")
    print(f"       advantages:   {advantages[:4]}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_compute_advantage_basic,
        test_compute_advantage_normalisation,
        test_compute_advantage_clip,
        test_compute_advantage_small_group,
        test_log_functions_run,
        test_resolve_finetune_timestep_window_default,
        test_resolve_finetune_timestep_window_middle_range,
        test_resolve_finetune_timestep_window_invalid_ranges,
        test_resolve_finetune_timestep_window_schedule_segments,
        test_resolve_finetune_timestep_window_schedule_invalid_ranges,
        test_branch_rollout_tree_propagation,
        test_branch_rollout_tree_advantages,
        test_flat_rollout_buffer,
        test_advantage_data_flow,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
