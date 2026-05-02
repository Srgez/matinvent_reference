"""
Rollout data structures for GRPO training.

Phase 2 (strict GRPO):  FlatRolloutBuffer
Phase 3 (branch GRPO):  BranchRolloutTree + BranchNode

Credit-assignment design (branch GRPO, batchsize=4, factors {300:4, 500:4, 800:2}):

    Tree structure per root:
        root (step 0-300)
          └── [x4] step-300 node  (step 300-500)
                └── [x4] step-500 node  (step 500-800)
                      └── [x2] step-800 node  (step 800-1000)  ← leaf

    Advantage computation per window:
        0-300   : branch_return = mean(32 descendant leaf rewards)
                  advantage    = normalise over 4 step-300 siblings
        300-500 : branch_return = mean(8 descendant leaf rewards)
                  advantage    = normalise over 4 step-500 siblings
        500-800 : branch_return = mean(2 leaf rewards)
                  advantage    = normalise over 4 step-800 siblings (per 500-parent)
        800-1000: branch_return = single leaf reward
                  advantage    = normalise over 2 sibling leaves (per 800-parent)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core transition record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransitionRecord:
    """One predictor step for a (mini-)batch of samples.

    x_t and x_next store the raw tensors for ONE field (pos, cell, or
    atomic_numbers).  Logprob contributions from all fields are summed
    into logp_old at record time (see GRPOSampler).
    """
    timestep: float                           # diffusion time t (scalar)
    dt: float                                 # step size
    x_t: torch.Tensor                         # state before predictor step
    x_next: torch.Tensor                      # state after predictor step (stored action)
    logp_old: torch.Tensor                    # log p_old, shape (n_samples,) – sum of all fields
    batch_idx: Optional[torch.LongTensor]     # atom→sample mapping, shape (n_atoms,)
    batch_size: int                           # number of samples in this mini-batch
    conditioning_data: object = None          # optional: conditioning info for recompute


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Flat rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleRollout:
    """Complete denoising trajectory (all predictor steps) for one sample."""
    sample_idx: int
    transitions: List[TransitionRecord] = field(default_factory=list)
    reward: float = 0.0
    advantage: float = 0.0


class FlatRolloutBuffer:
    """Stores all predictor transitions for one RL step (strict GRPO).

    Typical lifecycle:
        1. GRPOSampler fills this buffer during sampling.
        2. reward_step() fills rewards via assign_rewards().
        3. grpo_utils.compute_advantage() fills advantages.
        4. Build a DataLoader over get_all_transitions() for training.
        5. Call clear() before next RL step.
    """

    def __init__(self):
        self.rollouts: List[SampleRollout] = []

    def add_rollout(self, rollout: SampleRollout):
        self.rollouts.append(rollout)

    def assign_rewards(self, rewards: np.ndarray):
        """Assign per-sample rewards (in the same order as rollouts)."""
        assert len(rewards) == len(self.rollouts), (
            f"rewards length {len(rewards)} != buffer length {len(self.rollouts)}"
        )
        for rollout, r in zip(self.rollouts, rewards):
            rollout.reward = float(r)

    def assign_advantages(self, advantages: np.ndarray):
        """Assign per-sample advantages computed by compute_advantage()."""
        assert len(advantages) == len(self.rollouts)
        for rollout, a in zip(self.rollouts, advantages):
            rollout.advantage = float(a)

    def get_all_transitions(self) -> List[Tuple[TransitionRecord, float]]:
        """Flatten all transitions into a list of (transition, advantage) pairs."""
        result = []
        for rollout in self.rollouts:
            for transition in rollout.transitions:
                result.append((transition, rollout.advantage))
        return result

    def get_rewards(self) -> np.ndarray:
        return np.array([r.reward for r in self.rollouts])

    def clear(self):
        self.rollouts.clear()

    def __len__(self):
        return len(self.rollouts)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Branch rollout tree
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BranchNode:
    """One node in the branch rollout tree.

    A node spans the transitions in the time window (window_start, window_end).
    branch_depth=0 means root-level (0..branch_steps[0]).
    Leaf nodes have no children and carry a single final reward.
    """
    node_id: int
    parent_id: Optional[int]          # None for root nodes
    branch_depth: int                  # 0, 1, 2, ... matches index in branch_steps
    window_start: int                  # inclusive timestep index
    window_end: int                    # exclusive timestep index
    root_idx: int                      # which original root batch element

    # Transitions recorded within this node's window
    transitions: List[TransitionRecord] = field(default_factory=list)

    # Tree links (populated by BranchRolloutTree.build_tree)
    children_ids: List[int] = field(default_factory=list)

    # Return / advantage (populated after reward computation)
    final_rewards: List[float] = field(default_factory=list)
    branch_return: float = 0.0         # mean of final_rewards over all descendant leaves
    advantage: float = 0.0             # normalised vs sibling nodes at same parent


class BranchRolloutTree:
    """Tree of rollout nodes for branch GRPO (Phase 3).

    Supports the branch configuration:
        batchsize=4, branch_steps=[300,500,800], branch_factors=[4,4,2]

    which yields 4 * 4*4*2 = 128 total leaf nodes.

    Usage:
        tree = BranchRolloutTree(branch_steps=[300,500,800], branch_factors=[4,4,2])
        # ... GRPOSampler fills tree.nodes with transitions ...
        tree.propagate_rewards(leaf_rewards)   # dict {node_id: reward}
        tree.compute_advantages()              # normalise within sibling groups
        # For each time window:
        transitions = tree.get_transitions_for_window(0, 300)
    """

    def __init__(self, branch_steps: List[int], branch_factors: List[int]):
        assert len(branch_steps) == len(branch_factors)
        self.branch_steps = branch_steps         # e.g. [300, 500, 800]
        self.branch_factors = branch_factors     # e.g. [4, 4, 2]
        self.nodes: Dict[int, BranchNode] = {}
        self._next_id = 0

    # ── Tree construction ──────────────────────────────────────────────────

    def new_node(
        self,
        parent_id: Optional[int],
        branch_depth: int,
        window_start: int,
        window_end: int,
        root_idx: int,
    ) -> BranchNode:
        """Allocate a new BranchNode and register it in the tree."""
        node = BranchNode(
            node_id=self._next_id,
            parent_id=parent_id,
            branch_depth=branch_depth,
            window_start=window_start,
            window_end=window_end,
            root_idx=root_idx,
        )
        self.nodes[self._next_id] = node
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(self._next_id)
        self._next_id += 1
        return node

    # ── Reward propagation ────────────────────────────────────────────────

    def propagate_rewards(self, leaf_rewards: Dict[int, float]):
        """Bottom-up: compute branch_return = mean(descendant leaf rewards).

        Args:
            leaf_rewards: {node_id: scalar reward} for leaf nodes only.
                          All leaf node IDs must be present.
        """
        for node_id, r in leaf_rewards.items():
            self.nodes[node_id].final_rewards = [r]

        # Process in reverse ID order so children are processed before parents.
        for node_id in sorted(self.nodes.keys(), reverse=True):
            node = self.nodes[node_id]
            if node.children_ids:
                node.final_rewards = self._collect_leaf_rewards(node_id)
            node.branch_return = float(np.mean(node.final_rewards)) if node.final_rewards else 0.0

    def _collect_leaf_rewards(self, node_id: int) -> List[float]:
        node = self.nodes[node_id]
        if not node.children_ids:
            return list(node.final_rewards)
        rewards = []
        for cid in node.children_ids:
            rewards.extend(self._collect_leaf_rewards(cid))
        return rewards

    # ── Advantage computation ─────────────────────────────────────────────

    def compute_advantages(self, adv_eps: float = 1e-8, adv_clip: float = 5.0):
        """Normalise branch_return within each sibling group -> set advantage.

        For each parent node, its children form a sibling group.
        Advantage = (branch_return - mean_siblings) / (std_siblings + eps),
        clipped to [-adv_clip, adv_clip].
        """
        from pipeline.grpo_utils import compute_advantage

        # Group siblings by parent_id
        parent_to_children: Dict[Optional[int], List[int]] = {}
        for node_id, node in self.nodes.items():
            parent_to_children.setdefault(node.parent_id, []).append(node_id)

        for parent_id, sibling_ids in parent_to_children.items():
            returns = np.array([self.nodes[sid].branch_return for sid in sibling_ids])
            # All-sibling baseline (group size can be 2 or 4)
            advantages = compute_advantage(
                rewards_train=returns,
                rewards_all=returns,
                adv_eps=adv_eps,
                adv_clip=adv_clip,
            )
            for sid, adv in zip(sibling_ids, advantages):
                self.nodes[sid].advantage = float(adv)

    # ── Transition retrieval for training ─────────────────────────────────

    def get_transitions_for_window(
        self,
        window_start: int,
        window_end: int,
    ) -> List[Tuple[TransitionRecord, float]]:
        """Get (transition, advantage) pairs for nodes whose window matches.

        Args:
            window_start: inclusive lower bound on node.window_start
            window_end:   exclusive upper bound on node.window_end

        Returns:
            List of (TransitionRecord, advantage_scalar) for training.
        """
        result = []
        for node in self.nodes.values():
            if node.window_start == window_start and node.window_end == window_end:
                for transition in node.transitions:
                    result.append((transition, node.advantage))
        return result

    def get_leaf_nodes(self) -> List[BranchNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [n for n in self.nodes.values() if not n.children_ids]

    def clear(self):
        self.nodes.clear()
        self._next_id = 0

    def summary(self) -> str:
        """Human-readable summary of the tree structure."""
        depth_counts: Dict[int, int] = {}
        for node in self.nodes.values():
            depth_counts[node.branch_depth] = depth_counts.get(node.branch_depth, 0) + 1
        lines = [f"BranchRolloutTree: {len(self.nodes)} total nodes"]
        for d, count in sorted(depth_counts.items()):
            lines.append(f"  depth {d}: {count} nodes")
        leaves = self.get_leaf_nodes()
        if leaves:
            returns = [n.branch_return for n in leaves]
            lines.append(
                f"  leaf rewards: mean={np.mean(returns):.4f}, "
                f"std={np.std(returns):.4f}, "
                f"min={np.min(returns):.4f}, max={np.max(returns):.4f}"
            )
        return "\n".join(lines)
