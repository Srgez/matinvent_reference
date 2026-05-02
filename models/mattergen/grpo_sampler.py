"""
GRPO-aware MatterGen sampler (Phase 2 & 3).

This module vendors the PredictorCorrector sampling logic locally and extends
it to record per-step transition logprobs for strict GRPO and branch GRPO.

External MatterGen package is still used for:
  - Score-model forward pass (DiffusionModule.score_fn)
  - Corruption process (MultiCorruption, SDE coefficients)
  - Prior sampling (_sample_prior)
  - Batch masking (_mask_replace)

Not modified in upstream MatterGen:
  - pc_sampler.py
  - predictors.py
  - predictors_correctors.py
  - d3pm_predictors_correctors.py

Phase 2 (strict GRPO):
  sample_with_rollout() -> (mean_batch, FlatRolloutBuffer)

Phase 3 (branch GRPO):
  sample_branch_rollout() -> (leaf_batches, BranchRolloutTree)
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_sampler import _sample_prior, _mask_replace
from mattergen.diffusion.sampling.predictors_correctors import LangevinCorrector

from models.mattergen.grpo_predictors import GRPOAncestralPredictor, GRPOD3PMPredictor
from models.mattergen.grpo_rollout import (
    BranchRolloutTree,
    FlatRolloutBuffer,
    SampleRollout,
    TransitionRecord,
)


class GRPOSampler:
    """MatterGen sampler that records rollout logprobs for GRPO training.

    Instantiation is analogous to PredictorCorrector.from_pl_module().

    Phase 2 usage (strict GRPO):
    ::
        sampler = GRPOSampler.from_pl_module(agent, N=1000, n_steps_corrector=1)
        mean_batch, rollout = sampler.sample_with_rollout(conditioning_data)
        # ... compute rewards ...
        rollout.assign_rewards(rewards)          # from reward_step()
        rollout.assign_advantages(advantages)    # from compute_advantage()
        # train from rollout ...

    Phase 3 usage (branch GRPO):
    ::
        sampler = GRPOSampler.from_pl_module(agent, ...)
        leaf_batches, tree = sampler.sample_branch_rollout(
            conditioning_data,
            branch_steps=[300, 500, 800],
            branch_factors=[4, 4, 2],
        )
        # ... compute leaf rewards ...
        tree.propagate_rewards(leaf_rewards)
        tree.compute_advantages()
        # train per window ...

    Correctors:
        Langevin correctors are supported during sampling to preserve MatterGen's
        original sampling quality.  Their transitions are NOT logged (logprob_scope
        is always 'predictor_only' in this version).
    """

    def __init__(
        self,
        *,
        diffusion_module,
        device: torch.device,
        N: int = 1000,
        eps_t: float = 1e-3,
        max_t: Optional[float] = None,
        n_steps_corrector: int = 0,
        corrector_snr: float = 0.2,
    ):
        self._diffusion_module = diffusion_module
        self._device = device
        self.N = N
        self.eps_t = eps_t
        self.n_steps_corrector = n_steps_corrector
        self.corrector_snr = corrector_snr

        mc: MultiCorruption = diffusion_module.corruption
        self._multi_corruption = mc
        self._max_t = max_t or mc.T

        # Build GRPO-aware predictors for each corrupted field
        self._grpo_predictors: Dict[str, GRPOAncestralPredictor | GRPOD3PMPredictor] = {}
        for field_name, corruption in mc.corruptions.items():
            try:
                pred = GRPOAncestralPredictor(corruption=corruption, score_fn=None)
            except Exception:
                try:
                    pred = GRPOD3PMPredictor(corruption=corruption, score_fn=None)
                except Exception:
                    logging.warning(
                        f"GRPOSampler: could not build GRPO predictor for field '{field_name}'. "
                        "This field will not contribute to logp_old."
                    )
                    pred = None
            if pred is not None:
                self._grpo_predictors[field_name] = pred

        # Build correctors (standard, no logprob)
        self._correctors: Dict[str, LangevinCorrector] = {}
        if n_steps_corrector > 0:
            for field_name, corruption in mc.corruptions.items():
                try:
                    corr = LangevinCorrector(
                        corruption=corruption,
                        score_fn=None,
                        n_steps=n_steps_corrector,
                        snr=corrector_snr,
                    )
                    self._correctors[field_name] = corr
                except Exception:
                    pass

        logging.info(
            f"GRPOSampler: GRPO predictors for fields {list(self._grpo_predictors.keys())}; "
            f"correctors for fields {list(self._correctors.keys())}"
        )

    @classmethod
    def from_pl_module(
        cls,
        pl_module: DiffusionLightningModule,
        **kwargs,
    ) -> GRPOSampler:
        return cls(
            diffusion_module=pl_module.diffusion_module,
            device=pl_module.device,
            **kwargs,
        )

    def _score_fn(self, batch: BatchedData, t: torch.Tensor) -> BatchedData:
        return self._diffusion_module.score_fn(batch, t)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Flat rollout sampling
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_with_rollout(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
    ) -> Tuple[BatchedData, FlatRolloutBuffer]:
        """Denoise from prior to eps_t while recording predictor logprobs.

        Returns:
            mean_batch:     final denoised batch (same semantics as MatterGenSampler)
            rollout_buffer: FlatRolloutBuffer containing one SampleRollout per sample
        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()

        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}

        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        mean_batch = batch.clone()

        # Initialise per-sample rollouts
        n_samples = batch.get_batch_size()
        rollout_buffer = FlatRolloutBuffer()
        sample_rollouts = [SampleRollout(sample_idx=i) for i in range(n_samples)]

        timesteps = torch.linspace(self._max_t, self.eps_t, self.N, device=self._device)
        dt = -torch.tensor(
            (self._max_t - self.eps_t) / (self.N - 1), device=self._device
        )

        for k in self._grpo_predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            t = torch.full((n_samples,), timesteps[i], device=self._device)

            # Corrector steps (no logprob recording)
            if self._correctors:
                for _ in range(self.n_steps_corrector):
                    score = self._score_fn(batch, t)
                    fns = {k: c.step_given_score for k, c in self._correctors.items()}
                    samples_means = apply(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch,
                        mean_batch=mean_batch, mask=mask,
                    )

            # Predictor steps with logprob recording
            score = self._score_fn(batch, t)
            batch_idx_map = self._multi_corruption._get_batch_indices(batch)

            new_samples: Dict[str, torch.Tensor] = {}
            new_means: Dict[str, torch.Tensor] = {}
            step_logp_total = torch.zeros(n_samples, device=self._device)

            for field_name, predictor in self._grpo_predictors.items():
                x = batch[field_name]
                s = score[field_name]
                bidx = batch_idx_map.get(field_name)

                sample_f, mean_f, logp_f = predictor.update_given_score_with_logprob(
                    x=x, t=t, dt=dt, batch_idx=bidx, score=s, batch=batch,
                )
                new_samples[field_name] = sample_f
                new_means[field_name] = mean_f
                step_logp_total = step_logp_total + logp_f  # sum over fields

            # Apply masks and update batch
            samples_means_dict = {
                k: (new_samples[k], new_means[k]) for k in new_samples
            }
            batch, mean_batch = _mask_replace(
                samples_means=samples_means_dict,
                batch=batch, mean_batch=mean_batch, mask=mask,
            )

            # Store one TransitionRecord per sample
            # We store the combined (summed) logp_old across all fields.
            # x_t and x_next are stored for pos only here as representative;
            # phase 2 training re-runs the full score model anyway.
            # TODO(phase2): store per-field x_t/x_next for precise recompute.
            transition = TransitionRecord(
                timestep=float(timesteps[i]),
                dt=float(dt),
                x_t=batch["pos"].detach().cpu() if "pos" in new_samples else None,
                x_next=new_samples.get("pos", torch.zeros(1)).detach().cpu(),
                logp_old=step_logp_total.detach().cpu(),
                batch_idx=batch_idx_map.get("pos", None),
                batch_size=n_samples,
            )
            for rollout in sample_rollouts:
                rollout.transitions.append(transition)

        for rollout in sample_rollouts:
            rollout_buffer.add_rollout(rollout)

        return mean_batch, rollout_buffer

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Branch rollout sampling
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_branch_rollout(
        self,
        conditioning_data: BatchedData,
        branch_steps: List[int],
        branch_factors: List[int],
        mask: Mapping[str, torch.Tensor] | None = None,
    ) -> Tuple[List[BatchedData], BranchRolloutTree]:
        """Denoise from prior with branching at specified timesteps.

        At each branch_step, the current batch state is cloned into
        branch_factor copies, which then continue independently.

        Returns:
            leaf_batches: list of final denoised batches for each leaf node
            branch_tree:  BranchRolloutTree with all nodes and transitions

        Note: Phase 3 is scaffolded here.  The implementation is a TODO.
        """
        # TODO (phase 3): implement full branch rollout
        # Outline:
        # 1. Sample root batch from prior.
        # 2. Run denoising from T to branch_steps[0]:
        #    - Record predictor transitions into root BranchNodes.
        # 3. At branch_steps[0]: clone batch state branch_factors[0] times.
        # 4. For each clone, run from branch_steps[0] to branch_steps[1]:
        #    - Record transitions into depth-1 BranchNodes.
        # 5. At branch_steps[1]: clone again by branch_factors[1].
        # 6. ... continue until all leaves are reached.
        # 7. Return leaf_batches and BranchRolloutTree.
        raise NotImplementedError(
            "GRPOSampler.sample_branch_rollout is Phase 3 (branch GRPO). "
            "Implement after Phase 2 (strict GRPO) is validated."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Training: recompute logprob for ratio computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_logp_new(
        self,
        transition: TransitionRecord,
        agent: DiffusionLightningModule,
    ) -> torch.Tensor:
        """Recompute log p_new(x_next | x_t) for a stored transition.

        Called during training (with grad) to get logp_new for ratio computation.

        Args:
            transition: stored from rollout buffer (x_t, x_next, logp_old, t, batch_idx)
            agent:      current (updating) agent model

        Returns:
            logp_new: shape (n_samples,), requires_grad=True

        TODO (phase 2): implement full recomputation.
        Currently raises NotImplementedError.
        """
        # Outline:
        # 1. Move x_t, batch_idx to device.
        # 2. Forward agent.score_fn(batch_with_x_t, t) -> score.
        # 3. For each field, call predictor.recompute_logprob(x_t, x_next, score).
        # 4. Sum over fields -> logp_new.
        raise NotImplementedError(
            "GRPOSampler.compute_logp_new is Phase 2 (strict GRPO). "
            "Implement after Phase 1 (GRPO-lite) is validated."
        )
