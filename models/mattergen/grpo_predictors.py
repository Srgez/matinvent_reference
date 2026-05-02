"""
GRPO-aware predictor wrappers for MatterGen (Phase 2 - Strict GRPO).

Vendors the necessary MatterGen predictor logic locally without modifying
the upstream MatterGen package.  Extends each predictor to return per-sample
log-probability alongside the sampled next state.

Correctors (Langevin): kept as-is.  Their logprob is not tracked in phase 2
(logprob_scope = 'predictor_only').  This is a deliberate trade-off to avoid
the complexities of Langevin corrector log-densities.

Field-level combination:
    logp_total = logp_pos + logp_cell + logp_atomic_numbers   (sum per sample)

The individual fields are tracked separately for diagnostic logging and can
be individually weighted in future iterations.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal, Categorical
from torch_scatter import scatter

from mattergen.diffusion.sampling.predictors import AncestralSamplingPredictor
from mattergen.diffusion.d3pm.d3pm_predictors_correctors import D3PMAncestralSamplingPredictor
from mattergen.diffusion.data.batched_data import BatchedData


# ─────────────────────────────────────────────────────────────────────────────
# Continuous-field predictor (pos, cell)
# ─────────────────────────────────────────────────────────────────────────────

class GRPOAncestralPredictor(AncestralSamplingPredictor):
    """AncestralSamplingPredictor extended with per-step logprob output.

    Subclasses the upstream predictor so all coefficient computations
    (alpha_t, sigma_t, etc.) remain identical.
    """

    def update_given_score_with_logprob(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        batch: BatchedData | None,
    ):
        """Predictor update that additionally returns log p(x_next | x_t).

        Returns:
            sample:          shape same as x, sampled next state
            mean:            shape same as x, deterministic next state
            logp_per_sample: shape (n_samples,), sum of logprobs over atoms/dims
        """
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x, t=t, dt=dt, batch_idx=batch_idx, batch=batch,
        )
        z = torch.randn_like(x)
        mean = x_coeff * x + score_coeff * score
        sample = mean + std * z

        # log p_old(sample | x_t) = Normal(mean, std).log_prob(sample)
        # Detach mean/std so gradients do not flow through old policy.
        safe_std = std.clamp(min=1e-12)
        logp_elem = Normal(mean.detach(), safe_std.detach()).log_prob(sample.detach())

        logp_per_sample = _aggregate_logp(logp_elem, batch_idx)

        return sample, mean, logp_per_sample

    def recompute_logprob(
        self,
        *,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> torch.Tensor:
        """Recompute log p_NEW(x_next | x_t) for a stored transition.

        Called during training to obtain logp_new for ratio computation.
        score here is from the CURRENT (updated) agent.

        Returns:
            logp_per_sample: shape (n_samples,)
        """
        x_coeff, score_coeff, std = self._get_coeffs(
            x=x_t, t=t, dt=dt, batch_idx=batch_idx, batch=batch,
        )
        mean = x_coeff * x_t + score_coeff * score
        safe_std = std.clamp(min=1e-12)
        logp_elem = Normal(mean, safe_std).log_prob(x_next)

        return _aggregate_logp(logp_elem, batch_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete-field predictor (atomic_numbers via D3PM)
# ─────────────────────────────────────────────────────────────────────────────

class GRPOD3PMPredictor(D3PMAncestralSamplingPredictor):
    """D3PMAncestralSamplingPredictor extended with per-step logprob output."""

    def update_given_score_with_logprob(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        batch: BatchedData | None,
    ):
        """Predictor update + log p(x_next | x_t) for discrete atom types.

        Returns:
            sample:          sampled next atom types
            mean:            argmax expected atom types
            logp_per_sample: shape (n_samples,)
        """
        sample, mean = self.update_given_score(
            x=x, t=t, dt=dt, batch_idx=batch_idx, score=score, batch=batch,
        )
        # score == class_logits (already in log-space if predict_x0, else raw)
        logp_per_sample = self._logp_from_logits_and_sample(score, sample, batch_idx)
        return sample, mean, logp_per_sample

    def recompute_logprob(
        self,
        *,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor | None,
        score: torch.Tensor,
        batch: BatchedData | None,
    ) -> torch.Tensor:
        """Recompute log p_NEW(x_next | x_t) using current agent's score/logits.

        Returns:
            logp_per_sample: shape (n_samples,)
        """
        return self._logp_from_logits_and_sample(score, x_next, batch_idx)

    def _logp_from_logits_and_sample(
        self,
        class_logits: torch.Tensor,
        sample: torch.Tensor,
        batch_idx: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Compute Categorical log-prob for sampled atom types.

        MatterGen's D3PM stores atom types with a 1-based offset
        (_to_zero_based/_to_non_zero_based).  We convert back to zero-based
        before evaluating the categorical distribution.
        """
        x_zero_based = self.corruption._to_zero_based(sample)
        logp_per_atom = Categorical(logits=class_logits).log_prob(x_zero_based)

        if batch_idx is not None:
            return scatter(logp_per_atom, batch_idx, dim=0, reduce='sum')
        return logp_per_atom.sum(dim=0, keepdim=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_logp(
    logp_elem: torch.Tensor,
    batch_idx: torch.LongTensor | None,
) -> torch.Tensor:
    """Reduce element-wise logp to per-sample scalar.

    logp_elem may have shape:
      (n_atoms, d)  for atom-level fields (pos)
      (n_samples, d1, d2)  for sample-level fields (cell)

    The function sums over all non-batch dimensions, then optionally scatters
    over graph batch indices.
    """
    # Flatten all non-first dims
    logp_flat = logp_elem.reshape(logp_elem.shape[0], -1).sum(dim=-1)

    if batch_idx is not None:
        return scatter(logp_flat, batch_idx, dim=0, reduce='sum')
    return logp_flat
