"""Loss functions for multimodal embedding training (Qwen3-VL-Embedding paper).

- masked_infonce_loss: Eq. 1 — 5-term Z_i with false-negative masking (stage 1/2)
- cosent_loss:         Eq. 2 — ordering-preserving STS loss
- mrl_infonce_loss:    MRL wrapper over InfoNCE (Section 5.1.1)
- mrl_cosent_loss:     MRL wrapper over CoSent
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

DEFAULT_MRL_DIMS = [1024, 256, 64]
DEFAULT_TEMPERATURE = 0.02
FALSE_NEGATIVE_MARGIN = 0.1


def masked_infonce_loss(
    q: torch.Tensor, p: torch.Tensor,
    hn: Optional[torch.Tensor] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: int = 1,
) -> torch.Tensor:
    """InfoNCE with 5-term Z_i and false-negative masking (paper Eq. 1).

    Z_i = exp(s(q_i,d_i+)/τ)                          [1: positive]
        + Σ_k m_ik·exp(s(q_i,d_{i,k}^-)/τ)            [2: hard negatives]
        + Σ_{j≠i} m_ij·exp(s(q_i,q_j)/τ)              [3: q-q, stage 1 only]
        + Σ_{j≠i} m_ij·exp(s(d_i+,d_j)/τ)             [4: d-d, stage 1 only]
        + Σ_{j≠i} m_ij·exp(s(q_i,d_j)/τ)              [5: q-d cross]

    m_ij = 0  if  s_ij > s(q_i,d_i+) + 0.1  or  d_j == d_i+

    Args:
        q: (B, D) L2-normalized query embeddings
        p: (B, D) L2-normalized positive doc embeddings
        hn: (B*K, D) or None — hard negatives (K per query, flattened)
        stage: 1 = all 5 terms; 2 = drops q-q and d-d
    """
    B, device = q.shape[0], q.device
    margin = FALSE_NEGATIVE_MARGIN

    pos = (q * p).sum(-1)                              # (B,)
    Z = torch.exp(pos / temperature)                   # start with positive

    eye = torch.eye(B, device=device, dtype=torch.bool)

    def _masked_contrib(scores, ref=pos):
        mask = (scores <= ref.unsqueeze(1) + margin) & ~eye
        return (mask.float() * torch.exp(scores / temperature)).sum(1)

    # [2] hard negatives
    if hn is not None and hn.shape[0] > 0 and hn.shape[0] % B == 0:
        K = hn.shape[0] // B
        hn_scores = torch.bmm(hn.view(B, K, -1), q.unsqueeze(-1)).squeeze(-1)
        hn_mask = (hn_scores <= pos.unsqueeze(1) + margin).float()
        Z = Z + (hn_mask * torch.exp(hn_scores / temperature)).sum(1)

    # [5] q-d cross (always)
    Z = Z + _masked_contrib(q @ p.T)

    if stage == 1:
        # [3] q-q  and  [4] d-d
        Z = Z + _masked_contrib(q @ q.T)
        Z = Z + _masked_contrib(p @ p.T)

    return -torch.log(torch.exp(pos / temperature) / (Z + 1e-12)).mean()


def cosent_loss(
    a: torch.Tensor, b: torch.Tensor, scores: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """CoSent loss for STS (paper Eq. 2).

    L = log(1 + Σ_{s_i>s_j} exp((cos_j - cos_i) / τ))

    Args:
        a, b: (B, D) L2-normalized embeddings
        scores: (B,) ground-truth similarity scores
    """
    cos = (a * b).sum(-1) / temperature                # (B,)
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (B,B)
    cos_diff = cos.unsqueeze(1) - cos.unsqueeze(0)          # (B,B)
    pairs = (score_diff > 0).float()
    if pairs.sum() == 0:
        return torch.tensor(0.0, device=a.device, requires_grad=True)
    return torch.log(1.0 + (pairs * torch.exp(cos_diff)).sum())


# ---------------------------------------------------------------------------
# MRL wrappers (Section 5.1.1)
# ---------------------------------------------------------------------------

def _mrl_loop(loss_fn, emb_a, emb_b, dims, **kw):
    total = torch.tensor(0.0, device=emb_a.device)
    for d in dims:
        a = F.normalize(emb_a[:, :d], dim=-1)
        b = F.normalize(emb_b[:, :d], dim=-1)
        total = total + loss_fn(a, b, **kw)
    return total / len(dims)


def mrl_infonce_loss(
    q: torch.Tensor, p: torch.Tensor,
    hn: Optional[torch.Tensor] = None,
    mrl_dims: List[int] = DEFAULT_MRL_DIMS,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: int = 1,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=q.device)
    for d in mrl_dims:
        qd = F.normalize(q[:, :d], dim=-1)
        pd = F.normalize(p[:, :d], dim=-1)
        hd = F.normalize(hn[:, :d], dim=-1) if hn is not None and hn.shape[0] > 0 else None
        total = total + masked_infonce_loss(qd, pd, hd, temperature=temperature, stage=stage)
    return total / len(mrl_dims)


def mrl_cosent_loss(
    a: torch.Tensor, b: torch.Tensor, scores: torch.Tensor,
    mrl_dims: List[int] = DEFAULT_MRL_DIMS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    return _mrl_loop(cosent_loss, a, b, mrl_dims, scores=scores, temperature=temperature)
