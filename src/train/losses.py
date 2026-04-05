"""
Loss functions for multimodal embedding training, following the Qwen3-VL-Embedding paper.

Implements:
- Masked InfoNCE (Stage 1 and Stage 2 variants) with false-negative masking
- Classification contrastive loss
- CoSent loss for Semantic Textual Similarity (STS)
- Matryoshka Representation Learning (MRL) wrapper
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


DEFAULT_MRL_DIMS = [1024, 768, 512, 256, 128, 64]
DEFAULT_TEMPERATURE = 0.02
FALSE_NEGATIVE_MARGIN = 0.1


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between all rows of a and b."""
    return a @ b.T


def masked_infonce_loss(
    query_embs: torch.Tensor,
    pos_doc_embs: torch.Tensor,
    hard_neg_embs: Optional[torch.Tensor] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: int = 1,
) -> torch.Tensor:
    """
    InfoNCE loss from the Qwen3-VL-Embedding paper (Equation 1).

    Z_i = exp(s(q_i, d_i+)/tau)                              [positive]
        + sum_k m_ik * exp(s(q_i, d_{i,k}^-)/tau)            [hard negatives]
        + sum_{j!=i} m_ij * exp(s(q_i, q_j)/tau)             [in-batch queries]       (Stage 1 only)
        + sum_{j!=i} m_ij * exp(s(d_i+, d_j)/tau)            [in-batch docs vs d_i+]  (Stage 1 only)
        + sum_{j!=i} m_ij * exp(s(q_i, d_j)/tau)             [in-batch docs vs q_i]

    m_ij = 0 if s_ij > s(q_i, d_i+) + 0.1 or d_j == d_i+, else 1

    Args:
        query_embs: (B, D) normalized query embeddings
        pos_doc_embs: (B, D) normalized positive document embeddings
        hard_neg_embs: (B*K, D) or None, normalized hard negative embeddings
                       If provided, assumed K hard negatives per query, flattened.
                       K = hard_neg_embs.shape[0] // query_embs.shape[0]
        temperature: temperature parameter tau
        stage: 1 includes q-q and d-d terms; 2 removes them
    """
    B = query_embs.shape[0]
    device = query_embs.device

    pos_scores = (query_embs * pos_doc_embs).sum(dim=-1)  # (B,)
    pos_exp = torch.exp(pos_scores / temperature)  # (B,)

    Z = pos_exp.clone()  # (B,)

    # --- Hard negatives ---
    if hard_neg_embs is not None and hard_neg_embs.shape[0] > 0 and hard_neg_embs.shape[0] % B == 0:
        K = hard_neg_embs.shape[0] // B
        hard_neg_reshaped = hard_neg_embs.view(B, K, -1)  # (B, K, D)
        qhn_scores = torch.bmm(hard_neg_reshaped, query_embs.unsqueeze(-1)).squeeze(-1)  # (B, K)
        mask_hn = (qhn_scores <= (pos_scores.unsqueeze(1) + FALSE_NEGATIVE_MARGIN)).float()
        Z = Z + (mask_hn * torch.exp(qhn_scores / temperature)).sum(dim=1)

    # --- In-batch docs vs q_i: s(q_i, d_j) for j != i ---
    qd_scores = _cosine_sim(query_embs, pos_doc_embs)  # (B, B)
    identity_mask = torch.eye(B, device=device, dtype=torch.bool)
    fn_mask_qd = (qd_scores <= (pos_scores.unsqueeze(1) + FALSE_NEGATIVE_MARGIN)) & ~identity_mask
    qd_contrib = (fn_mask_qd.float() * torch.exp(qd_scores / temperature))
    qd_contrib = qd_contrib.masked_fill(identity_mask, 0.0)
    Z = Z + qd_contrib.sum(dim=1)

    if stage == 1:
        # --- In-batch queries: s(q_i, q_j) for j != i ---
        qq_scores = _cosine_sim(query_embs, query_embs)  # (B, B)
        fn_mask_qq = (qq_scores <= (pos_scores.unsqueeze(1) + FALSE_NEGATIVE_MARGIN)) & ~identity_mask
        qq_contrib = fn_mask_qq.float() * torch.exp(qq_scores / temperature)
        qq_contrib = qq_contrib.masked_fill(identity_mask, 0.0)
        Z = Z + qq_contrib.sum(dim=1)

        # --- In-batch docs vs d_i+: s(d_i+, d_j) for j != i ---
        dd_scores = _cosine_sim(pos_doc_embs, pos_doc_embs)  # (B, B)
        fn_mask_dd = (dd_scores <= (pos_scores.unsqueeze(1) + FALSE_NEGATIVE_MARGIN)) & ~identity_mask
        dd_contrib = fn_mask_dd.float() * torch.exp(dd_scores / temperature)
        dd_contrib = dd_contrib.masked_fill(identity_mask, 0.0)
        Z = Z + dd_contrib.sum(dim=1)

    loss = -torch.log(pos_exp / (Z + 1e-12)).mean()
    return loss


def classification_contrastive_loss(
    query_embs: torch.Tensor,
    pos_label_embs: torch.Tensor,
    neg_label_embs: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """
    Contrastive loss for classification data (paper Section 5.1).

    Negative samples are restricted to explicitly incorrect labels for the
    same query; other in-batch labels are ignored.
    """
    B = query_embs.shape[0]

    pos_scores = (query_embs * pos_label_embs).sum(dim=-1)  # (B,)
    pos_exp = torch.exp(pos_scores / temperature)

    if neg_label_embs.dim() == 2 and neg_label_embs.shape[0] == B:
        neg_scores = (query_embs * neg_label_embs).sum(dim=-1)  # (B,)
        neg_exp = torch.exp(neg_scores / temperature)
    elif neg_label_embs.dim() == 3:
        neg_scores = torch.bmm(neg_label_embs, query_embs.unsqueeze(-1)).squeeze(-1)
        neg_exp = torch.exp(neg_scores / temperature).sum(dim=1)
    else:
        neg_scores = (query_embs * neg_label_embs).sum(dim=-1)
        neg_exp = torch.exp(neg_scores / temperature)

    loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-12)).mean()
    return loss


def cosent_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    scores: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """
    CoSent loss for STS data (paper Equation 2).

    L_sts = log(1 + sum_{s_hat(i,j) > s_hat(m,n)} exp((cos(m,n) - cos(i,j)) / tau))

    where s_hat is the ground-truth score and cos is the cosine similarity of embeddings.

    Args:
        embeddings_a: (B, D) normalized embeddings for sentence A
        embeddings_b: (B, D) normalized embeddings for sentence B
        scores: (B,) ground-truth similarity scores
        temperature: temperature parameter tau
    """
    cos_sims = (embeddings_a * embeddings_b).sum(dim=-1)  # (B,)
    cos_sims = cos_sims / temperature

    # All pairs (i, j) where score_i > score_j
    score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # (B, B)
    cos_diff = cos_sims.unsqueeze(1) - cos_sims.unsqueeze(0)  # (B, B)

    # For pairs where score_i > score_j, we want cos_i > cos_j,
    # so we penalize cos_j - cos_i (i.e. cos_diff where score_diff > 0)
    pair_mask = (score_diff > 0).float()

    if pair_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings_a.device, requires_grad=True)

    loss = torch.log(1.0 + (pair_mask * torch.exp(cos_diff)).sum())
    return loss


def mrl_loss(
    loss_fn,
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
    mrl_dims: List[int] = DEFAULT_MRL_DIMS,
    **loss_kwargs,
) -> torch.Tensor:
    """
    Matryoshka Representation Learning wrapper (paper Section 5.1.1).

    Computes the given loss function on the full embedding and on each
    truncated lower-dimensional prefix, then averages.
    """
    total_loss = torch.tensor(0.0, device=query_embs.device)
    for d in mrl_dims:
        q_trunc = F.normalize(query_embs[:, :d], dim=-1)
        d_trunc = F.normalize(doc_embs[:, :d], dim=-1)
        total_loss = total_loss + loss_fn(q_trunc, d_trunc, **loss_kwargs)
    return total_loss / len(mrl_dims)


def mrl_infonce_loss(
    query_embs: torch.Tensor,
    pos_doc_embs: torch.Tensor,
    hard_neg_embs: Optional[torch.Tensor] = None,
    mrl_dims: List[int] = DEFAULT_MRL_DIMS,
    temperature: float = DEFAULT_TEMPERATURE,
    stage: int = 1,
) -> torch.Tensor:
    """MRL-wrapped InfoNCE with hard negatives and false-negative masking."""
    total_loss = torch.tensor(0.0, device=query_embs.device)
    for d in mrl_dims:
        q_trunc = F.normalize(query_embs[:, :d], dim=-1)
        p_trunc = F.normalize(pos_doc_embs[:, :d], dim=-1)
        hn_trunc = None
        if hard_neg_embs is not None and hard_neg_embs.shape[0] > 0:
            hn_trunc = F.normalize(hard_neg_embs[:, :d], dim=-1)
        total_loss = total_loss + masked_infonce_loss(
            q_trunc, p_trunc, hn_trunc,
            temperature=temperature,
            stage=stage,
        )
    return total_loss / len(mrl_dims)


def mrl_cosent_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    scores: torch.Tensor,
    mrl_dims: List[int] = DEFAULT_MRL_DIMS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """MRL-wrapped CoSent loss for STS data."""
    total_loss = torch.tensor(0.0, device=embeddings_a.device)
    for d in mrl_dims:
        a_trunc = F.normalize(embeddings_a[:, :d], dim=-1)
        b_trunc = F.normalize(embeddings_b[:, :d], dim=-1)
        total_loss = total_loss + cosent_loss(a_trunc, b_trunc, scores, temperature)
    return total_loss / len(mrl_dims)
