"""
X-TADDY: Baseline Explainability Methods
=========================================
Three post-hoc attribution baselines applied to the same trained TADDY model.
All baselines use the same Fidelity measurement protocol as X-TADDY for fair
comparison (embedding-space top-k masking).

Methods
-------
RandomExplainer    — random Shapley values (lower bound)
AttentionExplainer — TADDY's own Transformer attention weights as attributions
GradientExplainer  — L2 norm of input gradient w.r.t. embedding output
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class BaselineResult:
    """Result from a single baseline explanation."""
    method:        str
    fidelity:      float
    sparsity:      float
    phi:           np.ndarray   # per-token attribution scores


class RandomExplainer:
    """
    Lower-bound baseline: assigns uniformly random attribution values.
    Token selection for Fidelity is therefore uninformed.
    """

    def __init__(self, model, top_k: int = 5, random_state: int = 42):
        self.model = model
        self.top_k = top_k
        self.rng   = np.random.default_rng(random_state)
        self.model.eval()

    def explain_edges(
        self,
        int_emb, hop_emb, time_emb,
        edges: list,
    ) -> List[BaselineResult]:
        results = []
        D = int_emb[edges[0][1]].shape[1]

        for _, snap, ei in edges:
            phi = self.rng.standard_normal(D).astype(np.float32)

            with torch.no_grad():
                out, _ = self.model.forward(
                    int_emb[snap][ei:ei+1],
                    hop_emb[snap][ei:ei+1],
                    time_emb[snap][ei:ei+1],
                )
                true_score = torch.sigmoid(out).item()

            h_base = self._get_embedding(int_emb, hop_emb, time_emb, snap, ei)
            fidelity = self._fidelity(h_base, phi, true_score, D)
            results.append(BaselineResult(
                method='Random',
                fidelity=fidelity,
                sparsity=1.0 - self.top_k / D,
                phi=phi,
            ))
        return results

    def _get_embedding(self, int_emb, hop_emb, time_emb, snap, ei):
        with torch.no_grad():
            return self.model.transformer.embeddings(
                init_pos_ids=int_emb[snap][ei:ei+1],
                hop_dis_ids=hop_emb[snap][ei:ei+1],
                time_dis_ids=time_emb[snap][ei:ei+1],
            ).detach()

    def _fidelity(self, h_base, phi, true_score, D):
        top_k   = min(self.top_k, D)
        mask    = torch.zeros(D).bool()
        for idx in np.argsort(np.abs(phi))[-top_k:]:
            mask[idx] = True
        h = h_base.clone()
        active = mask
        if active.sum() > 0:
            h[0, ~active, :] = h[0, active, :].mean(dim=0)
        else:
            h[0, :, :] = 0.0
        with torch.no_grad():
            out, _ = self.model.forward_from_embeddings(h)
            topk_score = torch.sigmoid(out).item()
        return float(1.0 - abs(true_score - topk_score))


class AttentionExplainer:
    """
    Attention-based baseline: uses TADDY's last-layer Transformer attention
    weights (averaged over heads) as token attribution scores.

    As noted by Jain & Wallace (2019), attention weights are not guaranteed
    to be causally faithful, but they serve as a natural internal baseline.
    """

    def __init__(self, model, top_k: int = 5):
        self.model = model
        self.top_k = top_k
        self.model.eval()

    def explain_edges(
        self,
        int_emb, hop_emb, time_emb,
        edges: list,
    ) -> List[BaselineResult]:
        results = []
        D = int_emb[edges[0][1]].shape[1]

        for _, snap, ei in edges:
            ii = int_emb[snap][ei:ei+1]
            hh = hop_emb[snap][ei:ei+1]
            tt = time_emb[snap][ei:ei+1]

            with torch.no_grad():
                out, all_attn = self.model.forward(ii, hh, tt)
                true_score    = torch.sigmoid(out).item()

                # Last-layer attention: [1, H, T, T] → mean over heads
                # → mean attention received per token → shape [T]
                attn_last = all_attn[-1]              # [1, H, T, T]
                phi = attn_last[0].mean(dim=0).mean(dim=0).cpu().numpy()

            h_base   = self._get_embedding(int_emb, hop_emb, time_emb, snap, ei)
            fidelity = self._fidelity(h_base, phi, true_score, D)
            results.append(BaselineResult(
                method='Attention',
                fidelity=fidelity,
                sparsity=1.0 - self.top_k / D,
                phi=phi,
            ))
        return results

    def _get_embedding(self, int_emb, hop_emb, time_emb, snap, ei):
        with torch.no_grad():
            return self.model.transformer.embeddings(
                init_pos_ids=int_emb[snap][ei:ei+1],
                hop_dis_ids=hop_emb[snap][ei:ei+1],
                time_dis_ids=time_emb[snap][ei:ei+1],
            ).detach()

    def _fidelity(self, h_base, phi, true_score, D):
        top_k = min(self.top_k, D)
        mask  = torch.zeros(D).bool()
        for idx in np.argsort(np.abs(phi))[-top_k:]:
            mask[idx] = True
        h = h_base.clone()
        active = mask
        if active.sum() > 0:
            h[0, ~active, :] = h[0, active, :].mean(dim=0)
        else:
            h[0, :, :] = 0.0
        with torch.no_grad():
            out, _ = self.model.forward_from_embeddings(h)
            topk_score = torch.sigmoid(out).item()
        return float(1.0 - abs(true_score - topk_score))


class GradientExplainer:
    """
    Gradient-based baseline: L2 norm of the gradient of TADDY's output
    w.r.t. the dense embedding tensor (post EdgeEncoding).

    Captures first-order sensitivity of the anomaly score to each token's
    embedding vector.
    """

    def __init__(self, model, top_k: int = 5):
        self.model = model
        self.top_k = top_k
        self.model.eval()

    def explain_edges(
        self,
        int_emb, hop_emb, time_emb,
        edges: list,
    ) -> List[BaselineResult]:
        results = []
        D = int_emb[edges[0][1]].shape[1]

        for _, snap, ei in edges:
            ii = int_emb[snap][ei:ei+1]
            hh = hop_emb[snap][ei:ei+1]
            tt = time_emb[snap][ei:ei+1]

            # Get embedding with gradient tracking enabled
            emb_g = self.model.transformer.embeddings(
                init_pos_ids=ii,
                hop_dis_ids=hh,
                time_dis_ids=tt,
            ).detach().requires_grad_(True)

            # Forward through encoder and classifier
            enc, _ = self.model.transformer.encoder(emb_g)
            seq = sum(enc[:, i, :] for i in range(self.model.config.k + 1))
            seq = seq / float(self.model.config.k + 1)
            out = torch.sigmoid(self.model.cls_y(seq))
            out.sum().backward()

            # L2 norm of gradient per token position
            phi        = emb_g.grad[0].norm(dim=-1).detach().numpy()
            true_score = out.item()

            h_base   = emb_g.detach().unsqueeze(0) if emb_g.dim() == 2 else emb_g.detach()
            # h_base shape should be [1, D, d]
            h_base_clean = self.model.transformer.embeddings(
                init_pos_ids=ii,
                hop_dis_ids=hh,
                time_dis_ids=tt,
            ).detach()

            fidelity = self._fidelity(h_base_clean, phi, true_score, D)
            results.append(BaselineResult(
                method='Gradient',
                fidelity=fidelity,
                sparsity=1.0 - self.top_k / D,
                phi=phi,
            ))
        return results

    def _fidelity(self, h_base, phi, true_score, D):
        top_k = min(self.top_k, D)
        mask  = torch.zeros(D).bool()
        for idx in np.argsort(np.abs(phi))[-top_k:]:
            mask[idx] = True
        h = h_base.clone()
        active = mask
        if active.sum() > 0:
            h[0, ~active, :] = h[0, active, :].mean(dim=0)
        else:
            h[0, :, :] = 0.0
        with torch.no_grad():
            out, _ = self.model.forward_from_embeddings(h)
            topk_score = torch.sigmoid(out).item()
        return float(1.0 - abs(true_score - topk_score))
