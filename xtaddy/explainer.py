"""
X-TADDY: Post-hoc Shapley Explainability for Dynamic Graph Anomaly Detection
=============================================================================
Authors: Iyad Assaad Nekka, Hamida Seba, Khaled-Walid Hidouci
ESI Algiers / University Claude Bernard Lyon 1

This module implements the XTADDYExplainer class, which wraps a trained
DynADModel (TADDY) and produces per-edge Shapley attributions decomposed
into spatial (neighbor) and temporal (lookback step) components.

Key design choice — embedding-space masking
-------------------------------------------
Excluded tokens are replaced by the mean embedding of active tokens,
computed AFTER EdgeEncoding (dense vector space). This avoids the
pathological behaviour of index-space zeroing, where index 0 in TADDY's
intimacy-rank table maps to the highest-ranked neighbour — the opposite
of a neutral baseline.

References
----------
Liu et al., "Anomaly Detection in Dynamic Graphs via Transformer", TKDE 2022.
Duval & Malliaros, "GraphSVX: Shapley Value Explanations for GNNs", ECML 2021.
Shapley, "A value for n-person games", 1953.
"""

import torch
import numpy as np
import scipy.special
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EdgeExplanation:
    """
    Holds the full X-TADDY output for one explained edge.

    Attributes
    ----------
    true_score : float
        TADDY's original anomaly score for this edge (0–1).
    phi : np.ndarray, shape [D]
        Per-token Shapley values (WLS surrogate coefficients).
    phi_spatial : np.ndarray, shape [k+2]
        Shapley values aggregated by neighbor node position.
        phi_spatial[0] = node u, phi_spatial[1] = node v,
        phi_spatial[2..k+1] = neighbors n1..nk.
    phi_temporal : np.ndarray, shape [W]
        Shapley values aggregated by lookback step.
        phi_temporal[0] = current snapshot (l=0),
        phi_temporal[1] = one step back (l=1), etc.
    base_value : float
        Surrogate intercept — expected TADDY score with no tokens active.
    fidelity : float
        1 - |true_score - score_with_top_k_tokens|. Measures how causally
        faithful the top-k explanation is.
    sparsity : float
        1 - top_k / D. Fixed by construction for a given (k, D) pair.
    top_k_indices : List[int]
        Token positions selected by the explanation (largest |phi_j|).
    """
    true_score:    float
    phi:           np.ndarray
    phi_spatial:   np.ndarray
    phi_temporal:  np.ndarray
    base_value:    float
    fidelity:      float
    sparsity:      float
    top_k_indices: List[int]


# ---------------------------------------------------------------------------
# Core explainer
# ---------------------------------------------------------------------------

class XTADDYExplainer:
    """
    Post-hoc Shapley explainer for a trained TADDY (DynADModel) instance.

    The explainer is constructed once and can be called repeatedly on
    different edges without any retraining or modification of TADDY.

    Parameters
    ----------
    model : DynADModel
        A fully trained TADDY model (in eval mode).
    num_samples : int
        Number of coalition samples per explained edge (default 150).
    top_k : int
        Number of tokens retained for Fidelity measurement (default 5).
    ridge_alpha : float
        L2 regularisation for the WLS Ridge surrogate (default 0.01).
    random_state : int
        Seed for reproducible coalition sampling (default 42).
    """

    def __init__(
        self,
        model,
        num_samples: int = 150,
        top_k:       int = 5,
        ridge_alpha: float = 0.01,
        random_state: int  = 42,
    ):
        self.model        = model
        self.num_samples  = num_samples
        self.top_k        = top_k
        self.ridge_alpha  = ridge_alpha
        self.rng          = np.random.default_rng(random_state)

        self.model.eval()

        self.k = model.config.k
        self.W = model.config.window_size
        # Total number of tokens per edge: 2 endpoint nodes + k neighbours,
        # repeated for each of W lookback steps.
        self.D = (self.k + 2) * self.W

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def explain_edge(
        self,
        edge_idx:  int,
        snap:      int,
        int_emb:   list,
        hop_emb:   list,
        time_emb:  list,
    ) -> EdgeExplanation:
        """
        Compute the X-TADDY explanation for a single edge.

        Parameters
        ----------
        edge_idx : int
            Row index of the target edge in the snapshot tensor.
        snap : int
            Snapshot index containing the edge.
        int_emb, hop_emb, time_emb : list of Tensors
            Precomputed embedding index tensors from model.generate_embedding().
            Each is a list indexed by snapshot; each element is a LongTensor
            of shape [N_edges_in_snap, D].

        Returns
        -------
        EdgeExplanation
        """
        # Step 1: compute TADDY's true score and the dense edge embedding
        true_score, h_base = self._get_score_and_embedding(
            int_emb, hop_emb, time_emb, snap, edge_idx
        )

        D = self.D

        # Step 2: sample coalitions using the Smarter strategy
        Z, weights = self._sample_coalitions(D)

        # Step 3: score each coalition via embedding-space masking
        fz = self._score_coalitions(h_base, Z, D)

        # Step 4: fit WLS Ridge surrogate
        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        reg.fit(
            Z.numpy(),
            fz.numpy(),
            sample_weight=np.clip(weights.numpy(), 0.0, 1000.0),
        )
        phi        = reg.coef_          # shape [D]
        base_value = float(reg.intercept_)

        # Step 5: aggregate into spatial and temporal vectors
        phi_spatial, phi_temporal = self._aggregate(phi)

        # Step 6: compute Fidelity and Sparsity
        top_k_indices = list(np.argsort(np.abs(phi))[-self.top_k:])
        fidelity      = self._compute_fidelity(h_base, phi, true_score, D)
        sparsity      = 1.0 - self.top_k / D

        return EdgeExplanation(
            true_score    = true_score,
            phi           = phi,
            phi_spatial   = phi_spatial,
            phi_temporal  = phi_temporal,
            base_value    = base_value,
            fidelity      = fidelity,
            sparsity      = sparsity,
            top_k_indices = top_k_indices,
        )

    def explain_top_edges(
        self,
        loaded_data: dict,
        int_emb:     list,
        hop_emb:     list,
        time_emb:    list,
        preds:       list,
        y_test:      list,
        n:           int = 50,
    ) -> List[EdgeExplanation]:
        """
        Explain the top-n most anomalous true-positive edges.

        Parameters
        ----------
        loaded_data : dict
            Dataset dict returned by DynamicDatasetLoader.load().
        int_emb, hop_emb, time_emb : list
            Precomputed embedding tensors from model.generate_embedding().
        preds : list
            Per-snapshot sigmoid prediction arrays (from eval loop).
        y_test : list
            Per-snapshot ground-truth label arrays.
        n : int
            Number of edges to explain (default 50).

        Returns
        -------
        List[EdgeExplanation]
        """
        candidates = []
        for si, snap in enumerate(loaded_data['snap_test']):
            p = preds[si]
            y = y_test[si]
            for ei in range(len(p)):
                if y[ei] == 1:
                    candidates.append((float(p[ei]), snap, ei))
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_edges = candidates[:n]

        print(f"Explaining {len(top_edges)} true-positive edges "
              f"(D={self.D}, top_k={self.top_k}) ...")

        results = []
        for rank, (score, snap, ei) in enumerate(top_edges):
            expl = self.explain_edge(ei, snap, int_emb, hop_emb, time_emb)
            results.append(expl)
            if (rank + 1) % 10 == 0:
                mean_fid = np.mean([r.fidelity for r in results])
                print(f"  {rank+1}/{len(top_edges)} explained | "
                      f"Mean Fidelity so far: {mean_fid:.4f}")

        return results

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_score_and_embedding(
        self, int_emb, hop_emb, time_emb, snap, edge_idx
    ):
        """
        Compute the true anomaly score and dense embedding for one edge.

        Returns
        -------
        true_score : float
        h_base : Tensor, shape [1, D, hidden_size]
            Dense embedding from EdgeEncoding (before Transformer).
        """
        ii = int_emb[snap][edge_idx:edge_idx+1]
        hh = hop_emb[snap][edge_idx:edge_idx+1]
        tt = time_emb[snap][edge_idx:edge_idx+1]

        with torch.no_grad():
            out, _ = self.model.forward(ii, hh, tt)
            true_score = torch.sigmoid(out).item()

            # Compute dense embedding: EdgeEncoding output, shape [1, D, d]
            h_base = self.model.transformer.embeddings(
                init_pos_ids=ii,
                hop_dis_ids=hh,
                time_dis_ids=tt,
            )

        return true_score, h_base.detach()

    def _sample_coalitions(self, D: int):
        """
        GraphSVX Smarter coalition sampler adapted to D temporal tokens.

        Always includes:
          - Full coalition (all ones)  — anchors surrogate upper bound
          - Empty coalition (all zeros) — anchors surrogate lower bound
          - 2D singleton perturbations — capture individual token effects

        Remaining budget filled with uniformly random coalitions.

        Returns
        -------
        Z : FloatTensor, shape [num_samples, D]
        weights : FloatTensor, shape [num_samples]
            Shapley kernel weights, clamped to 1000.
        """
        ns = self.num_samples
        Z  = torch.zeros(ns, D)

        # Full and empty coalitions
        Z[0] = torch.ones(D)
        Z[1] = torch.zeros(D)

        i = 2
        for j in range(D):
            # One token masked out
            if i >= ns:
                break
            v = torch.ones(D); v[j] = 0.0; Z[i] = v; i += 1
            # One token active only
            if i >= ns:
                break
            v = torch.zeros(D); v[j] = 1.0; Z[i] = v; i += 1

        if i < ns:
            rand = torch.from_numpy(
                self.rng.integers(0, 2, size=(ns - i, D)).astype(np.float32)
            )
            Z[i:] = rand

        # Shapley kernel weights
        s = Z.sum(dim=1).long()
        w = []
        for si in s:
            si = si.item()
            if si == 0 or si == D:
                w.append(1000.0)
            else:
                denom = scipy.special.comb(D, si) * si * (D - si)
                w.append(float((D - 1) / denom) if denom > 0 else 1.0)

        return Z, torch.tensor(w, dtype=torch.float32).clamp(max=1000.0)

    def _score_coalition(self, h_base: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Score one coalition by replacing inactive token embeddings with the
        mean embedding of active tokens, then running the Transformer.

        This is the core perturbation strategy: neutral baseline =
        mean(active token embeddings), operated in dense embedding space.

        Parameters
        ----------
        h_base : Tensor, shape [1, D, hidden_size]
        mask   : BoolTensor, shape [D]   — True = active

        Returns
        -------
        score : float   (sigmoid of TADDY classifier output)
        """
        h = h_base.clone()                # [1, D, d]
        active = mask.bool()

        if active.sum() > 0:
            # Mean embedding of active tokens: shape [d]
            h_mean = h[0, active, :].mean(dim=0)
        else:
            h_mean = torch.zeros(h.shape[-1])

        # Replace inactive token positions with neutral baseline
        h[0, ~active, :] = h_mean

        with torch.no_grad():
            out, _ = self.model.forward_from_embeddings(h)
            score  = torch.sigmoid(out).item()

        return score

    def _score_coalitions(
        self,
        h_base: torch.Tensor,
        Z:      torch.Tensor,
        D:      int,
    ) -> torch.Tensor:
        """
        Score all coalition masks in Z.

        Returns
        -------
        fz : FloatTensor, shape [num_samples]
        """
        fz = torch.zeros(Z.shape[0])
        for s_idx in range(Z.shape[0]):
            mask     = Z[s_idx].bool()
            fz[s_idx] = self._score_coalition(h_base, mask)
        return fz

    def _aggregate(self, phi: np.ndarray):
        """
        Aggregate per-token Shapley values into spatial and temporal vectors.

        Token layout (TADDY with k=5, W=2, D=14):
          l=0: tokens 0..6  (u, v, n1..n5)
          l=1: tokens 7..13 (u, v, n1..n5)

        phi_spatial[i] = sum of phi_j for all tokens j where position i
                         within a lookback step equals i.
        phi_temporal[l] = sum of phi_j for all tokens j in lookback step l.

        Returns
        -------
        phi_spatial  : np.ndarray, shape [k+2]
        phi_temporal : np.ndarray, shape [W]
        """
        tokens_per_step = self.k + 2
        phi_spatial  = np.zeros(tokens_per_step)
        phi_temporal = np.zeros(self.W)

        for j, p in enumerate(phi):
            l = j // tokens_per_step     # lookback step
            i = j  % tokens_per_step     # position within step
            phi_spatial[i]  += p
            if l < self.W:
                phi_temporal[l] += p

        return phi_spatial, phi_temporal

    def _compute_fidelity(
        self,
        h_base:     torch.Tensor,
        phi:        np.ndarray,
        true_score: float,
        D:          int,
    ) -> float:
        """
        Fidelity = 1 - |true_score - score_with_top_k_tokens_only|.

        The top-k mask uses the same embedding-space masking as coalition
        scoring, ensuring consistent measurement.
        """
        top_k        = min(self.top_k, D)
        top_indices  = set(np.argsort(np.abs(phi))[-top_k:])
        mask         = torch.zeros(D).bool()
        for idx in top_indices:
            mask[idx] = True

        topk_score = self._score_coalition(h_base, mask)
        return float(1.0 - abs(true_score - topk_score))
