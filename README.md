# Explainable TADDY: X-TADDY
Implementation of X-TADDY, a Post-Hoc Explainability Method for Transformer Based Anomaly Detection in Dynamic Graphs
> **Post-hoc Shapley explainability framework for Transformer-based dynamic graph anomaly detection.**

**Authors:** Iyad Assaad Nekka · Hamida Seba · Khaled-Walid Hidouci  
**Affiliations:** Higher National School of Computer Science (ESI), Algiers, Algeria · University Claude Bernard Lyon 1, Lyon, France

---

## What is X-TADDY?

[TADDY](https://arxiv.org/abs/2106.09876) is a strong Transformer-based dynamic graph anomaly detector — but like most deep learning methods, it is a black box: it flags an anomalous edge and tells you nothing about *why*.

**X-TADDY** wraps a trained TADDY model with a post-hoc Shapley explanation layer. For each flagged edge, it produces two attribution vectors:

- **Spatial attribution** — which neighbor nodes drove the anomaly score
- **Temporal attribution** — which historical snapshot carried the most signal

X-TADDY leaves TADDY's weights, training, and inference completely unchanged. Detection AUC is preserved exactly.

---

## Key Design: Embedding-Space Masking

The critical technical contribution is the **graph-aware neutral baseline**. Naive index-space zeroing is wrong for TADDY: index 0 in its intimacy-rank table maps to the *highest-ranked* neighbor — the opposite of neutral. X-TADDY instead operates in TADDY's dense embedding space. For a coalition S of active tokens, excluded tokens are replaced by the mean embedding of active ones, and the modified sequence feeds directly into TADDY's Transformer — bypassing EdgeEncoding entirely.

---

## Results

| Method | Fidelity ↑ | Sparsity |
|---|---|---|
| TADDY (no explainer) | --- | --- |
| Random | 0.9357 ± 0.0878 | 0.6429 |
| Attention | 0.9530 ± 0.0618 | 0.6429 |
| Gradient | 0.9551 ± 0.0598 | 0.6429 |
| **X-TADDY (ours)** | **0.9559 ± 0.0580** | **0.6429** |

*Averaged across UCI-Message, Bitcoin-Alpha, Bitcoin-OTC. Top-50 true-positive edges per dataset, top-k=5, D=14 tokens.*

Detection AUC (TADDY = X-TADDY, identical by construction):

| Dataset | AUC |
|---|---|
| UCI-Message | 0.8284 |
| Bitcoin-Alpha | 0.9554 |
| Bitcoin-OTC | 0.9596 |

---

## Package Structure

```
x-taddy/
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── xtaddy/
│   ├── __init__.py         # Public API
│   ├── explainer.py        # XTADDYExplainer — core Shapley method
│   ├── baselines.py        # Random, Attention, Gradient baselines
│   ├── metrics.py          # Fidelity, Sparsity, reporting utilities
│   └── visualise.py        # Shapley bar chart plotting (Fig. 1)
└── examples/
    └── run_evaluation.py   # Complete Colab evaluation script
```

---

## Quick Start (Google Colab, T4 GPU)

Upload the `xtaddy/` folder to `/content/xtaddy/`, then:

```bash
python examples/run_evaluation.py
```

This handles everything: clones TADDY, patches for Python 3.12 compatibility, prepares data, trains TADDY, runs all 4 explainers, and prints the final comparison tables. Runtime ~90 min on T4.

---

## Standalone Usage

```python
from xtaddy import XTADDYExplainer, plot_explanation

# Wrap a trained DynADModel
explainer = XTADDYExplainer(model, num_samples=150, top_k=5)

# Explain a single anomalous edge
expl = explainer.explain_edge(
    edge_idx=0, snap=7,
    int_emb=int_emb, hop_emb=hop_emb, time_emb=time_emb,
)

print(f"True score : {expl.true_score:.4f}")
print(f"Fidelity   : {expl.fidelity:.4f}")
print(f"Spatial φ  : {expl.phi_spatial}")    # shape [k+2]
print(f"Temporal φ : {expl.phi_temporal}")   # shape [W]

# Plot Figure 1
plot_explanation(expl, k=5, W=2, save_path='fig_xtaddy.pdf')

# Explain top-50 true-positive edges
results = explainer.explain_top_edges(
    loaded_data, int_emb, hop_emb, time_emb, preds, y_test, n=50
)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Upload to GitHub

1. Create a new repository on GitHub named **x-taddy**
2. On the repo page click **"uploading an existing file"**
3. Drag and drop all files from this zip (unzipped)
4. Click **"Commit changes"**

---

## Citation

```bibtex
@article{nekka2025xtaddy,
  title  = {Explainability in Dynamic Graph Anomaly Detection: X-TADDY},
  author = {Nekka, Iyad Assaad and Seba, Hamida and Hidouci, Khaled-Walid},
  year   = {2025},
  note   = {ESI Algiers / University Claude Bernard Lyon 1}
}
```

---

## References

- Liu et al., "Anomaly Detection in Dynamic Graphs via Transformer", *IEEE TKDE*, 2022. [[paper]](https://arxiv.org/abs/2106.09876) [[code]](https://github.com/yuetan031/TADDY_pytorch)
- Duval & Malliaros, "GraphSVX: Shapley Value Explanations for Graph Neural Networks", *ECML-PKDD*, 2021.
- Shapley, "A value for n-person games", 1953.
