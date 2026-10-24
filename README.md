# Explainable TADDY: Dual Spatial-Temporal Shapley Attribution

Code for the IEEE SNAMS 2026 paper:

> **Dual Spatial-Temporal Shapley Attribution for Explainable Anomaly Detection in Dynamic Social Graphs**
> Iyad Assaad Nekka, Hamida Seba, Walid Khaled Hidouci, Karima Amrouche
> IEEE SNAMS 2026, Barcelona, Spain, 17–20 November 2026.

**Affiliations:** LCSI Research Laboratory, Higher National School of Computer Science (ESI), Algiers, Algeria · Université Claude Bernard Lyon 1, Lyon, France

---

## What it does

[TADDY](https://arxiv.org/abs/2106.09876) is a strong Transformer-based detector of anomalous edges in dynamic graphs, but it gives no reason for an alarm. This framework wraps a **trained, frozen** TADDY model and returns two attribution vectors for each flagged edge:

- **Spatial** — which role drove the score: the two users (*u*, *v*) or a neighbour of PageRank rank *r*.
- **Temporal** — which snapshot carried the signal.

Detection is unchanged (ΔAUC = 0). Explanations are Shapley values estimated with a GraphSVX-style kernel-weighted Ridge surrogate.

**Key design: embedding-space masking.** Index 0 in TADDY's intimacy-rank table is the *highest-ranked* neighbour, not a neutral token, so zeroing indices is misleading. We replace masked tokens with the mean embedding of the active ones and feed the result directly to the Transformer.

---

## Results (from the paper)

Controlled comparison: all methods explain the same top-50 true positives per dataset on the same trained models (k = 5, D = 14). Mean of the three datasets (UCI-Message, Bitcoin-Alpha, Bitcoin-OTC).

| Method | Fidelity ↑ | Sufficiency ↑ | Comprehensiveness ↑ |
|---|---|---|---|
| Random | 0.9555 | 0.9403 | 0.0149 |
| Attention | 0.9647 | 0.9460 | 0.0518 |
| Gradient | 0.9703 | **0.9873** | 0.0507 |
| Gradient × Input | 0.9556 | 0.9370 | 0.0381 |
| **Ours** | **0.9763** | 0.9847 | **0.0626** |
| *Ablation:* index-space masking | 0.9663 | 0.9444 | 0.0370 |
| *Ablation:* random sampler | 0.9735 | 0.9862 | 0.0617 |
| *Ablation:* OLS (no kernel weights) | 0.9744 | 0.9887 | 0.0665 |

- Highest Comprehensiveness on **every** dataset; paired Wilcoxon p < 1e-10 against every baseline.
- Highest Fidelity on UCI-Message (0.934 vs. 0.916), where Fidelity is not saturated. On the Bitcoin datasets every method reaches ≈ 0.997.
- Embedding-space masking is the essential component (index-space masking cuts Comprehensiveness by 41%).
- One explanation costs about 0.18 s on a CPU core.

Detection AUC (TADDY = ours, identical by construction): UCI 0.8284 · Bitcoin-Alpha 0.9554 · Bitcoin-OTC 0.9596.

---

## Repository structure

```
Explainable-Taddy/
├── xtaddy/
│   ├── explainer.py      # Shapley explainer (embedding-space masking, Smarter sampler, Ridge surrogate)
│   ├── baselines.py      # Random, Attention, Gradient baselines
│   ├── metrics.py        # summaries and tables
│   └── visualise.py      # attribution plots
├── examples/
│   └── run_evaluation.py               # trains TADDY, runs the explainers (main paper tables)
├── experiments/
│   └── snams2026_baselines_ablation.py # controlled comparison, ablation, Wilcoxon tests, cost, worked example
├── paper/
│   ├── SNAMS2026_paper.pdf
│   └── SNAMS2026_paper.tex
├── requirements.txt
└── LICENSE
```

---

## Reproduce (Google Colab)

Main evaluation (~90 min):
```bash
git clone https://github.com/Iyad-Assaad-Nekka/Explainable-Taddy.git
cp -r Explainable-Taddy/xtaddy /content/xtaddy
python Explainable-Taddy/examples/run_evaluation.py
```

Baselines, ablation, statistical tests, cost and worked example (~1.5–2 h, CPU is enough): paste `experiments/snams2026_baselines_ablation.py` into one Colab cell and run it. It clones TADDY and this repository, trains TADDY with seed 1 and prints all tables. Results are saved to `/content/snams_cr/results.json`.

---

## Usage

```python
from xtaddy import XTADDYExplainer

explainer = XTADDYExplainer(model, num_samples=150, top_k=5)   # model = trained TADDY (DynADModel)
expl = explainer.explain_edge(edge_idx=0, snap=7,
                              int_emb=int_emb, hop_emb=hop_emb, time_emb=time_emb)
print(expl.true_score, expl.fidelity)
print(expl.phi_spatial)    # [u, v, n1..nk]
print(expl.phi_temporal)   # one value per lookback step
```

---

## Citation

```bibtex
@inproceedings{nekka2026dual,
  title     = {Dual Spatial-Temporal Shapley Attribution for Explainable Anomaly Detection in Dynamic Social Graphs},
  author    = {Nekka, Iyad Assaad and Seba, Hamida and Hidouci, Walid Khaled and Amrouche, Karima},
  booktitle = {Proc. 13th International Conference on Social Networks Analysis, Management and Security (SNAMS)},
  year      = {2026}
}
```

## License
MIT
