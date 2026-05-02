"""
X-TADDY: Evaluation Metrics and Reporting Utilities
====================================================
Functions for computing, aggregating, and printing Fidelity and Sparsity
results across multiple datasets and methods.
"""

import numpy as np
from typing import List, Dict
from .explainer import EdgeExplanation
from .baselines import BaselineResult


def summarise_explanations(results: List[EdgeExplanation]) -> Dict:
    """
    Aggregate Fidelity and Sparsity across a list of EdgeExplanation objects.

    Parameters
    ----------
    results : list of EdgeExplanation

    Returns
    -------
    dict with keys:
        fidelity_mean, fidelity_std, sparsity_mean, sparsity_std,
        n_edges, phi_spatial_mean, phi_temporal_mean
    """
    fids   = [r.fidelity  for r in results]
    spars  = [r.sparsity  for r in results]
    sp_mat = np.stack([r.phi_spatial  for r in results], axis=0)
    tp_mat = np.stack([r.phi_temporal for r in results], axis=0)

    return {
        'fidelity_mean':       float(np.mean(fids)),
        'fidelity_std':        float(np.std(fids)),
        'sparsity_mean':       float(np.mean(spars)),
        'sparsity_std':        float(np.std(spars)),
        'n_edges':             len(results),
        'phi_spatial_mean':    sp_mat.mean(axis=0).tolist(),
        'phi_temporal_mean':   tp_mat.mean(axis=0).tolist(),
    }


def summarise_baseline(results: List[BaselineResult]) -> Dict:
    """
    Aggregate Fidelity and Sparsity across a list of BaselineResult objects.
    """
    fids  = [r.fidelity for r in results]
    spars = [r.sparsity for r in results]
    return {
        'method':        results[0].method if results else 'unknown',
        'fidelity_mean': float(np.mean(fids)),
        'fidelity_std':  float(np.std(fids)),
        'sparsity_mean': float(np.mean(spars)),
        'sparsity_std':  float(np.std(spars)),
        'n_edges':       len(results),
    }


def print_comparison_table(
    dataset_results: Dict[str, Dict],
    title: str = "Explainability Evaluation",
):
    """
    Print a formatted comparison table of all methods across datasets.

    Parameters
    ----------
    dataset_results : dict
        Keys: dataset names.
        Values: dict with keys 'random', 'attention', 'gradient', 'xtaddy',
                each containing a summarise_* output dict.
    title : str
        Table title.
    """
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")

    methods = ['random', 'attention', 'gradient', 'xtaddy']
    labels  = ['Random', 'Attention', 'Gradient', 'X-TADDY (ours)']

    # Per-dataset tables
    for dataset, results in dataset_results.items():
        print(f"\n  Dataset: {dataset.upper()}")
        print(f"  {'─'*55}")
        print(f"  {'Method':17s}  {'Fidelity':>18}  {'Sparsity':>10}")
        print(f"  {'─'*55}")
        for m, label in zip(methods, labels):
            if m not in results:
                continue
            r = results[m]
            mark = ' ◄' if m == 'xtaddy' else ''
            print(f"  {label:17s}  "
                  f"{r['fidelity_mean']:>7.4f} ± {r['fidelity_std']:.4f}  "
                  f"{r['sparsity_mean']:>7.4f}{mark}")
        print(f"  {'─'*55}")

    # Averaged across all datasets
    print(f"\n  {'═'*55}")
    print(f"  AVERAGED ACROSS ALL DATASETS")
    print(f"  {'─'*55}")
    print(f"  {'Method':17s}  {'Fidelity':>18}  {'Sparsity':>10}")
    print(f"  {'─'*55}")

    for m, label in zip(methods, labels):
        all_fids  = []
        all_spars = []
        for results in dataset_results.values():
            if m in results:
                all_fids.append(results[m]['fidelity_mean'])
                all_spars.append(results[m]['sparsity_mean'])
        if not all_fids:
            continue
        mark = ' ◄' if m == 'xtaddy' else ''
        print(f"  {label:17s}  "
              f"{np.mean(all_fids):>7.4f} ± {np.std(all_fids):.4f}  "
              f"{np.mean(all_spars):>7.4f}{mark}")

    print(f"  {'═'*55}")
    print(f"  Sparsity = 1 - top_k/D, fixed by construction for all methods.")
    print(f"  Decisive metric: Fidelity ↑")
