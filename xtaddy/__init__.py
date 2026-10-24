"""
X-TADDY: Post-hoc Shapley Explainability for Dynamic Graph Anomaly Detection
=============================================================================
Authors: Iyad Assaad Nekka, Hamida Seba, Khaled-Walid Hidouci
ESI Algiers / University Claude Bernard Lyon 1
"""

from .explainer  import XTADDYExplainer, EdgeExplanation
from .baselines  import RandomExplainer, AttentionExplainer, GradientExplainer
from .metrics    import summarise_explanations, summarise_baseline, print_comparison_table
from .visualise  import plot_explanation, plot_mean_attribution

__version__ = '1.0.0'
__all__     = [
    'XTADDYExplainer',
    'EdgeExplanation',
    'RandomExplainer',
    'AttentionExplainer',
    'GradientExplainer',
    'summarise_explanations',
    'summarise_baseline',
    'print_comparison_table',
    'plot_explanation',
    'plot_mean_attribution',
]
