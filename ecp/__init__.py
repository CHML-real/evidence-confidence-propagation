"""
Evidence Confidence Propagation (ECP)
--------------------------------------
A Python package for computing evidence-weighted confidence scores
along causal chains and DAG structures.

Confidence is computed as:

    confidence(v) = base_score(v)
                  × ladder_propagation(v)
                  - residue_penalty(v)
                  + laurent_correction(v)

Singular points (chain boundaries, zero evidence) are handled via
residue theory and Laurent series corrections. Confidence propagates
through causal chains using ladder operators inspired by quantum
harmonic oscillator raising/lowering operators.

Contributors
------------
lajjadred  https://github.com/lajjadred   project lead
이채문      https://github.com/CHML-real   mathematical algorithm development
CUBE       https://github.com/90cube      idea proposal and data collection
"""

from .schema import (
    EvidenceItem,
    ConfidenceNode,
    PropagationEdge,
    ChainConfig,
    SourceType,
    source_weight,
)
from .residue import ResidueEngine, ResidueResult
from .ladder import LadderOperator, LadderResult
from .propagator import ConfidencePropagator, PropagationResult, NodeConfidenceResult

__version__ = "0.1.0"
__all__ = [
    "EvidenceItem",
    "ConfidenceNode",
    "PropagationEdge",
    "ChainConfig",
    "SourceType",
    "source_weight",
    "ResidueEngine",
    "ResidueResult",
    "LadderOperator",
    "LadderResult",
    "ConfidencePropagator",
    "PropagationResult",
    "NodeConfidenceResult",
]
