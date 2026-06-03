"""
residue.py
----------
Laurent coefficient extraction for singular confidence points.

For a node v, the local meromorphic confidence model is:

    C_v(z) = A_v(z) / (z - z0)^m

where m is the pole order and A_v(z) is analytic around z0. This gives exact
local Laurent coefficients:

    simple pole: a_-1 = A_v(z0)
    double pole: a_-2 = A_v(z0), a_-1 = A'_v(z0)

The engine maps these coefficients into bounded penalty/correction terms used
by the propagator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schema import ChainConfig, ConfidenceNode


@dataclass(slots=True)
class ResidueResult:
    """Residue and Laurent correction data for one node."""

    node_id: str
    pole_order: int
    residue: float
    laurent_a2: float
    penalty: float
    correction: float
    pole_location: float = 0.0
    analytic_value: float = 0.0
    analytic_derivative: float = 0.0
    pole_sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def net_adjustment(self) -> float:
        """Return penalty - correction, the net subtraction from confidence."""
        return self.penalty - self.correction


class ResidueEngine:
    """Compute Laurent coefficients and confidence penalties at singular nodes."""

    def __init__(self, config: ChainConfig):
        self.config = config

    def compute(self, node: ConfidenceNode, chain_length: int) -> ResidueResult:
        if chain_length < 1:
            raise ValueError("chain_length must be >= 1.")

        pole_sources = self._pole_sources(node)
        pole_order = len(pole_sources)
        z0 = self._normalized_position(node, chain_length)

        if pole_order == 0:
            return ResidueResult(
                node_id=node.id,
                pole_order=0,
                residue=0.0,
                laurent_a2=0.0,
                penalty=0.0,
                correction=0.0,
                pole_location=z0,
                pole_sources=(),
            )

        analytic_value = self._analytic_part(node, z0)
        analytic_derivative = self._analytic_derivative(node, z0)

        if pole_order == 1:
            residue = analytic_value
            laurent_a2 = 0.0
        else:
            laurent_a2 = analytic_value
            residue = analytic_derivative

        penalty = self._bounded(abs(residue)) * self.config.residue_penalty_scale
        correction = 0.0
        if pole_order == 2:
            correction = self._bounded(abs(laurent_a2)) * self.config.laurent_correction_scale

        return ResidueResult(
            node_id=node.id,
            pole_order=pole_order,
            residue=residue,
            laurent_a2=laurent_a2,
            penalty=penalty,
            correction=correction,
            pole_location=z0,
            analytic_value=analytic_value,
            analytic_derivative=analytic_derivative,
            pole_sources=tuple(pole_sources),
        )

    def local_laurent_coefficients(
        self, node: ConfidenceNode, chain_length: int
    ) -> dict[str, float]:
        """Return explicit Laurent coefficients for external inspection."""
        result = self.compute(node, chain_length)
        return {
            "a_-2": result.laurent_a2,
            "a_-1": result.residue,
            "z0": result.pole_location,
            "pole_order": float(result.pole_order),
        }

    def _pole_sources(self, node: ConfidenceNode) -> list[str]:
        sources: list[str] = []
        if node.has_position_pole:
            sources.append("position_boundary")
        if node.has_evidence_pole:
            sources.append("zero_evidence")
        return sources

    def _normalized_position(self, node: ConfidenceNode, chain_length: int) -> float:
        if node.causal_position is None:
            return 0.0
        return max(0.0, min(1.0, node.causal_position / max(chain_length - 1, 1)))

    def _evidence_anchor(self, node: ConfidenceNode) -> float:
        """Analytic amplitude floor used when evidence is absent."""
        return node.base_score if node.base_score > 0.0 else self.config.epsilon

    def _analytic_part(self, node: ConfidenceNode, z: float) -> float:
        """
        Analytic amplitude A_v(z).

        exp(-z) encodes the documented boundary sensitivity while keeping the
        local function analytic around z0.
        """
        return self._evidence_anchor(node) * math.exp(-z)

    def _analytic_derivative(self, node: ConfidenceNode, z: float) -> float:
        """Derivative A'_v(z) for A_v(z)=anchor×exp(-z)."""
        return -self._analytic_part(node, z)

    def _bounded(self, value: float) -> float:
        """Map a nonnegative coefficient to [0, 1) without hard capping."""
        if value < 0.0:
            raise ValueError("value must be nonnegative.")
        return value / (1.0 + value)
