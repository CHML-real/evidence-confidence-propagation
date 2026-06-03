"""
ladder.py
---------
Ladder-operator transport for confidence propagation.

This implementation separates two concepts that were previously conflated:
1. local vacuum/self support of a root node, and
2. transition transport from an upstream node to a downstream node.

The raising operator controls outgoing propagation from a source state. The
end boundary a⁺|N⟩ = 0 therefore blocks propagation beyond the chain end, not
incoming propagation into the final node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import ChainConfig, ConfidenceNode


@dataclass(slots=True)
class LadderResult:
    """Result of ladder transport evaluation for one node."""

    node_id: str
    n_normalized: float
    raising_factor: float
    lowering_factor: float
    propagated_confidence: float
    upstream_component: float
    self_component: float
    path_count: int = 0


class LadderOperator:
    """Evaluate QHO-style raising/lowering factors and confidence transport."""

    def __init__(self, config: ChainConfig):
        self.config = config

    def evaluate(
        self,
        node: ConfidenceNode,
        chain_length: int,
        upstream_contributions: list[tuple[float, float]] | None = None,
    ) -> LadderResult:
        """
        Evaluate local ladder state and incoming transported confidence.

        upstream_contributions contains (transported_confidence, path_weight)
        pairs. Transport factors should already have been applied by
        transition_contribution().
        """
        if chain_length < 1:
            raise ValueError("chain_length must be >= 1.")

        n = self._get_position(node)
        n_max = chain_length - 1
        n_norm = self._normalize(n, chain_length)
        raising = self.raising_factor(n, n_max)
        lowering = self.lowering_factor(n)

        self_component = 1.0 if not upstream_contributions else node.base_score
        upstream_component, path_count = self._weighted_average(upstream_contributions or [])

        if upstream_contributions:
            propagated = upstream_component
        else:
            # Vacuum identity: a root node's own evidence is not destroyed by a⁻|0⟩ = 0.
            propagated = self_component

        propagated *= self.config.damping ** n_norm
        propagated = self.config.clip(propagated)

        return LadderResult(
            node_id=node.id,
            n_normalized=n_norm,
            raising_factor=raising,
            lowering_factor=lowering,
            propagated_confidence=propagated,
            upstream_component=upstream_component,
            self_component=self_component,
            path_count=path_count,
        )

    def transition_contribution(
        self,
        source_node: ConfidenceNode,
        target_node: ConfidenceNode,
        source_confidence: float,
        edge_weight: float,
        chain_length: int,
    ) -> tuple[float, float]:
        """
        Transport confidence from source to target through a raising operator.

        Returns a (transported_confidence, path_weight) pair suitable for
        evaluate(..., upstream_contributions=...).
        """
        if chain_length < 1:
            raise ValueError("chain_length must be >= 1.")
        if not (0.0 < edge_weight <= 1.0):
            raise ValueError("edge_weight must be in (0, 1].")

        source_n = self._get_position(source_node)
        target_n = self._get_position(target_node)
        n_max = chain_length - 1
        distance = max(1, target_n - source_n)
        distance_norm = distance / max(chain_length - 1, 1)

        transport = self.raising_factor(source_n, n_max) * (self.config.damping ** distance_norm)
        value = self.config.clip(source_confidence * transport * edge_weight)
        return value, edge_weight

    def raising_factor(self, n: int, n_max: int) -> float:
        """Return a⁺ scaling factor at position n with end-boundary zero."""
        if n < 0:
            raise ValueError("n must be >= 0.")
        if n_max < 0:
            raise ValueError("n_max must be >= 0.")
        if n >= n_max:
            return 0.0
        # Normalized QHO amplitude keeps the confidence transfer bounded.
        denominator = math.sqrt(max(n_max + 1, 1))
        return (math.sqrt(n + 1) / denominator) * self.config.damping

    def lowering_factor(self, n: int) -> float:
        """Return a⁻ scaling factor at position n with start-boundary zero."""
        if n < 0:
            raise ValueError("n must be >= 0.")
        if n == 0:
            return 0.0
        return math.sqrt(n) * self.config.damping

    def _weighted_average(self, contributions: list[tuple[float, float]]) -> tuple[float, int]:
        if not contributions:
            return 0.0, 0
        for confidence, weight in contributions:
            if confidence < 0.0:
                raise ValueError("upstream confidence cannot be negative.")
            if weight <= 0.0:
                raise ValueError("upstream path weight must be positive.")
        total_weight = sum(weight for _, weight in contributions)
        weighted = sum(confidence * weight for confidence, weight in contributions)
        return self.config.clip(weighted / total_weight), len(contributions)

    def _get_position(self, node: ConfidenceNode) -> int:
        return node.causal_position if node.causal_position is not None else 0

    def _normalize(self, n: int, chain_length: int) -> float:
        if chain_length <= 1:
            return 0.0
        return max(0.0, min(1.0, n / (chain_length - 1)))
