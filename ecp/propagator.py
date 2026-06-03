"""
propagator.py
-------------
Full confidence propagation engine for Evidence Confidence Propagation (ECP).

The default calculation preserves the documented model:

    confidence(v) = clip(
        base_score(v) × ladder_propagation(v)
        - residue_penalty(v)
        + laurent_correction(v)
    )

Unlike the earlier implementation, root nodes are handled with a vacuum
identity in the ladder engine, so direct evidence at the chain boundary is not
accidentally annihilated by a⁻|0⟩ = 0.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .ladder import LadderOperator, LadderResult
from .residue import ResidueEngine, ResidueResult
from .schema import ChainConfig, ConfidenceNode, PropagationEdge


@dataclass(slots=True)
class NodeConfidenceResult:
    """Full confidence breakdown for one node."""

    node_id: str
    base_score: float
    ladder_result: LadderResult
    residue_result: ResidueResult
    confidence: float
    raw_confidence: float

    @property
    def is_singular(self) -> bool:
        return self.residue_result.pole_order > 0

    @property
    def is_double_pole(self) -> bool:
        return self.residue_result.pole_order == 2


@dataclass(slots=True)
class PropagationResult:
    """Propagation result for a full causal chain or DAG."""

    node_results: dict[str, NodeConfidenceResult] = field(default_factory=dict)
    config: Optional[ChainConfig] = None

    def confidence(self, node_id: str) -> float:
        if node_id not in self.node_results:
            raise KeyError(f"Node {node_id!r} not found in propagation result.")
        return self.node_results[node_id].confidence

    def singular_nodes(self) -> list[str]:
        return [node_id for node_id, r in self.node_results.items() if r.is_singular]

    def double_pole_nodes(self) -> list[str]:
        return [node_id for node_id, r in self.node_results.items() if r.is_double_pole]

    def summary(self) -> dict[str, float]:
        return {node_id: r.confidence for node_id, r in self.node_results.items()}

    def breakdown(self, node_id: str) -> dict[str, float | int]:
        """Return a numeric diagnostic breakdown for one node."""
        if node_id not in self.node_results:
            raise KeyError(f"Node {node_id!r} not found in propagation result.")
        r = self.node_results[node_id]
        return {
            "base_score": r.base_score,
            "ladder_propagation": r.ladder_result.propagated_confidence,
            "residue_penalty": r.residue_result.penalty,
            "laurent_correction": r.residue_result.correction,
            "raw_confidence": r.raw_confidence,
            "confidence": r.confidence,
            "pole_order": r.residue_result.pole_order,
            "path_count": r.ladder_result.path_count,
        }


class ConfidencePropagator:
    """Propagate evidence confidence through a causal DAG."""

    def __init__(self, config: Optional[ChainConfig] = None):
        self.config = config or ChainConfig()
        self._ladder = LadderOperator(self.config)
        self._residue = ResidueEngine(self.config)

    def propagate(
        self,
        nodes: list[ConfidenceNode],
        edges: list[PropagationEdge] | None = None,
    ) -> PropagationResult:
        """Run confidence propagation over nodes and optional DAG edges."""
        if not nodes:
            return PropagationResult(config=self.config)

        edges = edges or []
        self._validate_unique_nodes(nodes)
        self._validate_edges(nodes, edges)

        node_map = {node.id: node for node in nodes}
        incoming = self._incoming_edges(nodes, edges)
        order = self._topological_sort(nodes, edges)
        chain_length = self._effective_chain_length(nodes)

        result = PropagationResult(config=self.config)
        computed: dict[str, float] = {}

        for node_id in order:
            node = node_map[node_id]
            upstream = []
            for edge in incoming[node_id]:
                if edge.source_id not in computed:
                    continue
                source_node = node_map[edge.source_id]
                upstream.append(
                    self._ladder.transition_contribution(
                        source_node=source_node,
                        target_node=node,
                        source_confidence=computed[edge.source_id],
                        edge_weight=edge.weight,
                        chain_length=chain_length,
                    )
                )

            ladder_result = self._ladder.evaluate(
                node,
                chain_length=chain_length,
                upstream_contributions=upstream if upstream else None,
            )
            residue_result = self._residue.compute(node, chain_length=chain_length)
            raw = self._combine(node.base_score, ladder_result, residue_result)
            confidence = self.config.clip(raw)

            computed[node_id] = confidence
            result.node_results[node_id] = NodeConfidenceResult(
                node_id=node_id,
                base_score=node.base_score,
                ladder_result=ladder_result,
                residue_result=residue_result,
                raw_confidence=raw,
                confidence=confidence,
            )

        return result

    def _combine(
        self,
        base_score: float,
        ladder_result: LadderResult,
        residue_result: ResidueResult,
    ) -> float:
        if self.config.combine_mode == "multiplicative":
            structural = base_score * ladder_result.propagated_confidence
        else:
            total = self.config.direct_weight + self.config.upstream_weight
            structural = (
                self.config.direct_weight * base_score
                + self.config.upstream_weight * ladder_result.propagated_confidence
            ) / total
        return structural - residue_result.penalty + residue_result.correction

    def _effective_chain_length(self, nodes: list[ConfidenceNode]) -> int:
        max_position = max((node.causal_position or 0) for node in nodes)
        return max(len(nodes), max_position + 1, 1)

    def _incoming_edges(
        self,
        nodes: list[ConfidenceNode],
        edges: list[PropagationEdge],
    ) -> dict[str, list[PropagationEdge]]:
        incoming: dict[str, list[PropagationEdge]] = {node.id: [] for node in nodes}
        for edge in edges:
            incoming[edge.target_id].append(edge)
        return incoming

    def _validate_unique_nodes(self, nodes: list[ConfidenceNode]) -> None:
        ids = [node.id for node in nodes]
        duplicates = sorted({node_id for node_id in ids if ids.count(node_id) > 1})
        if duplicates:
            raise ValueError(f"Duplicate node ids are not allowed: {duplicates}")

    def _validate_edges(self, nodes: list[ConfidenceNode], edges: list[PropagationEdge]) -> None:
        node_ids = {node.id for node in nodes}
        for edge in edges:
            if edge.source_id not in node_ids:
                raise KeyError(f"Edge source {edge.source_id!r} is not present in nodes.")
            if edge.target_id not in node_ids:
                raise KeyError(f"Edge target {edge.target_id!r} is not present in nodes.")

    def _topological_sort(
        self,
        nodes: list[ConfidenceNode],
        edges: list[PropagationEdge],
    ) -> list[str]:
        if not edges:
            return [
                node.id
                for node in sorted(
                    nodes,
                    key=lambda n: (n.causal_position if n.causal_position is not None else 0, n.id),
                )
            ]

        node_ids = {node.id for node in nodes}
        in_degree = {node.id: 0 for node in nodes}
        adjacency = {node.id: [] for node in nodes}

        for edge in edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                continue
            adjacency[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

        queue = deque(sorted(node_id for node_id, degree in in_degree.items() if degree == 0))
        order: list[str] = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for neighbor in sorted(adjacency[node_id]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(nodes):
            raise ValueError("Cycle detected in causal chain. Confidence propagation requires a DAG.")
        return order
