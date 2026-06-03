"""
schema.py
---------
Core data schemas for Evidence Confidence Propagation (ECP).

This version treats the package's mathematical language as a real modeling
contract rather than as documentation-only metaphor. Confidence nodes expose
both evidence-derived base scores and pole classification metadata used by the
residue engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


SourceType = Literal["primary", "secondary", "tertiary", "unknown"]
CombineMode = Literal["multiplicative", "convex"]

_SOURCE_WEIGHT: dict[str, float] = {
    "primary": 1.0,
    "secondary": 0.6,
    "tertiary": 0.3,
    "unknown": 0.1,
}


def source_weight(source_type: SourceType) -> float:
    """Return the default trust weight for a source type."""
    return _SOURCE_WEIGHT.get(source_type, _SOURCE_WEIGHT["unknown"])


@dataclass(slots=True)
class EvidenceItem:
    """
    A single evidence item supporting a ConfidenceNode.

    The weighted contribution of an item is strength × source_weight.
    """

    key: str
    source_type: SourceType = "unknown"
    strength: float = 0.5
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("EvidenceItem.key cannot be empty.")
        if self.source_type not in _SOURCE_WEIGHT:
            raise ValueError(
                f"EvidenceItem.source_type must be one of {list(_SOURCE_WEIGHT)}."
            )
        if not (0.0 < self.strength <= 1.0):
            raise ValueError("EvidenceItem.strength must be in (0, 1].")
        self.key = self.key.strip()
        self.note = self.note.strip() if self.note else ""

    @property
    def weighted_strength(self) -> float:
        """Return evidence strength scaled by source trust."""
        return self.strength * source_weight(self.source_type)


@dataclass(slots=True)
class ConfidenceNode:
    """
    A node in a causal chain or DAG whose confidence is evaluated.

    Singular conditions are deliberately explicit:
    - position pole: causal_position is None or 0
    - evidence pole: evidence_count is 0
    - double pole: both conditions hold
    """

    id: str
    label: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    causal_position: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("ConfidenceNode.id cannot be empty.")
        if not self.label or not self.label.strip():
            raise ValueError("ConfidenceNode.label cannot be empty.")
        if self.causal_position is not None and self.causal_position < 0:
            raise ValueError("ConfidenceNode.causal_position must be >= 0.")
        self.id = self.id.strip()
        self.label = self.label.strip()

    @property
    def evidence_count(self) -> int:
        """Return the number of direct evidence items."""
        return len(self.evidence)

    @property
    def has_position_pole(self) -> bool:
        """Return True when the node is at the causal boundary."""
        return self.causal_position is None or self.causal_position == 0

    @property
    def has_evidence_pole(self) -> bool:
        """Return True when the node has no direct evidence support."""
        return self.evidence_count == 0

    @property
    def pole_order(self) -> int:
        """Return Laurent pole order: 0, 1, or 2."""
        return int(self.has_position_pole) + int(self.has_evidence_pole)

    @property
    def is_singular(self) -> bool:
        """Return True when at least one singular condition holds."""
        return self.pole_order > 0

    @property
    def is_double_pole(self) -> bool:
        """Return True when both singular conditions hold."""
        return self.pole_order == 2

    @property
    def base_score(self) -> float:
        """Return the direct evidence score as a weighted arithmetic mean."""
        if self.evidence_count == 0:
            return 0.0
        return sum(ev.weighted_strength for ev in self.evidence) / self.evidence_count


@dataclass(slots=True)
class PropagationEdge:
    """A directed causal edge along which confidence can propagate."""

    source_id: str
    target_id: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("PropagationEdge.source_id cannot be empty.")
        if not self.target_id or not self.target_id.strip():
            raise ValueError("PropagationEdge.target_id cannot be empty.")
        self.source_id = self.source_id.strip()
        self.target_id = self.target_id.strip()
        if self.source_id == self.target_id:
            raise ValueError("PropagationEdge cannot connect a node to itself.")
        if not (0.0 < self.weight <= 1.0):
            raise ValueError("PropagationEdge.weight must be in (0, 1].")


@dataclass(slots=True)
class ChainConfig:
    """
    Configuration for ECP propagation.

    combine_mode="multiplicative" preserves the documented core formula:
        confidence = base_score × ladder_propagation - penalty + correction

    combine_mode="convex" is available for applications where direct evidence
    and upstream propagation should be blended rather than multiplied.
    """

    epsilon: float = 0.01
    residue_penalty_scale: float = 1.0
    laurent_correction_scale: float = 0.5
    damping: float = 1.0
    combine_mode: CombineMode = "multiplicative"
    direct_weight: float = 0.65
    upstream_weight: float = 0.35

    def __post_init__(self) -> None:
        if not (0.0 < self.epsilon < 0.5):
            raise ValueError("ChainConfig.epsilon must be in (0, 0.5).")
        if self.residue_penalty_scale < 0.0:
            raise ValueError("ChainConfig.residue_penalty_scale must be >= 0.")
        if self.laurent_correction_scale < 0.0:
            raise ValueError("ChainConfig.laurent_correction_scale must be >= 0.")
        if not (0.0 < self.damping <= 1.0):
            raise ValueError("ChainConfig.damping must be in (0, 1].")
        if self.combine_mode not in {"multiplicative", "convex"}:
            raise ValueError("ChainConfig.combine_mode must be 'multiplicative' or 'convex'.")
        if self.direct_weight < 0.0 or self.upstream_weight < 0.0:
            raise ValueError("ChainConfig direct/upstream weights must be >= 0.")
        if self.combine_mode == "convex" and self.direct_weight + self.upstream_weight <= 0.0:
            raise ValueError("Convex combine mode requires positive total weight.")

    def clip(self, value: float) -> float:
        """Clip confidence to the configured closed interval [epsilon, 1]."""
        return max(self.epsilon, min(1.0, value))
