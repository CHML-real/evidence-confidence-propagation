"""
basic_usage.py
--------------
End-to-end example of evidence-confidence-propagation.

Scenario
--------
Three events from a fictional lore timeline:
    Kain Incident → Rift Opening → Archive Fall

Each event has direct evidence of varying strength and source trust.
We propagate confidence through the causal chain and inspect results.

Run
---
    python examples/basic_usage.py
"""

from ecp import (
    EvidenceItem,
    ConfidenceNode,
    PropagationEdge,
    ChainConfig,
    ConfidencePropagator,
)


# ---------------------------------------------------------------------------
# 1. Configure
# ---------------------------------------------------------------------------

config = ChainConfig(
    epsilon=0.01,
    residue_penalty_scale=1.0,
    laurent_correction_scale=0.5,
    damping=0.9,
    combine_mode="multiplicative",
)


# ---------------------------------------------------------------------------
# 2. Define nodes
# ---------------------------------------------------------------------------

nodes = [
    ConfidenceNode(
        id="kain_incident",
        label="Kain Incident",
        causal_position=0,
        evidence=[
            EvidenceItem(key="official_001", source_type="primary",   strength=0.9),
            EvidenceItem(key="fan_002",      source_type="tertiary",  strength=0.4),
        ],
    ),
    ConfidenceNode(
        id="rift_opening",
        label="Rift Opening",
        causal_position=1,
        evidence=[
            EvidenceItem(key="official_003", source_type="primary",   strength=0.8),
            EvidenceItem(key="secondary_004",source_type="secondary", strength=0.6),
        ],
    ),
    ConfidenceNode(
        id="archive_fall",
        label="Archive Fall",
        causal_position=2,
        evidence=[
            EvidenceItem(key="secondary_005", source_type="secondary", strength=0.6),
        ],
    ),
]


# ---------------------------------------------------------------------------
# 3. Define causal edges
# ---------------------------------------------------------------------------

edges = [
    PropagationEdge(source_id="kain_incident", target_id="rift_opening",  weight=1.0),
    PropagationEdge(source_id="rift_opening",  target_id="archive_fall",  weight=0.8),
]


# ---------------------------------------------------------------------------
# 4. Propagate
# ---------------------------------------------------------------------------

propagator = ConfidencePropagator(config=config)
result = propagator.propagate(nodes, edges)


# ---------------------------------------------------------------------------
# 5. Inspect results
# ---------------------------------------------------------------------------

print("=== Confidence Summary ===")
for node_id, confidence in result.summary().items():
    r = result.node_results[node_id]
    print(
        f"  {node_id:<20} confidence={confidence:.4f}"
        f"  singular={r.is_singular}"
        f"  double_pole={r.is_double_pole}"
    )
print()

print("=== Breakdown: kain_incident ===")
for key, value in result.breakdown("kain_incident").items():
    print(f"  {key:<25} {value}")
print()

print("=== Breakdown: rift_opening ===")
for key, value in result.breakdown("rift_opening").items():
    print(f"  {key:<25} {value}")
print()

print("=== Singular Nodes ===")
print(" ", result.singular_nodes())
print()

print("=== Double Pole Nodes ===")
print(" ", result.double_pole_nodes())
print()


# ---------------------------------------------------------------------------
# 6. DAG example: two upstream paths converging
# ---------------------------------------------------------------------------

print("=== DAG Example: Two paths to one node ===")

dag_nodes = [
    ConfidenceNode(
        id="source_a",
        label="Source A",
        causal_position=1,
        evidence=[
            EvidenceItem(key="ev1", source_type="primary", strength=0.9),
            EvidenceItem(key="ev2", source_type="primary", strength=0.8),
        ],
    ),
    ConfidenceNode(
        id="source_b",
        label="Source B",
        causal_position=1,
        evidence=[
            EvidenceItem(key="ev3", source_type="tertiary", strength=0.3),
        ],
    ),
    ConfidenceNode(
        id="conclusion",
        label="Conclusion",
        causal_position=2,
        evidence=[
            EvidenceItem(key="ev4", source_type="secondary", strength=0.6),
        ],
    ),
    ConfidenceNode(
        id="final",
        label="Final",
        causal_position=3,
        evidence=[
            EvidenceItem(key="ev5", source_type="primary", strength=0.7),
        ],
    ),
]

dag_edges = [
    PropagationEdge(source_id="source_a",   target_id="conclusion", weight=0.9),
    PropagationEdge(source_id="source_b",   target_id="conclusion", weight=0.3),
    PropagationEdge(source_id="conclusion", target_id="final",      weight=1.0),
]

dag_result = propagator.propagate(dag_nodes, dag_edges)

for node_id, confidence in dag_result.summary().items():
    r = dag_result.node_results[node_id]
    print(f"  {node_id:<15} confidence={confidence:.4f}  paths={r.ladder_result.path_count}")
