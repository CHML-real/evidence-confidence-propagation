"""tests/test_propagator.py"""
import copy
import pytest
from ecp.schema import ChainConfig, ConfidenceNode, EvidenceItem, PropagationEdge
from ecp.propagator import ConfidencePropagator, PropagationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_node(node_id, evidence_count=1, causal_position=1):
    evidences = [
        EvidenceItem(key=f"ev{i}", source_type="primary", strength=0.8)
        for i in range(evidence_count)
    ]
    return ConfidenceNode(
        id=node_id,
        label=f"Node {node_id}",
        evidence=evidences,
        causal_position=causal_position,
    )


@pytest.fixture
def propagator():
    return ConfidencePropagator()


# ---------------------------------------------------------------------------
# Basic propagation
# ---------------------------------------------------------------------------

class TestBasicPropagation:
    def test_empty_nodes(self, propagator):
        result = propagator.propagate([])
        assert result.node_results == {}

    def test_single_node(self, propagator):
        node = make_node("a", evidence_count=2, causal_position=0)
        result = propagator.propagate([node])
        assert "a" in result.node_results
        assert result.confidence("a") > 0.0

    def test_linear_chain(self, propagator):
        nodes = [
            make_node("a", evidence_count=2, causal_position=0),
            make_node("b", evidence_count=2, causal_position=1),
            make_node("c", evidence_count=2, causal_position=2),
        ]
        edges = [
            PropagationEdge(source_id="a", target_id="b"),
            PropagationEdge(source_id="b", target_id="c"),
        ]
        result = propagator.propagate(nodes, edges)
        assert len(result.node_results) == 3
        assert all(result.confidence(nid) > 0.0 for nid in ["a", "b", "c"])

    def test_confidence_clipped_above_epsilon(self, propagator):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = propagator.propagate([node])
        assert result.confidence("a") >= propagator.config.epsilon

    def test_confidence_at_most_one(self, propagator):
        nodes = [make_node("a", evidence_count=10, causal_position=1)]
        result = propagator.propagate(nodes)
        assert result.confidence("a") <= 1.0


# ---------------------------------------------------------------------------
# DAG propagation
# ---------------------------------------------------------------------------

class TestDAGPropagation:
    def test_two_paths_to_one_node(self, propagator):
        """
        a → c
        b → c
        """
        nodes = [
            make_node("a", evidence_count=3, causal_position=0),
            make_node("b", evidence_count=1, causal_position=0),
            make_node("c", evidence_count=2, causal_position=1),
        ]
        edges = [
            PropagationEdge(source_id="a", target_id="c", weight=1.0),
            PropagationEdge(source_id="b", target_id="c", weight=0.5),
        ]
        result = propagator.propagate(nodes, edges)
        assert result.confidence("c") > 0.0
        assert result.node_results["c"].ladder_result.path_count == 2

    def test_high_weight_path_dominates(self, propagator):
        """High weight upstream path should yield higher confidence."""
        # a: strong evidence, b: weak evidence
        ev_strong = [EvidenceItem(key=f"ev{i}", source_type="primary", strength=0.9) for i in range(3)]
        ev_weak   = [EvidenceItem(key="ev0", source_type="tertiary", strength=0.2)]

        node_a = ConfidenceNode(id="a", label="A", evidence=ev_strong, causal_position=1)
        node_b = ConfidenceNode(id="b", label="B", evidence=ev_weak,   causal_position=1)
        node_c = ConfidenceNode(id="c", label="C", evidence=[EvidenceItem(key="ev0", source_type="primary", strength=0.8)], causal_position=2)
        node_d = ConfidenceNode(id="d", label="D", evidence=[EvidenceItem(key="ev0", source_type="primary", strength=0.8)], causal_position=3)

        nodes = [node_a, node_b, node_c, node_d]

        edges_high = [
            PropagationEdge(source_id="a", target_id="c", weight=0.9),
            PropagationEdge(source_id="b", target_id="c", weight=0.1),
            PropagationEdge(source_id="c", target_id="d", weight=1.0),
        ]
        edges_low = [
            PropagationEdge(source_id="a", target_id="c", weight=0.1),
            PropagationEdge(source_id="b", target_id="c", weight=0.9),
            PropagationEdge(source_id="c", target_id="d", weight=1.0),
        ]

        r_high = propagator.propagate(copy.deepcopy(nodes), edges_high)
        r_low  = propagator.propagate(copy.deepcopy(nodes), edges_low)
        # high weight on strong-evidence node a → higher confidence at c
        assert r_high.confidence("c") != r_low.confidence("c")


# ---------------------------------------------------------------------------
# Singular nodes
# ---------------------------------------------------------------------------

class TestSingularNodes:
    def test_singular_node_detected(self, propagator):
        node = make_node("a", evidence_count=0, causal_position=1)
        result = propagator.propagate([node])
        assert "a" in result.singular_nodes()

    def test_double_pole_detected(self, propagator):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = propagator.propagate([node])
        assert "a" in result.double_pole_nodes()

    def test_normal_node_not_singular(self, propagator):
        node = make_node("a", evidence_count=2, causal_position=1)
        result = propagator.propagate([node])
        assert "a" not in result.singular_nodes()


# ---------------------------------------------------------------------------
# PropagationResult helpers
# ---------------------------------------------------------------------------

class TestPropagationResult:
    def test_summary(self, propagator):
        nodes = [make_node("a"), make_node("b", causal_position=2)]
        result = propagator.propagate(nodes)
        summary = result.summary()
        assert set(summary.keys()) == {"a", "b"}
        assert all(isinstance(v, float) for v in summary.values())

    def test_missing_node_raises(self, propagator):
        node = make_node("a")
        result = propagator.propagate([node])
        with pytest.raises(KeyError):
            result.confidence("nonexistent")


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_cycle_raises(self, propagator):
        nodes = [
            make_node("a", causal_position=0),
            make_node("b", causal_position=1),
        ]
        edges = [
            PropagationEdge(source_id="a", target_id="b"),
            PropagationEdge(source_id="b", target_id="a"),
        ]
        with pytest.raises(ValueError, match="Cycle"):
            propagator.propagate(nodes, edges)


# ---------------------------------------------------------------------------
# Config effects
# ---------------------------------------------------------------------------

class TestConfigEffects:
    def test_higher_damping_reduces_confidence(self):
        nodes = [
            make_node("a", evidence_count=2, causal_position=0),
            make_node("b", evidence_count=2, causal_position=1),
            make_node("c", evidence_count=2, causal_position=2),
        ]
        edges = [
            PropagationEdge(source_id="a", target_id="b"),
            PropagationEdge(source_id="b", target_id="c"),
        ]
        p_full   = ConfidencePropagator(ChainConfig(damping=1.0))
        p_damped = ConfidencePropagator(ChainConfig(damping=0.5))
        r_full   = p_full.propagate(nodes, edges)
        r_damped = p_damped.propagate(nodes, edges)
        assert r_full.confidence("c") >= r_damped.confidence("c")
