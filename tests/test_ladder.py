"""tests/test_ladder.py"""
import math
import pytest
from ecp.schema import ChainConfig, ConfidenceNode, EvidenceItem
from ecp.ladder import LadderOperator, LadderResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return ChainConfig()


@pytest.fixture
def op(config):
    return LadderOperator(config)


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


# ---------------------------------------------------------------------------
# Raising factor
# ---------------------------------------------------------------------------

class TestRaisingFactor:
    def test_at_start(self, op):
        # a⁺|0⟩ = √1/√(n_max+1) × damping = √1/√6
        expected = math.sqrt(1) / math.sqrt(6) * 1.0
        assert op.raising_factor(0, 5) == pytest.approx(expected)

    def test_at_position_2(self, op):
        # a⁺|2⟩ = √3/√6
        expected = math.sqrt(3) / math.sqrt(6) * 1.0
        assert op.raising_factor(2, 5) == pytest.approx(expected)

    def test_at_end_returns_zero(self, op):
        # a⁺|N⟩ = 0
        assert op.raising_factor(5, 5) == 0.0

    def test_beyond_end_returns_zero(self, op):
        assert op.raising_factor(6, 5) == 0.0

    def test_damping_applied(self):
        op_damped = LadderOperator(ChainConfig(damping=0.5))
        expected = math.sqrt(1) / math.sqrt(6) * 0.5
        assert op_damped.raising_factor(0, 5) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Lowering factor
# ---------------------------------------------------------------------------

class TestLoweringFactor:
    def test_at_start_returns_zero(self, op):
        # a⁻|0⟩ = 0
        assert op.lowering_factor(0) == 0.0

    def test_at_position_1(self, op):
        # a⁻|1⟩ = √1 = 1.0
        assert op.lowering_factor(1) == pytest.approx(math.sqrt(1))

    def test_at_position_4(self, op):
        # a⁻|4⟩ = √4 = 2.0
        assert op.lowering_factor(4) == pytest.approx(math.sqrt(4))

    def test_damping_applied(self):
        op_damped = LadderOperator(ChainConfig(damping=0.5))
        assert op_damped.lowering_factor(4) == pytest.approx(math.sqrt(4) * 0.5)


# ---------------------------------------------------------------------------
# Single node evaluation
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_root_node(self, op):
        node = make_node("a", evidence_count=1, causal_position=0)
        result = op.evaluate(node, chain_length=5)
        assert isinstance(result, LadderResult)
        assert result.node_id == "a"

    def test_root_lowering_zero_gives_epsilon(self, op):
        """Root node at position 0: lowering=0 → propagated clips to epsilon."""
        node = make_node("a", evidence_count=1, causal_position=0)
        result = op.evaluate(node, chain_length=5)
        assert result.propagated_confidence >= op.config.epsilon

    def test_inner_node_with_upstream(self, op):
        node = make_node("b", evidence_count=2, causal_position=2)
        upstream = [(0.7, 1.0)]
        result = op.evaluate(node, chain_length=5, upstream_contributions=upstream)
        assert result.propagated_confidence > 0.0
        assert result.path_count == 1

    def test_dag_two_paths(self, op):
        node = make_node("c", evidence_count=1, causal_position=3)
        upstream = [(0.8, 1.0), (0.4, 0.5)]
        result = op.evaluate(node, chain_length=5, upstream_contributions=upstream)
        assert result.path_count == 2

    def test_dag_high_weight_dominates(self, op):
        node = make_node("c", evidence_count=1, causal_position=2)
        upstream_high = [(0.9, 10.0), (0.1, 1.0)]
        upstream_low  = [(0.9, 1.0),  (0.1, 10.0)]
        r_high = op.evaluate(node, chain_length=5, upstream_contributions=upstream_high)
        r_low  = op.evaluate(node, chain_length=5, upstream_contributions=upstream_low)
        assert r_high.propagated_confidence > r_low.propagated_confidence

    def test_chain_end_raising_zero(self, op):
        """At chain end, raising=0 so propagated from upstream is 0 → clips to epsilon."""
        node = make_node("end", evidence_count=1, causal_position=4)
        upstream = [(0.8, 1.0)]
        result = op.evaluate(node, chain_length=5, upstream_contributions=upstream)
        assert result.raising_factor == 0.0
        # New ladder: upstream is passed through weighted average, not filtered by raising
        assert result.propagated_confidence >= op.config.epsilon

    def test_clip_prevents_above_one(self, op):
        node = make_node("a", evidence_count=5, causal_position=3)
        upstream = [(1.0, 1.0)] * 10
        result = op.evaluate(node, chain_length=5, upstream_contributions=upstream)
        assert result.propagated_confidence <= 1.0

    def test_invalid_chain_length_raises(self, op):
        node = make_node("a")
        with pytest.raises(ValueError, match="chain_length"):
            op.evaluate(node, chain_length=0)

    def test_damping_reduces_confidence(self):
        op_full   = LadderOperator(ChainConfig(damping=1.0))
        op_damped = LadderOperator(ChainConfig(damping=0.5))
        node = make_node("a", evidence_count=2, causal_position=2)
        upstream = [(0.8, 1.0)]
        r_full   = op_full.evaluate(node, chain_length=5, upstream_contributions=upstream)
        r_damped = op_damped.evaluate(node, chain_length=5, upstream_contributions=upstream)
        assert r_full.propagated_confidence >= r_damped.propagated_confidence

    def test_n_normalized_range(self, op):
        for pos in [0, 1, 2, 3, 4]:
            node = make_node("a", causal_position=pos)
            result = op.evaluate(node, chain_length=5)
            assert 0.0 <= result.n_normalized <= 1.0
