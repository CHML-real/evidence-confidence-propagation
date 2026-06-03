"""tests/test_residue.py"""
import pytest
from ecp.schema import ChainConfig, ConfidenceNode, EvidenceItem
from ecp.residue import ResidueEngine, ResidueResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return ChainConfig()


@pytest.fixture
def engine(config):
    return ResidueEngine(config)


def make_node(node_id, evidence_count=0, causal_position=1):
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
# Pole classification
# ---------------------------------------------------------------------------

class TestPoleClassification:
    def test_no_pole(self, engine):
        node = make_node("a", evidence_count=2, causal_position=1)
        result = engine.compute(node, chain_length=5)
        assert result.pole_order == 0

    def test_simple_pole_no_evidence(self, engine):
        node = make_node("a", evidence_count=0, causal_position=2)
        result = engine.compute(node, chain_length=5)
        assert result.pole_order == 1

    def test_simple_pole_position_zero(self, engine):
        node = make_node("a", evidence_count=2, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.pole_order == 1

    def test_double_pole(self, engine):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.pole_order == 2

    def test_double_pole_none_position(self, engine):
        node = ConfidenceNode(id="a", label="A")  # causal_position=None, no evidence
        result = engine.compute(node, chain_length=5)
        assert result.pole_order == 2


# ---------------------------------------------------------------------------
# No pole
# ---------------------------------------------------------------------------

class TestNoPole:
    def test_zero_penalty(self, engine):
        node = make_node("a", evidence_count=2, causal_position=1)
        result = engine.compute(node, chain_length=5)
        assert result.penalty == 0.0
        assert result.correction == 0.0
        assert result.net_adjustment == 0.0

    def test_zero_residue(self, engine):
        node = make_node("a", evidence_count=2, causal_position=1)
        result = engine.compute(node, chain_length=5)
        assert result.residue == 0.0


# ---------------------------------------------------------------------------
# Simple pole
# ---------------------------------------------------------------------------

class TestSimplePole:
    def test_penalty_positive(self, engine):
        node = make_node("a", evidence_count=0, causal_position=2)
        result = engine.compute(node, chain_length=5)
        assert result.penalty >= 0.0

    def test_no_laurent_correction(self, engine):
        node = make_node("a", evidence_count=0, causal_position=2)
        result = engine.compute(node, chain_length=5)
        assert result.correction == 0.0
        assert result.laurent_a2 == 0.0

    def test_boundary_penalty_larger(self, engine):
        """Node at position 0 gets larger penalty than node further in chain."""
        node_boundary = make_node("a", evidence_count=0, causal_position=0)
        node_inner    = make_node("b", evidence_count=0, causal_position=3)
        r_boundary = engine.compute(node_boundary, chain_length=5)
        r_inner    = engine.compute(node_inner, chain_length=5)
        assert r_boundary.penalty >= r_inner.penalty

    def test_penalty_scale_applied(self):
        config_high = ChainConfig(residue_penalty_scale=2.0)
        config_low  = ChainConfig(residue_penalty_scale=0.5)
        node = make_node("a", evidence_count=0, causal_position=2)
        r_high = ResidueEngine(config_high).compute(node, chain_length=5)
        r_low  = ResidueEngine(config_low).compute(node, chain_length=5)
        assert r_high.penalty > r_low.penalty


# ---------------------------------------------------------------------------
# Double pole
# ---------------------------------------------------------------------------

class TestDoublePole:
    def test_pole_order_two(self, engine):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.pole_order == 2

    def test_has_laurent_a2(self, engine):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.laurent_a2 > 0.0

    def test_has_correction(self, engine):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.correction >= 0.0

    def test_net_adjustment_positive(self, engine):
        """Double pole net adjustment should be positive (penalty > correction)."""
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.net_adjustment >= 0.0

    def test_laurent_scale_affects_correction(self):
        config_high = ChainConfig(laurent_correction_scale=1.0)
        config_low  = ChainConfig(laurent_correction_scale=0.1)
        node = make_node("a", evidence_count=0, causal_position=0)
        r_high = ResidueEngine(config_high).compute(node, chain_length=5)
        r_low  = ResidueEngine(config_low).compute(node, chain_length=5)
        assert r_high.correction > r_low.correction


# ---------------------------------------------------------------------------
# Chain length validation
# ---------------------------------------------------------------------------

class TestChainLength:
    def test_invalid_chain_length_raises(self, engine):
        node = make_node("a", evidence_count=1, causal_position=1)
        with pytest.raises(ValueError, match="chain_length"):
            engine.compute(node, chain_length=0)

    def test_chain_length_one(self, engine):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=1)
        assert result.pole_order == 2


# ---------------------------------------------------------------------------
# net_adjustment property
# ---------------------------------------------------------------------------

class TestNetAdjustment:
    def test_net_adjustment_no_pole(self, engine):
        node = make_node("a", evidence_count=2, causal_position=1)
        result = engine.compute(node, chain_length=5)
        assert result.net_adjustment == 0.0

    def test_net_adjustment_equals_penalty_minus_correction(self, engine):
        node = make_node("a", evidence_count=0, causal_position=0)
        result = engine.compute(node, chain_length=5)
        assert result.net_adjustment == pytest.approx(
            result.penalty - result.correction
        )
