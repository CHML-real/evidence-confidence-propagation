"""tests/test_schema.py"""
import pytest
from ecp.schema import (
    EvidenceItem,
    ConfidenceNode,
    PropagationEdge,
    ChainConfig,
    source_weight,
)


class TestEvidenceItem:
    def test_normal(self):
        ev = EvidenceItem(key="ev1", source_type="primary", strength=0.8)
        assert ev.weighted_strength == pytest.approx(0.8 * 1.0)

    def test_secondary_weight(self):
        ev = EvidenceItem(key="ev1", source_type="secondary", strength=1.0)
        assert ev.weighted_strength == pytest.approx(0.6)

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key"):
            EvidenceItem(key="", source_type="primary")

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError, match="strength"):
            EvidenceItem(key="ev1", source_type="primary", strength=0.0)

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValueError, match="source_type"):
            EvidenceItem(key="ev1", source_type="invalid")

    def test_strength_exactly_one(self):
        ev = EvidenceItem(key="ev1", source_type="primary", strength=1.0)
        assert ev.strength == 1.0


class TestConfidenceNode:
    def test_normal(self):
        node = ConfidenceNode(id="a", label="Event A", causal_position=1)
        assert node.id == "a"
        assert node.evidence_count == 0

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id"):
            ConfidenceNode(id="", label="X")

    def test_negative_position_raises(self):
        with pytest.raises(ValueError, match="causal_position"):
            ConfidenceNode(id="a", label="A", causal_position=-1)

    def test_is_singular_no_evidence(self):
        node = ConfidenceNode(id="a", label="A", causal_position=1)
        assert node.is_singular is True

    def test_is_singular_position_zero(self):
        node = ConfidenceNode(id="a", label="A", causal_position=0)
        assert node.is_singular is True

    def test_is_singular_position_none(self):
        node = ConfidenceNode(id="a", label="A")
        assert node.is_singular is True

    def test_not_singular(self):
        ev = EvidenceItem(key="ev1", source_type="primary", strength=0.8)
        node = ConfidenceNode(id="a", label="A", evidence=[ev], causal_position=1)
        assert node.is_singular is False

    def test_is_double_pole(self):
        node = ConfidenceNode(id="a", label="A", causal_position=0)
        assert node.is_double_pole is True

    def test_is_double_pole_none_position(self):
        node = ConfidenceNode(id="a", label="A")
        assert node.is_double_pole is True

    def test_not_double_pole_has_evidence(self):
        ev = EvidenceItem(key="ev1", source_type="primary", strength=0.8)
        node = ConfidenceNode(id="a", label="A", evidence=[ev], causal_position=0)
        assert node.is_double_pole is False

    def test_base_score_no_evidence(self):
        node = ConfidenceNode(id="a", label="A")
        assert node.base_score == 0.0

    def test_base_score_with_evidence(self):
        ev = EvidenceItem(key="ev1", source_type="primary", strength=0.8)
        node = ConfidenceNode(id="a", label="A", evidence=[ev], causal_position=1)
        assert node.base_score == pytest.approx(0.8)

    def test_unicode_label(self):
        node = ConfidenceNode(id="kain", label="카인 사건", causal_position=1)
        assert node.label == "카인 사건"


class TestPropagationEdge:
    def test_normal(self):
        e = PropagationEdge(source_id="a", target_id="b", weight=0.8)
        assert e.weight == 0.8

    def test_self_loop_raises(self):
        with pytest.raises(ValueError):
            PropagationEdge(source_id="a", target_id="a")

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="source_id"):
            PropagationEdge(source_id="", target_id="b")

    def test_weight_out_of_range_raises(self):
        with pytest.raises(ValueError, match="weight"):
            PropagationEdge(source_id="a", target_id="b", weight=0.0)

    def test_weight_exactly_one(self):
        e = PropagationEdge(source_id="a", target_id="b", weight=1.0)
        assert e.weight == 1.0


class TestChainConfig:
    def test_defaults(self):
        c = ChainConfig()
        assert c.epsilon == 0.01
        assert c.damping == 1.0

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            ChainConfig(epsilon=0.0)

    def test_invalid_damping_raises(self):
        with pytest.raises(ValueError, match="damping"):
            ChainConfig(damping=0.0)

    def test_negative_penalty_scale_raises(self):
        with pytest.raises(ValueError, match="residue_penalty_scale"):
            ChainConfig(residue_penalty_scale=-1.0)


class TestSourceWeight:
    def test_primary(self):
        assert source_weight("primary") == 1.0

    def test_secondary(self):
        assert source_weight("secondary") == 0.6

    def test_unknown(self):
        assert source_weight("unknown") == 0.1
