# evidence-confidence-propagation

[![PyPI version](https://badge.fury.io/py/evidence-confidence-propagation.svg)](https://pypi.org/project/evidence-confidence-propagation/)
[![Python](https://img.shields.io/pypi/pyversions/evidence-confidence-propagation)](https://pypi.org/project/evidence-confidence-propagation/)
[![Tests](https://github.com/CHML-real/evidence-confidence-propagation/actions/workflows/tests.yml/badge.svg)](https://github.com/CHML-real/evidence-confidence-propagation/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A Python package for computing **evidence-weighted confidence scores** along causal chains and DAG structures using residue theory and ladder operators.

Instead of treating confidence as a simple count or average, `evidence-confidence-propagation` models how trustworthiness propagates through a causal chain — accounting for source reliability, chain position, and singular boundary conditions.

---

## Why this exists

Existing packages compute confidence as a flat score from evidence count or source trust alone. This ignores two critical structural factors:

- **Where** in a causal chain an item sits (chain boundaries behave differently)
- **How** confidence flows from upstream nodes to downstream nodes

`evidence-confidence-propagation` was built to handle exactly this:

- A narrative timeline where early events have no prior support
- A lore database where some items sit at the end of long inference chains
- A historical reconstruction where confidence must decay naturally along causal paths
- Any system where **"how much should I trust this item given its position and evidence"** needs a principled answer

---

## Mathematical basis

Confidence is computed as:

```
confidence(v) = base_score(v)
              × ladder_propagation(v)
              - residue_penalty(v)
              + laurent_correction(v)
```

### Singular points

Singular points arise when:
- `causal_position = 0` — chain boundary (simple pole)
- `evidence_count = 0` — no supporting evidence (simple pole)
- Both simultaneously — double pole → Laurent series correction applied

### Ladder operators

Confidence propagates through the chain using raising/lowering operators inspired by the quantum harmonic oscillator:

```
a⁺|n⟩ = √(n+1) × damping × |n+1⟩   (forward propagation)
a⁻|n⟩ = √n     × damping × |n-1⟩   (backward propagation)

Boundary: a⁻|0⟩ = 0,  a⁺|N⟩ = 0
```

For branching DAG chains, confidence at a node with multiple upstream paths is computed as a weighted average of all path contributions.

### Residue correction

At singular points, the residue of the confidence function is extracted using contour integral theory and applied as a penalty. At double poles, the Laurent series `a₋₂` coefficient drives an additional correction.

---

## Installation

```bash
pip install evidence-confidence-propagation
```

Requires Python 3.10 or later. No external dependencies.

---

## Quick start

```python
from ecp import (
    EvidenceItem,
    ConfidenceNode,
    PropagationEdge,
    ChainConfig,
    ConfidencePropagator,
)

# 1. Configure
config = ChainConfig(
    epsilon=0.01,
    residue_penalty_scale=1.0,
    laurent_correction_scale=0.5,
    damping=0.9,
)

# 2. Define nodes
nodes = [
    ConfidenceNode(
        id="kain_incident",
        label="Kain Incident",
        causal_position=0,
        evidence=[
            EvidenceItem(key="official_001", source_type="primary", strength=0.9),
            EvidenceItem(key="fan_002",      source_type="tertiary", strength=0.4),
        ],
    ),
    ConfidenceNode(
        id="rift_opening",
        label="Rift Opening",
        causal_position=1,
        evidence=[
            EvidenceItem(key="official_003", source_type="primary", strength=0.8),
        ],
    ),
    ConfidenceNode(
        id="archive_fall",
        label="Archive Fall",
        causal_position=2,
        evidence=[
            EvidenceItem(key="secondary_004", source_type="secondary", strength=0.6),
        ],
    ),
]

# 3. Define causal edges
edges = [
    PropagationEdge(source_id="kain_incident", target_id="rift_opening",  weight=1.0),
    PropagationEdge(source_id="rift_opening",  target_id="archive_fall",  weight=0.8),
]

# 4. Propagate
propagator = ConfidencePropagator(config=config)
result = propagator.propagate(nodes, edges)

# 5. Inspect
for node_id, r in result.node_results.items():
    print(f"{node_id}: confidence={r.confidence:.4f}, singular={r.is_singular}")

print(result.summary())
print("Singular nodes:", result.singular_nodes())
print("Double poles:",   result.double_pole_nodes())
```

---

## Core concepts

### EvidenceItem

A single piece of evidence supporting a node.

```python
EvidenceItem(
    key="official_001",
    source_type="primary",    # primary / secondary / tertiary / unknown
    strength=0.9,             # (0, 1]
)
```

Source type default weights:

| Source type | Weight |
|-------------|--------|
| `primary`   | 1.0    |
| `secondary` | 0.6    |
| `tertiary`  | 0.3    |
| `unknown`   | 0.1    |

### ConfidenceNode

A single item in the causal chain.

```python
ConfidenceNode(
    id="event_a",
    label="Event A",
    causal_position=1,        # position in chain; None = unassigned
    evidence=[...],
)
```

Key properties:

| Property | Meaning |
|----------|---------|
| `is_singular` | True when position=0/None OR evidence_count=0 |
| `is_double_pole` | True when both conditions hold simultaneously |
| `base_score` | Weighted average of evidence strengths |

### ConfidencePropagator

Runs full confidence propagation in topological order.

```python
result = ConfidencePropagator(config).propagate(nodes, edges)
result.confidence("event_a")   # float
result.summary()               # {node_id: confidence}
result.singular_nodes()        # list of singular node ids
result.double_pole_nodes()     # list of double pole node ids
```

### ChainConfig

```python
ChainConfig(
    epsilon=0.01,                    # minimum confidence floor
    residue_penalty_scale=1.0,       # scales residue penalty
    laurent_correction_scale=0.5,    # scales double pole correction
    damping=1.0,                     # ladder propagation decay per step
)
```

---

## Package structure

```
ecp/
├── schema.py      EvidenceItem, ConfidenceNode, PropagationEdge, ChainConfig
├── residue.py     ResidueEngine, ResidueResult
├── ladder.py      LadderOperator, LadderResult
└── propagator.py  ConfidencePropagator, PropagationResult, NodeConfidenceResult
```

---

## Relationship to temporal-belief-graph

This package is designed to complement [`temporal-belief-graph`](https://github.com/CHML-real/temporal-belief-graph).

| Package | Question answered |
|---------|-------------------|
| `temporal-belief-graph` | "Does A happen before B?" (ordering probability) |
| `evidence-confidence-propagation` | "How much should I trust this item?" (confidence score) |

They operate on different layers and can be used together.

---

## Development

```bash
pip install -e ".[dev]"
pytest
tox
python -m build
```

> This package is currently in alpha. APIs may change before version 1.0.0.

---

## Contributing

Contributions, issues, and feature requests are welcome.  
Please open an issue before submitting a pull request.

---

## Contributors

| Handle | GitHub | Role |
|--------|--------|------|
| lajjadred | [@lajjadred](https://github.com/lajjadred) | Project lead |
| Chae Mun Lee | [@CHML-real](https://github.com/CHML-real) | Mathematical algorithm development |
| CUBE | [@90cube](https://github.com/90cube) | Idea proposal and data collection |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
