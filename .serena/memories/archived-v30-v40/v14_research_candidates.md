# V14 Research Candidates - Ralph Loop Next-Generation Enhancements

## Research Date: 2026-01-22

## Top V14 Candidate Subsystems

### 1. Causal World Models (HIGH PRIORITY)
**Source**: "Robust agents learn causal world models" (ICLR 2024), "Language Agents Meet Causality" (ICLR 2025)

**Key Insight**: Any agent capable of satisfying a regret bound under distributional shifts must have learned an approximate causal model.

**Implementation Ideas**:
- Structural Causal Model (SCM) for Ralph Loop state transitions
- Counterfactual reasoning for "what-if" scenario planning
- Causal discovery from iteration history
- Intervention planning based on causal graph

**Data Structures**:
```python
@dataclass
class CausalVariable:
    name: str
    parents: List[str]
    mechanism: Callable
    
@dataclass  
class CausalWorldModel:
    variables: Dict[str, CausalVariable]
    adjacency_matrix: np.ndarray
    intervention_history: List[Intervention]
```

### 2. Hierarchical World Models with Causal Curation (HIGH PRIORITY)
**Source**: NeurIPS 2025 Position Paper - "Hierarchical World Models with Causal Curation for Generalizing Agents"

**Three-Level Framework**:
1. **Low-level Executors**: Context-specific predictive models for motor control
2. **Mid-level Controllers**: Skill execution and composition
3. **High-level Curator**: Counterfactual reasoning, maintains causal library

**Key Concept**: Recast agent intelligence as proactive "curation" of causal knowledge for robust generalization.

**Implementation Ideas**:
- Curator maintains compact library of transferable causal knowledge
- Counterfactual reasoning over imagined tasks
- Skills/knowledge pruned based on transfer potential

### 3. Intrinsic Metacognitive Learning (MEDIUM PRIORITY)
**Source**: PMLR 267, 2025 - Position paper on self-improving agents

**Key Insight**: Sustained self-improvement requires intrinsic metacognition:
1. **Metacognitive Monitoring**: Evaluate own learning progress
2. **Metacognitive Control**: Adapt learning strategies
3. **Metacognitive Knowledge**: Represent beliefs about own capabilities

**Implementation Ideas**:
- Learning-to-learn module that adapts hyperparameters
- Self-evaluation of exploration vs exploitation balance
- Confidence calibration on predictions

### 4. Self-Adapting Language Models (SEAL) (MEDIUM PRIORITY)
**Source**: arXiv:2506.10943 (2025)

**Mechanism**:
- Model generates "self-edits" (finetuning data + update directives)
- RL loop rewards effective self-edits
- Persistent weight updates via SFT

**Integration Point**: Could enhance V13 Program Synthesis with self-editing capabilities.

### 5. Recursive Self-Improvement (RSI) Framework (HIGH PRIORITY)
**Source**: ICLR 2026 Workshop on AI with Recursive Self-Improvement

**Real-world RSI Examples**:
- LLM agents rewriting own code/prompts
- Scientific discovery pipelines with continual fine-tuning
- Robotics stacks patching controllers

**Implementation Ideas**:
- Formal verification of self-modifications
- Safety constraints on recursive improvement
- Bounded optimization to prevent unbounded changes

### 6. Temporal Hierarchies (THICK) (LOW PRIORITY)
**Source**: ICLR 2024 - "Learning Hierarchical World Models with Adaptive Temporal Abstractions"

**Key Concept**: Discrete latent dynamics enable reasoning across multiple time scales.

**Implementation Ideas**:
- Multi-scale planning (iteration, session, project levels)
- Categorical temporal abstractions
- Emergent skill hierarchies

## Recommended V14 Implementation Order

### Phase 1: Causal Foundation
1. **CausalWorldModel** dataclass and graph representation
2. **_discover_causal_structure()** from iteration history
3. **_counterfactual_reasoning()** for what-if analysis

### Phase 2: Hierarchical Curation
4. **CausalCurator** high-level knowledge management
5. **_curate_causal_library()** prune non-transferable knowledge
6. **_plan_with_counterfactuals()** imagined trajectory planning

### Phase 3: Metacognitive Layer
7. **MetacognitiveMonitor** self-evaluation module
8. **_adapt_learning_strategy()** based on performance
9. **_calibrate_confidence()** on predictions

### Phase 4: RSI Safety
10. **RSIConstraints** bounded self-modification
11. **_verify_self_modification()** safety checks
12. **_rollback_mechanism()** for failed modifications

## Research Sources

| Paper | Venue | Key Contribution |
|-------|-------|------------------|
| Hierarchical World Models with Causal Curation | NeurIPS 2025 | Three-level causal curation framework |
| Language Agents Meet Causality | ICLR 2025 | LLM + causal world model integration |
| CausalARC | NeurIPS 2025 LAW | Causal reasoning benchmark |
| Robust agents learn causal world models | ICLR 2024 | Theoretical foundation |
| Intrinsic Metacognitive Learning | PMLR 2025 | Metacognition requirements |
| SEAL | arXiv 2025 | Self-adapting LLMs |
| ICLR 2026 RSI Workshop | ICLR 2026 | RSI algorithmic foundations |
| THICK | ICLR 2024 | Temporal hierarchies |

## V14 vs V13 Comparison

| Aspect | V13 | V14 (Proposed) |
|--------|-----|----------------|
| Generalization | Compositional (SCAN-style) | Causal (intervention-based) |
| Adaptation | Meta-RL (MAML-style) | Metacognitive (self-evaluating) |
| Synthesis | Evolutionary (AlphaEvolve) | Causal Program Induction |
| Planning | Single-level | Hierarchical with curation |
| Self-Improvement | Fixed algorithms | RSI with safety bounds |

## Next Steps
1. Prototype CausalWorldModel dataclass
2. Implement causal discovery from existing iteration history
3. Add counterfactual reasoning to run_iteration()
4. Design safety constraints for RSI
