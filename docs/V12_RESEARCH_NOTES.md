# V12 Research Notes - Iteration 25 (Phase 1)

## Date: 2026-01-22
## Focus: RIAL/DIAL, DARTS, VAE Memory Consolidation

---

## 1. RIAL/DIAL (Emergent Communication)

### Key Paper: Foerster et al. 2016 (arXiv:1605.06676)
"Learning to Communicate with Deep Multi-Agent Reinforcement Learning"

### RIAL (Reinforced Inter-Agent Learning)
- Uses deep Q-learning with recurrent network (DRQN)
- Messages are discrete symbols from fixed vocabulary
- Policy gradient updates: `∇J = E[Q(s,a,m) ∇log π(m|s)]`
- Messages treated as discrete actions

### DIAL (Differentiable Inter-Agent Learning)
- End-to-end differentiable during training
- Continuous relaxation during training, discretize at execution
- Uses DRU (Discretize/Regularize Unit):
  - Training: `m = σ(μ) + N(0, σ²)` (continuous)
  - Execution: `m = 1 if logit > 0 else 0` (discrete)
- Backprop through communication channel

### Implementation Pattern (from minqi/learning-to-communicate-pytorch)
```python
class DRU(nn.Module):
    """Discretize/Regularize Unit for DIAL"""
    def forward(self, m, train_mode=True):
        if train_mode:
            # Training: pass gradients through
            return torch.sigmoid(m) + torch.randn_like(m) * 0.5
        else:
            # Execution: discretize
            return (m > 0).float()
```

### Key Implementation Decisions for `_run_communication_round()`:
1. Support both RIAL (discrete messages, policy gradient) and DIAL (differentiable)
2. Track `communication_success_rate` and `compositionality_score`
3. Use `information_bottleneck` parameter for compression pressure
4. Maintain `emergent_vocabulary` mapping symbols to meanings

---

## 2. DARTS (Differentiable Architecture Search)

### Key Paper: Liu et al. 2018 (arXiv:1806.09055, ICLR 2019)

### Core Idea: Continuous Relaxation
- Discrete architecture choice → continuous mixture
- Instead of selecting one op, use weighted sum:
  ```
  ō(x) = Σ [exp(α_o) / Σ exp(α_o')] × o(x)
  ```
- Bilevel optimization:
  - Inner loop: train network weights w
  - Outer loop: train architecture params α

### Search Space (from NATS-Bench)
Operations: `sep_conv_3x3, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5, max_pool_3x3, avg_pool_3x3, skip_connect, none`

### Genotype Structure
```python
Genotype(
    normal=[  # Normal cell operations
        [('sep_conv_3x3', 0), ('dil_conv_5x5', 1)],  # Node 2
        [('skip_connect', 0), ('dil_conv_3x3', 2)],  # Node 3
        [('sep_conv_3x3', 1), ('skip_connect', 0)],  # Node 4
        [('sep_conv_3x3', 1), ('skip_connect', 0)]   # Node 5
    ],
    normal_concat=range(2, 6),
    reduce=[...],  # Reduction cell
    reduce_concat=range(2, 6)
)
```

### Key Implementation Decisions for `_evaluate_architecture_candidate()`:
1. Calculate validation accuracy from architecture params
2. Estimate training cost (FLOPs, latency)
3. Update Pareto front for multi-objective optimization
4. Support multiple search strategies: darts, enas, random, evolutionary

### Scoring Formula
```python
def evaluate_candidate(candidate):
    # Accuracy component
    accuracy_score = candidate.validation_accuracy

    # Efficiency component (lower is better)
    efficiency_score = 1.0 / (candidate.training_cost + 1)

    # Combined score (Pareto-aware)
    combined = accuracy_score * 0.7 + efficiency_score * 0.3
    return combined
```

---

## 3. VAE Memory Consolidation (Generative Replay)

### Key Paper: Spens & Burgess 2024 (Nature Human Behaviour)
"A generative model of memory construction and consolidation"

### Brain-Inspired Architecture
- **Hippocampus (Teacher)**: Autoassociative network for episodic storage
- **Neocortex (Student)**: VAE for semantic compression
- **Consolidation**: Teacher replays to train student

### Teacher-Student Training Pattern
```python
# Teacher: Hopfield/Autoassociative Network
teacher_output = hopfield_network.recall(cue)

# Student: VAE
z = vae.encode(teacher_output)  # Compress to latent
reconstruction = vae.decode(z)

# Training loss
loss = reconstruction_loss + β * kl_divergence
```

### Generative Replay for Continual Learning
From Brain-inspired Replay (van de Ven et al. 2020):
1. Train VAE on task N experiences
2. When learning task N+1, interleave:
   - Real data from task N+1
   - Generated samples from VAE (pseudo-task N data)
3. Prevents catastrophic forgetting

### Implementation for `_run_memory_consolidation()`:
```python
def _run_memory_consolidation(self):
    mc_state = self.state.memory_consolidation_state

    # 1. Sample from replay buffer with priority
    batch = mc_state.sample_for_replay(batch_size=32)

    # 2. Compress experiences via VAE-like process
    compressed = self._compress_experiences(batch)

    # 3. Generate pseudo-experiences for replay
    if mc_state.generative_replay_enabled:
        pseudo = self._generate_replay(compressed)

    # 4. Distill knowledge from teacher to student
    self._distill_knowledge(batch, compressed)

    # 5. Create consolidated memories
    memory = ConsolidatedMemory(
        memory_id=f"consolidated_{mc_state.consolidation_rounds}",
        compressed_representation=compressed,
        importance_score=self._calculate_importance(batch)
    )

    return {"consolidated": [memory], "compression_ratio": mc_state.compression_ratio}
```

---

## 4. get_v12_insights() Design

Should return comprehensive V12 subsystem status:
```python
def get_v12_insights(self) -> Dict[str, Any]:
    return {
        "world_models": {
            "prediction_accuracy": self.state.world_model_state.prediction_accuracy,
            "imagined_trajectories": len(self.state.world_model_state.imagined_trajectories),
            "model_error_history": self.state.world_model_state.model_error_history[-10:]
        },
        "predictive_coding": {
            "current_free_energy": self.state.predictive_coding_state.current_free_energy,
            "prediction_accuracy": self.state.predictive_coding_state.get_prediction_accuracy(),
            "precision_weights": self.state.predictive_coding_state.precision_weights
        },
        "active_inference": {
            "epistemic_value": self.state.active_inference_state.epistemic_value,
            "pragmatic_value": self.state.active_inference_state.pragmatic_value,
            "selected_policies": self.state.active_inference_state.selected_policies[-5:]
        },
        "emergent_communication": {
            "success_rate": self.state.emergent_communication_state.communication_success_rate,
            "compositionality": self.state.emergent_communication_state.compositionality_score,
            "vocabulary_size": len(self.state.emergent_communication_state.emergent_vocabulary),
            "total_messages": self.state.emergent_communication_state.total_messages
        },
        "neural_architecture_search": {
            "best_accuracy": self.state.nas_state.best_validation_accuracy,
            "search_iterations": self.state.nas_state.search_iterations,
            "pareto_front_size": len(self.state.nas_state.pareto_front)
        },
        "memory_consolidation": {
            "compression_ratio": self.state.memory_consolidation_state.compression_ratio,
            "consolidation_rounds": self.state.memory_consolidation_state.consolidation_rounds,
            "memories_stored": len(self.state.memory_consolidation_state.consolidated_memories)
        },
        "v12_methods_implemented": 11,  # After implementing remaining 4
        "v12_data_structures": 18
    }
```

---

## 5. Integration Points in run_iteration()

### Success Path Integration
```python
if fitness > self.best_fitness:
    # Run communication round for multi-agent coordination
    if self.state.emergent_communication_state:
        comm_result = self._run_communication_round()
        self.logger.info(f"V12 EC: {comm_result}")

    # Evaluate current architecture
    if self.state.nas_state:
        arch_eval = self._evaluate_architecture_candidate(current_architecture)
        self.state.nas_state.add_candidate(arch_eval)
```

### Periodic Consolidation (every N iterations)
```python
# Memory consolidation (sleep-like)
if self.state.memory_consolidation_state:
    if self.state.memory_consolidation_state.should_consolidate():
        consolidation_result = self._run_memory_consolidation()
        self.logger.info(f"V12 MC: {consolidation_result}")
```

---

## Implementation Timeline

| Phase | Iterations | Focus |
|-------|-----------|-------|
| Phase 1 | 25-34 | Research (COMPLETE) |
| Phase 2 | 35-54 | `_run_communication_round()` |
| Phase 3 | 55-74 | `_evaluate_architecture_candidate()` |
| Phase 4 | 75-94 | `_run_memory_consolidation()` |
| Phase 5 | 95-109 | `get_v12_insights()` + Integration |
| Phase 6 | 110-124 | Testing & Validation |

---

## References

1. Foerster et al. (2016) - Learning to Communicate with Deep Multi-Agent RL
2. Liu et al. (2018) - DARTS: Differentiable Architecture Search
3. Spens & Burgess (2024) - Generative model of memory consolidation
4. van de Ven et al. (2020) - Brain-inspired replay for continual learning
5. NATS-Bench - Neural Architecture Search Benchmark
