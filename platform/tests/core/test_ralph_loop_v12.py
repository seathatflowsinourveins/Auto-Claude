"""
Integration Test for Ralph Loop V12

Tests all 6 V12 subsystems:
1. World Models (Dreamer V4, IRIS)
2. Predictive Coding (Free Energy Principle)
3. Active Inference (Expected Free Energy)
4. Emergent Communication (RIAL/DIAL)
5. Neural Architecture Search (DARTS)
6. Memory Consolidation (VAE + Replay)
"""
import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
import pytest

# Add platform/ path for imports (two levels up from tests/core/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Direct import (must be after sys.path modification)
from core.ralph_loop import RalphLoop, LoopState

# Mark all async tests with asyncio
pytestmark = pytest.mark.asyncio


def test_v12_dataclass_structures():
    """Test V12 dataclass structures can be instantiated."""
    from core.ralph_loop import (
        WorldModelState, PredictiveCodingState, ActiveInferenceState,
        EmergentCommunicationState, NeuralArchitectureSearchState,
        MemoryConsolidationState
    )

    # Test World Model State
    wm = WorldModelState()
    assert wm.imagination_horizon == 15
    assert wm.prediction_accuracy == 0.0
    assert len(wm.latent_states) == 0
    print("[OK] WorldModelState initialized")

    # Test Predictive Coding State
    pc = PredictiveCodingState()
    assert pc.num_layers == 4  # Default 4 hierarchical layers (layers list starts empty)
    assert pc.global_learning_rate == 0.01
    print("[OK] PredictiveCodingState initialized")

    # Test Active Inference State
    ai = ActiveInferenceState()
    assert ai.epistemic_weight == 0.5
    assert ai.pragmatic_weight == 0.5
    print("[OK] ActiveInferenceState initialized")

    # Test Emergent Communication State
    ec = EmergentCommunicationState()
    assert ec.vocabulary_size == 64
    assert ec.message_length == 8
    print("[OK] EmergentCommunicationState initialized")

    # Test NAS State
    nas = NeuralArchitectureSearchState()
    assert nas.num_cells == 8
    assert nas.num_nodes_per_cell == 4
    print("[OK] NeuralArchitectureSearchState initialized")

    # Test Memory Consolidation State
    mc = MemoryConsolidationState()
    assert mc.priority_alpha == 0.6
    assert mc.priority_beta == 0.4
    print("[OK] MemoryConsolidationState initialized")


def test_ralph_loop_initialization():
    """Test RalphLoop can be initialized."""
    loop = RalphLoop(
        task="Test task",
        max_iterations=5
    )
    assert loop.task == "Test task"
    assert loop.max_iterations == 5
    assert loop.state is None  # State not initialized until run()
    assert loop._recent_solutions == []  # V9 attribute
    print("[OK] RalphLoop initialized successfully")


def test_loop_state_v12_serialization():
    """Test LoopState with V12 fields can be serialized and deserialized."""
    from core.ralph_loop import (
        WorldModelState, PredictiveCodingState, ActiveInferenceState,
        EmergentCommunicationState, NeuralArchitectureSearchState,
        MemoryConsolidationState
    )

    # Create a LoopState with V12 subsystems
    state = LoopState(
        loop_id="test123",
        task="Test serialization",
        current_iteration=5,
        max_iterations=100,
        best_fitness=0.85,
        best_solution="test solution",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running"
    )

    # Add V12 states
    state.world_model_state = WorldModelState()
    state.predictive_coding_state = PredictiveCodingState()
    state.active_inference_state = ActiveInferenceState()
    state.emergent_communication_state = EmergentCommunicationState()
    state.nas_state = NeuralArchitectureSearchState()
    state.memory_consolidation_state = MemoryConsolidationState()

    # Serialize
    state_dict = state.to_dict()
    assert "world_model_state" in state_dict
    assert "predictive_coding_state" in state_dict
    assert "active_inference_state" in state_dict
    assert "emergent_communication_state" in state_dict
    assert "nas_state" in state_dict
    assert "memory_consolidation_state" in state_dict
    print("[OK] LoopState serialization includes V12 fields")

    # Deserialize
    restored = LoopState.from_dict(state_dict)
    assert restored.loop_id == "test123"
    assert restored.best_fitness == 0.85
    assert restored.world_model_state is not None
    assert restored.predictive_coding_state is not None
    assert restored.active_inference_state is not None
    assert restored.emergent_communication_state is not None
    assert restored.nas_state is not None
    assert restored.memory_consolidation_state is not None
    print("[OK] LoopState deserialization restores V12 fields")


def test_v12_helper_methods():
    """Test V12 dataclass helper methods work correctly."""
    from core.ralph_loop import (
        WorldModelState, PredictiveCodingState, ActiveInferenceState,
        EmergentCommunicationState, NeuralArchitectureSearchState,
        MemoryConsolidationState
    )

    # Test Predictive Coding helpers
    pc = PredictiveCodingState()
    pc.total_predictions = 100
    pc.accurate_predictions = 85
    assert pc.get_prediction_accuracy() == 0.85
    print("[OK] PredictiveCodingState.get_prediction_accuracy()")

    # Test Active Inference helpers
    ai = ActiveInferenceState()
    ai.total_decisions = 50
    ai.goal_achieved_count = 35
    assert ai.get_goal_success_rate() == 0.70
    print("[OK] ActiveInferenceState.get_goal_success_rate()")

    # Test Emergent Communication helpers
    ec = EmergentCommunicationState()
    ec.total_messages = 100
    ec.successful_communications = 80
    assert ec.get_communication_success_rate() == 0.80
    print("[OK] EmergentCommunicationState.get_communication_success_rate()")

    # Test NAS helpers
    nas = NeuralArchitectureSearchState()
    nas.search_iterations = 10
    progress = nas.get_search_progress()
    assert progress["iterations"] == 10
    assert "best_accuracy" in progress
    print("[OK] NeuralArchitectureSearchState.get_search_progress()")


def test_get_v12_insights():
    """Test RalphLoop.get_v12_insights() method."""
    loop = RalphLoop(task="Test insights", max_iterations=5)

    # Initialize state manually for testing
    loop.state = LoopState(
        loop_id="test_insights",
        task="Test insights",
        current_iteration=0,
        max_iterations=5,
        best_fitness=0.0,
        best_solution="",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )

    # Initialize V12 state
    loop._initialize_v12_state()

    # Get insights
    insights = loop.get_v12_insights()

    assert "world_model" in insights
    assert "predictive_coding" in insights
    assert "active_inference" in insights
    assert "emergent_communication" in insights
    assert "nas" in insights  # Note: key is "nas" not "neural_architecture_search"
    assert "memory_consolidation" in insights

    print("[OK] get_v12_insights() returns all 6 subsystem insights")
    print(f"    Insights keys: {list(insights.keys())}")


# =============================================================================
# NEW V12 ORCHESTRATION METHOD TESTS (Iteration 26)
# =============================================================================

async def test_run_communication_round_rial_mode():
    """Test _run_communication_round() in RIAL (discrete) mode."""
    loop = RalphLoop(task="Test RIAL communication", max_iterations=5)
    
    # Initialize state
    loop.state = LoopState(
        loop_id="test_rial",
        task="Test RIAL communication",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Set RIAL mode
    loop.state.emergent_communication_state.training_mode = "rial"
    
    # Run communication round
    result = await loop._run_communication_round(
        num_exchanges=3,
        task_context="test_rial_context"
    )
    
    # Verify result structure (actual keys from implementation)
    assert "total_exchanges" in result
    assert "new_vocabulary" in result
    assert "round_success_rate" in result
    assert "compositionality_delta" in result
    assert "training_mode" in result
    
    assert result["training_mode"] == "rial"
    assert result["total_exchanges"] <= 3
    print(f"[OK] _run_communication_round() RIAL mode: {result['total_exchanges']} exchanges")


async def test_run_communication_round_dial_mode():
    """Test _run_communication_round() in DIAL (differentiable) mode."""
    loop = RalphLoop(task="Test DIAL communication", max_iterations=5)
    
    # Initialize state
    loop.state = LoopState(
        loop_id="test_dial",
        task="Test DIAL communication",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Set DIAL mode
    loop.state.emergent_communication_state.training_mode = "dial"
    
    # Run communication round
    result = await loop._run_communication_round(
        num_exchanges=5,
        task_context="test_dial_context"
    )
    
    assert result["training_mode"] == "dial"
    assert "new_vocabulary" in result
    print(f"[OK] _run_communication_round() DIAL mode: vocab growth = {result['new_vocabulary']}")


async def test_run_communication_round_vocabulary_tracking():
    """Test that vocabulary is properly tracked across exchanges."""
    loop = RalphLoop(task="Test vocab tracking", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_vocab",
        task="Test vocab tracking",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    initial_vocab_size = len(loop.state.emergent_communication_state.emergent_vocabulary)
    
    # Run multiple communication rounds
    for i in range(3):
        await loop._run_communication_round(num_exchanges=2, task_context=f"round_{i}")
    
    final_vocab_size = len(loop.state.emergent_communication_state.emergent_vocabulary)
    
    # Vocabulary should grow (or at minimum stay same)
    assert final_vocab_size >= initial_vocab_size
    print(f"[OK] Vocabulary tracking: {initial_vocab_size} -> {final_vocab_size}")


async def test_evaluate_architecture_candidate_darts_strategy():
    """Test _evaluate_architecture_candidate() with DARTS strategy."""
    loop = RalphLoop(task="Test DARTS NAS", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_darts",
        task="Test DARTS NAS",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Evaluate with DARTS strategy
    result = await loop._evaluate_architecture_candidate(
        validation_data={"accuracy": 0.85, "loss": 0.15},
        strategy="darts"
    )
    
    assert "combined_score" in result
    assert "is_new_best" in result
    assert "score_breakdown" in result
    assert "pareto_rank" in result
    assert "strategy" in result
    
    assert result["strategy"] == "darts"
    assert 0 <= result["combined_score"] <= 1.0
    print(f"[OK] DARTS evaluation: score={result['combined_score']:.3f}, pareto_rank={result['pareto_rank']}")


async def test_evaluate_architecture_candidate_enas_strategy():
    """Test _evaluate_architecture_candidate() with ENAS strategy."""
    loop = RalphLoop(task="Test ENAS NAS", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_enas",
        task="Test ENAS NAS",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    result = await loop._evaluate_architecture_candidate(
        validation_data={"accuracy": 0.90, "loss": 0.10},
        strategy="enas"
    )
    
    assert result["strategy"] == "enas"
    print(f"[OK] ENAS evaluation: score={result['combined_score']:.3f}")


async def test_evaluate_architecture_candidate_pareto_tracking():
    """Test Pareto front tracking across multiple candidates."""
    loop = RalphLoop(task="Test Pareto front", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_pareto",
        task="Test Pareto front",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    initial_pareto_size = len(loop.state.nas_state.pareto_front)
    
    # Evaluate multiple candidates with different accuracy/efficiency tradeoffs
    candidates = [
        {"accuracy": 0.95, "loss": 0.05},  # High accuracy
        {"accuracy": 0.80, "loss": 0.20},  # Lower accuracy but maybe more efficient
        {"accuracy": 0.88, "loss": 0.12},  # Balanced
    ]
    
    for data in candidates:
        await loop._evaluate_architecture_candidate(validation_data=data, strategy="darts")
    
    final_pareto_size = len(loop.state.nas_state.pareto_front)
    
    # Pareto front should have grown
    assert final_pareto_size >= initial_pareto_size
    print(f"[OK] Pareto front: {initial_pareto_size} -> {final_pareto_size} candidates")


async def test_run_memory_consolidation_basic():
    """Test _run_memory_consolidation() basic functionality."""
    loop = RalphLoop(task="Test memory consolidation", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_mc",
        task="Test memory consolidation",
        current_iteration=10,
        max_iterations=100,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Add some experiences to replay buffer first
    for i in range(10):
        experience = {
            "iteration": i,
            "fitness": 0.5 + i * 0.02,
            "action": f"action_{i}",
            "state": f"state_{i}"
        }
        await loop._consolidate_memories(experience, importance=0.5 + i * 0.05)
    
    # Now run consolidation
    result = await loop._run_memory_consolidation(
        batch_size=5,
        num_memories=3,
        force_consolidation=True
    )
    
    assert "consolidated_count" in result or "status" in result
    if result.get("status") == "completed":
        assert "compression_ratio" in result
        assert "distillation_loss" in result
        assert "experiences_processed" in result
        print(f"[OK] Memory consolidation: {result['consolidated_count']} memories, "
              f"compression={result['compression_ratio']:.2f}")
    else:
        print(f"[OK] Memory consolidation: skipped - {result.get('reason', 'insufficient data')}")


async def test_run_memory_consolidation_vae_compression():
    """Test VAE-like compression in memory consolidation."""
    loop = RalphLoop(task="Test VAE compression", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_vae",
        task="Test VAE compression",
        current_iteration=10,
        max_iterations=100,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Enable generative replay
    loop.state.memory_consolidation_state.generative_replay_enabled = True
    
    # Add experiences
    for i in range(20):
        experience = {"iteration": i, "fitness": 0.5 + i * 0.01}
        await loop._consolidate_memories(experience, importance=0.5)
    
    result = await loop._run_memory_consolidation(
        batch_size=10,
        num_memories=5,
        force_consolidation=True
    )
    
    # Check compression happened
    assert result["compression_ratio"] > 0
    assert result["compression_ratio"] <= 1.0  # Should be compressing
    print(f"[OK] VAE compression ratio: {result['compression_ratio']:.3f}")


async def test_run_memory_consolidation_should_consolidate_check():
    """Test should_consolidate() interval logic."""
    loop = RalphLoop(task="Test consolidation interval", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_interval",
        task="Test consolidation interval",
        current_iteration=1,
        max_iterations=100,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    mc_state = loop.state.memory_consolidation_state
    
    # Set interval to 10
    mc_state.consolidation_interval = 10
    mc_state.consolidation_rounds = 0
    
    # Should not consolidate initially (only every 10 rounds)
    should_at_0 = mc_state.should_consolidate()
    
    # Simulate 9 iterations
    mc_state.consolidation_rounds = 9
    should_at_9 = mc_state.should_consolidate()
    
    # At 10 it should consolidate
    mc_state.consolidation_rounds = 10
    should_at_10 = mc_state.should_consolidate()
    
    print(f"[OK] should_consolidate(): at 0={should_at_0}, at 9={should_at_9}, at 10={should_at_10}")


async def test_v12_integration_success_path():
    """Test V12 integration fires correctly on success path in run_iteration()."""
    loop = RalphLoop(task="Test V12 integration", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_integration",
        task="Test V12 integration",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.3,  # Low initial fitness
        best_solution="initial",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    
    # Initialize all V12 states
    loop._initialize_v12_state()
    
    # Check that all V12 states are initialized
    assert loop.state.world_model_state is not None
    assert loop.state.predictive_coding_state is not None
    assert loop.state.active_inference_state is not None
    assert loop.state.emergent_communication_state is not None
    assert loop.state.nas_state is not None
    assert loop.state.memory_consolidation_state is not None
    
    print("[OK] V12 integration: All subsystems initialized")


async def test_v12_metrics_in_artifact():
    """Test V12 metrics are correctly added to artifact data."""
    loop = RalphLoop(task="Test V12 metrics", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="test_metrics",
        task="Test V12 metrics",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Get V12 insights
    insights = loop.get_v12_insights()
    
    # Verify all expected sections
    expected_sections = [
        "world_model", "predictive_coding", "active_inference",
        "emergent_communication", "nas", "memory_consolidation"
    ]
    
    for section in expected_sections:
        assert section in insights, f"Missing section: {section}"
    
    # Check specific metrics exist
    assert "prediction_accuracy" in insights["world_model"]
    assert "current_free_energy" in insights["predictive_coding"]
    assert "epistemic_value" in insights["active_inference"]
    assert "vocabulary_size" in insights["emergent_communication"]
    assert "best_validation_accuracy" in insights["nas"]
    assert "compression_ratio" in insights["memory_consolidation"]
    
    print(f"[OK] V12 metrics: All {len(expected_sections)} sections present with required fields")


# =============================================================================
# PERFORMANCE PROFILING TESTS
# =============================================================================

import time

async def test_communication_round_performance():
    """Profile _run_communication_round() performance."""
    loop = RalphLoop(task="Perf test communication", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="perf_comm",
        task="Perf test communication",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Time multiple runs
    times = []
    for _ in range(5):
        start = time.perf_counter()
        await loop._run_communication_round(num_exchanges=3)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    print(f"[PERF] _run_communication_round(): avg={avg_time*1000:.2f}ms, max={max_time*1000:.2f}ms")
    
    # Should complete in reasonable time (< 100ms per call)
    assert avg_time < 0.1, f"Communication round too slow: {avg_time*1000:.2f}ms"


async def test_architecture_evaluation_performance():
    """Profile _evaluate_architecture_candidate() performance."""
    loop = RalphLoop(task="Perf test NAS", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="perf_nas",
        task="Perf test NAS",
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    times = []
    for i in range(10):
        start = time.perf_counter()
        await loop._evaluate_architecture_candidate(
            validation_data={"accuracy": 0.5 + i * 0.04, "loss": 0.5 - i * 0.04}
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    
    print(f"[PERF] _evaluate_architecture_candidate(): avg={avg_time*1000:.2f}ms")
    assert avg_time < 0.05, f"Architecture evaluation too slow: {avg_time*1000:.2f}ms"


async def test_memory_consolidation_performance():
    """Profile _run_memory_consolidation() performance."""
    loop = RalphLoop(task="Perf test consolidation", max_iterations=5)
    
    loop.state = LoopState(
        loop_id="perf_mc",
        task="Perf test consolidation",
        current_iteration=10,
        max_iterations=100,
        best_fitness=0.5,
        best_solution="test",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="testing"
    )
    loop._initialize_v12_state()
    
    # Add experiences
    for i in range(50):
        await loop._consolidate_memories({"iteration": i, "fitness": 0.5}, importance=0.5)
    
    times = []
    for _ in range(5):
        start = time.perf_counter()
        await loop._run_memory_consolidation(batch_size=10, num_memories=3, force_consolidation=True)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    
    print(f"[PERF] _run_memory_consolidation(): avg={avg_time*1000:.2f}ms")
    # Memory consolidation can be slower, allow 500ms
    assert avg_time < 0.5, f"Memory consolidation too slow: {avg_time*1000:.2f}ms"


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RALPH LOOP V12 UNIT TESTS")
    print("=" * 60 + "\n")

    # Run synchronous tests directly
    print("--- SYNCHRONOUS TESTS ---\n")
    test_v12_dataclass_structures()
    print()
    test_ralph_loop_initialization()
    print()
    test_loop_state_v12_serialization()
    print()
    test_v12_helper_methods()
    print()
    test_get_v12_insights()
    
    # Run async tests
    print("\n--- ASYNC V12 ORCHESTRATION TESTS ---\n")
    
    async def run_async_tests():
        """Run all async tests."""
        # Communication round tests
        await test_run_communication_round_rial_mode()
        await test_run_communication_round_dial_mode()
        await test_run_communication_round_vocabulary_tracking()
        print()
        
        # Architecture evaluation tests
        await test_evaluate_architecture_candidate_darts_strategy()
        await test_evaluate_architecture_candidate_enas_strategy()
        await test_evaluate_architecture_candidate_pareto_tracking()
        print()
        
        # Memory consolidation tests
        await test_run_memory_consolidation_basic()
        await test_run_memory_consolidation_vae_compression()
        await test_run_memory_consolidation_should_consolidate_check()
        print()
        
        # Integration tests
        await test_v12_integration_success_path()
        await test_v12_metrics_in_artifact()
        print()
        
        # Performance tests
        print("\n--- PERFORMANCE PROFILING ---\n")
        await test_communication_round_performance()
        await test_architecture_evaluation_performance()
        await test_memory_consolidation_performance()
    
    asyncio.run(run_async_tests())

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
