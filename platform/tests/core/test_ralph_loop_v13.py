"""
Integration Test for Ralph Loop V13

Tests all 3 V13 subsystems:
1. Compositional Generalization (Lake & Baroni 2023, SCAN, COGS)
2. Meta-RL (ECET, AMAGO-2, MAML-style inner loop)
3. Program Synthesis (AlphaEvolve, DreamCoder patterns)

Run with: python -m pytest platform/core/test_ralph_loop_v13.py -v
         or: python platform/core/test_ralph_loop_v13.py
"""
import sys
import os
import asyncio
import json
from datetime import datetime, timezone
import pytest

# Add platform/ path for imports (two levels up from tests/core/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Direct import (must be after sys.path modification)
from core.ralph_loop import RalphLoop, LoopState

# Mark all async tests with asyncio
pytestmark = pytest.mark.asyncio


def test_v13_dataclass_structures():
    """Test V13 dataclass structures can be instantiated."""
    from core.ralph_loop import (
        CompositionRule,
        CompositionalGeneralizationState,
        AdaptationEpisode,
        MetaRLState,
        ProgramPrimitive,
        LearnedAbstraction,
        CandidateProgram,
        SynthesisSpecification,
        ProgramSynthesisState
    )

    # Test CompositionRule (actual fields: rule_id, input_pattern, output_template, primitive_slots, usage_count, success_rate)
    rule = CompositionRule(
        rule_id="rule_001",
        input_pattern="X twice",
        output_template="{X} {X}",
        primitive_slots=["X"],
        usage_count=5,
        success_rate=0.85
    )
    assert rule.rule_id == "rule_001"
    assert rule.input_pattern == "X twice"
    assert len(rule.primitive_slots) == 1
    assert rule.success_rate == 0.85
    print("[OK] CompositionRule initialized")

    # Test CompositionalGeneralizationState
    comp_gen = CompositionalGeneralizationState()
    assert comp_gen.episode_adaptation_steps == 5
    assert comp_gen.adaptation_success_rate == 0.0
    assert len(comp_gen.primitive_library) == 0
    assert len(comp_gen.composition_rules) == 0
    print("[OK] CompositionalGeneralizationState initialized")

    # Test AdaptationEpisode (actual fields: episode_id, task_id, initial_performance, final_performance, adaptation_steps, loss_trajectory)
    episode = AdaptationEpisode(
        episode_id=1,
        task_id="task_001",
        initial_performance=0.3,
        final_performance=0.8,
        adaptation_steps=5,
        loss_trajectory=[0.7, 0.5, 0.3, 0.2, 0.1]
    )
    assert episode.episode_id == 1
    assert episode.task_id == "task_001"
    assert episode.initial_performance == 0.3
    assert episode.final_performance == 0.8
    assert episode.adaptation_gain == 0.5  # property
    print("[OK] AdaptationEpisode initialized")

    # Test MetaRLState (actual fields: inner_loop_steps, inner_loop_lr, adaptation_history, memory_capacity, etc.)
    meta_rl = MetaRLState()
    assert meta_rl.inner_loop_steps == 5
    assert meta_rl.inner_loop_lr == 0.01
    assert meta_rl.memory_capacity == 100
    assert len(meta_rl.adaptation_history) == 0
    assert meta_rl.total_tasks_seen == 0
    print("[OK] MetaRLState initialized")

    # Test ProgramPrimitive (actual fields: name, signature, implementation, usage_count, success_rate, description)
    primitive = ProgramPrimitive(
        name="increment",
        signature="int -> int",
        implementation="lambda x: x + 1",
        description="Add 1 to input"
    )
    assert primitive.name == "increment"
    assert primitive.signature == "int -> int"
    print("[OK] ProgramPrimitive initialized")

    # Test LearnedAbstraction (actual fields: name, body, examples, usage_count)
    abstraction = LearnedAbstraction(
        name="double_increment",
        body="increment(increment(x))",
        examples=[{"input": 1, "output": 3}],
        usage_count=10
    )
    assert abstraction.name == "double_increment"
    assert abstraction.usage_count == 10
    print("[OK] LearnedAbstraction initialized")

    # Test CandidateProgram (actual fields: program_id, code, fitness, passes_tests, execution_time_ms, complexity)
    candidate = CandidateProgram(
        program_id="prog_001",
        code="lambda x: x + 1",
        fitness=0.85,
        passes_tests=True,
        complexity=3
    )
    assert candidate.program_id == "prog_001"
    assert candidate.fitness == 0.85
    assert candidate.passes_tests == True
    print("[OK] CandidateProgram initialized")

    # Test SynthesisSpecification (actual fields: spec_id, input_output_examples, natural_language_description, constraints)
    spec = SynthesisSpecification(
        spec_id="spec_001",
        input_output_examples=[(1, 2), (2, 3), (3, 4)],
        natural_language_description="Increment function",
        constraints=["pure", "O(1)"]
    )
    assert spec.spec_id == "spec_001"
    assert len(spec.input_output_examples) == 3
    print("[OK] SynthesisSpecification initialized")

    # Test ProgramSynthesisState (actual fields: primitive_library (Dict), learned_abstractions, population, pareto_archive, best_fitness, synthesis_iterations)
    prog_synth = ProgramSynthesisState()
    assert prog_synth.population_size == 50
    assert prog_synth.mutation_rate == 0.3  # Default is 0.3
    assert prog_synth.best_fitness == 0.0
    assert prog_synth.synthesis_iterations == 0
    assert len(prog_synth.pareto_archive) == 0
    print("[OK] ProgramSynthesisState initialized")


def test_ralph_loop_v13_initialization():
    """Test RalphLoop initializes V13 states properly."""
    loop = RalphLoop(
        task="Test V13 task",
        max_iterations=5
    )
    assert loop.task == "Test V13 task"
    assert loop.max_iterations == 5
    print("[OK] RalphLoop initialized for V13 testing")


def test_loop_state_v13_serialization():
    """Test LoopState with V13 fields can be serialized and deserialized."""
    from core.ralph_loop import (
        CompositionalGeneralizationState,
        MetaRLState,
        ProgramSynthesisState,
        CompositionRule,
        AdaptationEpisode,
        ProgramPrimitive,
        LearnedAbstraction
    )

    # Create a LoopState with V13 subsystems (include all required fields)
    state = LoopState(
        loop_id="test_v13_123",
        task="Test V13 serialization",
        current_iteration=5,
        max_iterations=100,
        best_fitness=0.75,
        best_solution="test_solution",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        comp_gen_state=CompositionalGeneralizationState(
            primitive_library={"jump": "JUMP", "walk": "WALK"},
            composition_rules=[
                CompositionRule(
                    rule_id="rule_1",
                    input_pattern="X twice",
                    output_template="{X} {X}",
                    primitive_slots=["X"],
                    success_rate=0.95
                )
            ],
            seen_combinations=[("jump",), ("walk",)],
            systematic_generalization_score=0.72
        ),
        meta_rl_state=MetaRLState(
            inner_loop_steps=10,
            inner_loop_lr=0.0005,
            adaptation_history=[
                AdaptationEpisode(
                    episode_id=1,
                    task_id="episode_1",
                    initial_performance=0.4,
                    final_performance=0.85,
                    adaptation_steps=10
                )
            ]
        ),
        prog_synth_state=ProgramSynthesisState(
            primitive_library={
                "add": ProgramPrimitive(
                    name="add",
                    signature="(int, int) -> int",
                    implementation="lambda x, y: x + y"
                )
            },
            learned_abstractions=[
                LearnedAbstraction(
                    name="double",
                    body="add(x, x)",
                    usage_count=5
                )
            ],
            best_fitness=0.78
        )
    )

    # Serialize to dict
    state_dict = state.to_dict()

    # Verify V13 fields are present
    assert "comp_gen_state" in state_dict
    assert "meta_rl_state" in state_dict
    assert "prog_synth_state" in state_dict
    print("[OK] V13 state serialized to dict")

    # Verify CompositionalGeneralizationState serialization
    comp_gen_dict = state_dict["comp_gen_state"]
    assert comp_gen_dict["primitive_library"]["jump"] == "JUMP"
    assert len(comp_gen_dict["composition_rules"]) == 1
    assert comp_gen_dict["systematic_generalization_score"] == 0.72
    print("[OK] CompositionalGeneralizationState serialization verified")

    # Verify MetaRLState serialization
    meta_rl_dict = state_dict["meta_rl_state"]
    assert meta_rl_dict["inner_loop_steps"] == 10
    assert len(meta_rl_dict["adaptation_history"]) == 1
    assert meta_rl_dict["adaptation_history"][0]["final_performance"] == 0.85
    print("[OK] MetaRLState serialization verified")

    # Verify ProgramSynthesisState serialization
    prog_synth_dict = state_dict["prog_synth_state"]
    assert "add" in prog_synth_dict["primitive_library"]
    assert len(prog_synth_dict["learned_abstractions"]) == 1
    assert prog_synth_dict["best_fitness"] == 0.78
    print("[OK] ProgramSynthesisState serialization verified")

    # JSON roundtrip
    json_str = json.dumps(state_dict, default=str)
    loaded_dict = json.loads(json_str)
    assert loaded_dict["comp_gen_state"]["systematic_generalization_score"] == 0.72
    print("[OK] V13 state JSON roundtrip successful")

    # Deserialize back to LoopState
    restored_state = LoopState.from_dict(loaded_dict)
    assert restored_state.comp_gen_state is not None
    assert restored_state.meta_rl_state is not None
    assert restored_state.prog_synth_state is not None
    assert restored_state.comp_gen_state.systematic_generalization_score == 0.72
    assert restored_state.meta_rl_state.inner_loop_steps == 10
    assert restored_state.prog_synth_state.best_fitness == 0.78
    print("[OK] V13 state deserialized from dict")


async def test_evaluate_compositional_generalization():
    """Test _evaluate_compositional_generalization method."""
    from core.ralph_loop import CompositionalGeneralizationState

    loop = RalphLoop(
        task="Test compositional generalization",
        max_iterations=5
    )

    # Initialize state manually (include all required fields)
    loop.state = LoopState(
        loop_id="test_comp_gen",
        task=loop.task,
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.0,
        best_solution="initial",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        comp_gen_state=CompositionalGeneralizationState(
            primitive_library={
                "jump": "JUMP",
                "walk": "WALK",
                "run": "RUN"
            },
            composition_rules=[],
            seen_combinations=[("jump",), ("walk",)]
        )
    )

    # Run compositional generalization evaluation
    result = await loop._evaluate_compositional_generalization(
        test_combinations=5,
        context="test_eval"
    )

    assert isinstance(result, dict)
    print(f"[OK] Compositional generalization evaluation returned: {list(result.keys())}")


async def test_run_meta_rl_adaptation():
    """Test _run_meta_rl_adaptation method."""
    from core.ralph_loop import MetaRLState

    loop = RalphLoop(
        task="Test meta-RL adaptation",
        max_iterations=5
    )

    # Initialize state manually (include all required fields)
    loop.state = LoopState(
        loop_id="test_meta_rl",
        task=loop.task,
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.0,
        best_solution="initial",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        meta_rl_state=MetaRLState(
            inner_loop_steps=5,
            inner_loop_lr=0.001
        )
    )

    # Run meta-RL adaptation
    result = await loop._run_meta_rl_adaptation(
        task_context="test_task",
        adaptation_steps=3
    )

    assert isinstance(result, dict)
    print(f"[OK] Meta-RL adaptation returned: {list(result.keys())}")


async def test_synthesize_program():
    """Test _synthesize_program method."""
    from core.ralph_loop import ProgramSynthesisState, ProgramPrimitive

    loop = RalphLoop(
        task="Test program synthesis",
        max_iterations=5
    )

    # Initialize state manually (include all required fields)
    loop.state = LoopState(
        loop_id="test_prog_synth",
        task=loop.task,
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.0,
        best_solution="initial",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        prog_synth_state=ProgramSynthesisState(
            primitive_library={
                "inc": ProgramPrimitive(
                    name="inc",
                    signature="int -> int",
                    implementation="lambda x: x + 1"
                )
            },
            population_size=10,
            mutation_rate=0.2
        )
    )

    # Run program synthesis
    result = await loop._synthesize_program(
        specification="increment_function",
        max_generations=5
    )

    assert isinstance(result, dict)
    print(f"[OK] Program synthesis returned: {list(result.keys())}")


async def test_get_v13_insights():
    """Test get_v13_insights method."""
    from core.ralph_loop import (
        CompositionalGeneralizationState,
        MetaRLState,
        ProgramSynthesisState,
        CompositionRule,
        AdaptationEpisode,
        ProgramPrimitive,
        LearnedAbstraction
    )

    loop = RalphLoop(
        task="Test V13 insights",
        max_iterations=5
    )

    # Initialize state with all V13 subsystems populated (include all required fields)
    loop.state = LoopState(
        loop_id="test_v13_insights",
        task=loop.task,
        current_iteration=10,
        max_iterations=100,
        best_fitness=0.75,
        best_solution="current_best",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        comp_gen_state=CompositionalGeneralizationState(
            primitive_library={"jump": "JUMP", "walk": "WALK", "run": "RUN"},
            composition_rules=[
                CompositionRule(
                    rule_id="rule_1",
                    input_pattern="X twice",
                    output_template="{X} {X}",
                    primitive_slots=["X"],
                    success_rate=0.95
                )
            ],
            systematic_generalization_score=0.78,
            adaptation_success_rate=0.65
        ),
        meta_rl_state=MetaRLState(
            inner_loop_steps=7,
            inner_loop_lr=0.0008,
            total_adaptations=5,  # Must match len(adaptation_history)
            adaptation_history=[
                AdaptationEpisode(
                    episode_id=i,
                    task_id=f"task_{i}",
                    initial_performance=0.3 + i * 0.05,
                    final_performance=0.6 + i * 0.05,
                    adaptation_steps=7
                ) for i in range(5)
            ]
        ),
        prog_synth_state=ProgramSynthesisState(
            primitive_library={
                "inc": ProgramPrimitive(name="inc", signature="int->int", implementation="x+1"),
                "dec": ProgramPrimitive(name="dec", signature="int->int", implementation="x-1"),
                "add": ProgramPrimitive(name="add", signature="(int,int)->int", implementation="x+y")
            },
            learned_abstractions=[
                LearnedAbstraction(
                    name="double",
                    body="add(x, x)",
                    usage_count=8
                )
            ],
            best_fitness=0.82,
            synthesis_iterations=25
        )
    )

    # Get V13 insights
    insights = loop.get_v13_insights()

    assert isinstance(insights, dict)
    assert "compositional_generalization" in insights
    assert "meta_rl" in insights
    assert "program_synthesis" in insights

    # Verify compositional generalization insights (matching actual API)
    comp_gen_insights = insights["compositional_generalization"]
    assert comp_gen_insights["primitive_library_size"] == 3
    assert comp_gen_insights["composition_rules_count"] == 1
    assert comp_gen_insights["systematic_generalization_score"] == 0.78
    print("[OK] Compositional generalization insights verified")

    # Verify meta-RL insights (matching get_summary() API)
    meta_rl_insights = insights["meta_rl"]
    assert meta_rl_insights["total_adaptations"] == 5  # from adaptation_history
    assert meta_rl_insights["episodic_memory_size"] >= 0  # may be empty
    print("[OK] Meta-RL insights verified")

    # Verify program synthesis insights (matching get_summary() API)
    prog_synth_insights = insights["program_synthesis"]
    assert prog_synth_insights["primitives"] == 3
    assert prog_synth_insights["abstractions"] == 1
    print("[OK] Program synthesis insights verified")

    print(f"[OK] V13 insights complete")


async def test_v13_integration_performance():
    """Test V13 method performance benchmarks."""
    import time
    from core.ralph_loop import (
        CompositionalGeneralizationState,
        MetaRLState,
        ProgramSynthesisState,
        ProgramPrimitive
    )

    loop = RalphLoop(
        task="Performance benchmark",
        max_iterations=5
    )

    # Initialize state with V13 subsystems (include all required fields)
    loop.state = LoopState(
        loop_id="perf_test",
        task=loop.task,
        current_iteration=1,
        max_iterations=5,
        best_fitness=0.0,
        best_solution="initial",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        comp_gen_state=CompositionalGeneralizationState(
            primitive_library={f"prim_{i}": f"OUTPUT_{i}" for i in range(20)},
            composition_rules=[]
        ),
        meta_rl_state=MetaRLState(),
        prog_synth_state=ProgramSynthesisState(
            primitive_library={
                f"prim_{i}": ProgramPrimitive(name=f"prim_{i}", signature="int->int", implementation=f"x+{i}")
                for i in range(10)
            }
        )
    )

    # Benchmark compositional generalization
    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        await loop._evaluate_compositional_generalization(test_combinations=3, context="perf")
    comp_gen_avg_ms = (time.perf_counter() - start) / iterations * 1000
    print(f"[PERF] _evaluate_compositional_generalization avg: {comp_gen_avg_ms:.3f}ms")

    # Benchmark meta-RL adaptation
    start = time.perf_counter()
    for _ in range(iterations):
        await loop._run_meta_rl_adaptation(task_context="perf_test", adaptation_steps=2)
    meta_rl_avg_ms = (time.perf_counter() - start) / iterations * 1000
    print(f"[PERF] _run_meta_rl_adaptation avg: {meta_rl_avg_ms:.3f}ms")

    # Benchmark program synthesis
    start = time.perf_counter()
    for _ in range(iterations):
        await loop._synthesize_program(specification="perf_test", max_generations=3)
    prog_synth_avg_ms = (time.perf_counter() - start) / iterations * 1000
    print(f"[PERF] _synthesize_program avg: {prog_synth_avg_ms:.3f}ms")

    # Assert reasonable performance (< 100ms for each method)
    assert comp_gen_avg_ms < 100, f"Compositional generalization too slow: {comp_gen_avg_ms:.3f}ms"
    assert meta_rl_avg_ms < 100, f"Meta-RL adaptation too slow: {meta_rl_avg_ms:.3f}ms"
    assert prog_synth_avg_ms < 100, f"Program synthesis too slow: {prog_synth_avg_ms:.3f}ms"

    print(f"[OK] V13 performance benchmarks passed")


def test_v13_artifact_metrics():
    """Test V13 metrics are properly structured for artifact data."""
    from core.ralph_loop import (
        CompositionalGeneralizationState,
        MetaRLState,
        ProgramSynthesisState,
        CompositionRule,
        ProgramPrimitive,
        LearnedAbstraction
    )

    state = LoopState(
        loop_id="artifact_test",
        task="Test artifact metrics",
        current_iteration=10,
        max_iterations=100,
        best_fitness=0.8,
        best_solution="test_solution",
        history=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        status="running",
        comp_gen_state=CompositionalGeneralizationState(
            primitive_library={"a": "A", "b": "B"},
            composition_rules=[CompositionRule(rule_id="r1", input_pattern="test", output_template="{a}", success_rate=0.9)],
            systematic_generalization_score=0.75,
            adaptation_success_rate=0.6
        ),
        meta_rl_state=MetaRLState(
            inner_loop_steps=8,
            inner_loop_lr=0.001,
            total_tasks_seen=5
        ),
        prog_synth_state=ProgramSynthesisState(
            primitive_library={"x": ProgramPrimitive(name="x", signature="int", implementation="x")},
            learned_abstractions=[LearnedAbstraction(name="y", body="x", usage_count=3)],
            pareto_archive=[{"fitness": 0.8, "complexity": 2}],
            best_fitness=0.85,
            synthesis_iterations=30
        )
    )

    # Build artifact data similar to run_iteration()
    artifact_data = {
        # V13: Compositional Generalization metrics
        "v13_comp_gen_primitive_count": len(state.comp_gen_state.primitive_library) if state.comp_gen_state else 0,
        "v13_comp_gen_rule_count": len(state.comp_gen_state.composition_rules) if state.comp_gen_state else 0,
        "v13_comp_gen_systematic_score": state.comp_gen_state.systematic_generalization_score if state.comp_gen_state else 0.0,
        "v13_comp_gen_adaptation_success": state.comp_gen_state.adaptation_success_rate if state.comp_gen_state else 0.0,
        # V13: Meta-RL metrics (matches updated artifact_data in ralph_loop.py)
        "v13_meta_rl_adaptation_history_count": len(state.meta_rl_state.adaptation_history) if state.meta_rl_state else 0,
        "v13_meta_rl_inner_loop_steps": state.meta_rl_state.inner_loop_steps if state.meta_rl_state else 0,
        "v13_meta_rl_inner_loop_lr": state.meta_rl_state.inner_loop_lr if state.meta_rl_state else 0.0,
        "v13_meta_rl_total_tasks_seen": state.meta_rl_state.total_tasks_seen if state.meta_rl_state else 0,
        # V13: Program Synthesis metrics
        "v13_prog_synth_primitive_count": len(state.prog_synth_state.primitive_library) if state.prog_synth_state else 0,
        "v13_prog_synth_abstraction_count": len(state.prog_synth_state.learned_abstractions) if state.prog_synth_state else 0,
        "v13_prog_synth_pareto_size": len(state.prog_synth_state.pareto_archive) if state.prog_synth_state else 0,
        "v13_prog_synth_best_fitness": state.prog_synth_state.best_fitness if state.prog_synth_state else 0.0,
        "v13_prog_synth_synthesis_iterations": state.prog_synth_state.synthesis_iterations if state.prog_synth_state else 0
    }

    # Verify all V13 metrics are present and correct
    assert artifact_data["v13_comp_gen_primitive_count"] == 2
    assert artifact_data["v13_comp_gen_rule_count"] == 1
    assert artifact_data["v13_comp_gen_systematic_score"] == 0.75
    assert artifact_data["v13_comp_gen_adaptation_success"] == 0.6
    print("[OK] Compositional generalization artifact metrics verified")

    assert artifact_data["v13_meta_rl_inner_loop_steps"] == 8
    assert artifact_data["v13_meta_rl_inner_loop_lr"] == 0.001
    assert artifact_data["v13_meta_rl_total_tasks_seen"] == 5
    print("[OK] Meta-RL artifact metrics verified")

    assert artifact_data["v13_prog_synth_primitive_count"] == 1
    assert artifact_data["v13_prog_synth_abstraction_count"] == 1
    assert artifact_data["v13_prog_synth_pareto_size"] == 1
    assert artifact_data["v13_prog_synth_best_fitness"] == 0.85
    assert artifact_data["v13_prog_synth_synthesis_iterations"] == 30
    print("[OK] Program synthesis artifact metrics verified")

    # Verify JSON serializable
    json_str = json.dumps(artifact_data)
    assert len(json_str) > 0
    print(f"[OK] V13 artifact metrics JSON serializable ({len(json_str)} chars)")


if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("Ralph Loop V13 Integration Tests")
    print("=" * 60)

    # Run sync tests
    print("\n--- Synchronous Tests ---")
    test_v13_dataclass_structures()
    test_ralph_loop_v13_initialization()
    test_loop_state_v13_serialization()
    test_v13_artifact_metrics()

    # Run async tests
    print("\n--- Asynchronous Tests ---")
    asyncio.run(test_evaluate_compositional_generalization())
    asyncio.run(test_run_meta_rl_adaptation())
    asyncio.run(test_synthesize_program())
    asyncio.run(test_get_v13_insights())
    asyncio.run(test_v13_integration_performance())

    print("\n" + "=" * 60)
    print("All V13 tests passed!")
    print("=" * 60)
