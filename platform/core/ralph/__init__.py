"""
Ralph Loop Module - V36 Architecture with Production Monitoring

Self-improvement loop based on the Ralph Claude Code pattern.
This module provides the main RalphLoop class and supporting state classes.

V36 Modular Structure:
```
ralph/
├── __init__.py              # Public API (this file)
├── state.py                 # Core state classes (IterationResult, LoopState)
├── production_monitoring.py # Production observability & drift detection
├── strategies/
│   ├── __init__.py          # Strategy exports
│   ├── reflexion.py         # V4: Reflection, DebatePosition, ProceduralSkill
│   ├── consistency.py       # V5: ConsistencyPath, VerificationStep, OODAState
│   ├── scheduling.py        # V6: StrategyArm, ConvergenceState, Thompson
│   ├── curriculum.py        # V7: CurriculumState, ExperienceReplay, STOP
│   ├── mcts.py              # V8: MCTSNode, MCTSState, SelfPlayState
│   ├── scpo_rlvr.py         # V9: ScPO, RLVR, Multi-Agent Coordination
│   ├── prm_cai.py           # V10: PRM, Constitutional AI, Test-time compute
│   └── speculative.py       # V11: Speculative decoding, Chain-of-Draft, RAG
└── metrics.py               # Fitness tracking (TODO)
```

Import Patterns:
    # Option 1: Import from package (backwards compatible)
    from core.ralph import RalphLoop, LoopState, IterationResult

    # Option 2: Import from specific modules (V36 style)
    from core.ralph.strategies import StrategyArm, MCTSState
    from core.ralph.state import LoopState

    # Option 3: Production monitoring
    from core.ralph.production_monitoring import (
        RalphProductionMonitor,
        DriftDetector,
        create_production_monitor
    )

Version History:
- V4: Reflexion pattern, Multi-agent debate
- V5: Self-consistency, Chain-of-Verification
- V6: Thompson sampling, Adaptive scheduling
- V7: Curriculum learning, Experience replay, STOP
- V8: MCTS, Multi-agent self-play
- V9: ScPO, RLVR, Multi-agent coordination
- V10: PRM, Constitutional AI, Test-time compute
- V11: Speculative decoding, Chain-of-Draft, Adaptive RAG
- V36: Production monitoring with chi-squared drift detection
"""

# V36 Modular imports - All strategy classes from modular files
try:
    from .strategies import (
        # V4 Reflexion
        Reflection,
        DebatePosition,
        ProceduralSkill,
        # V5 Consistency
        ConsistencyPath,
        VerificationStep,
        OODAState,
        RISEAttempt,
        # V6 Scheduling
        StrategyArm,
        ConvergenceState,
        IterationMomentum,
        MetaIterationState,
        # V7 Curriculum
        CurriculumState,
        ExperienceReplay,
        STOPState,
        HierarchicalLoopState,
        # V8 MCTS
        MCTSNode,
        MCTSState,
        SelfPlayAgent,
        SelfPlayState,
        StrategistState,
        # V9 ScPO/RLVR
        ConsistencyPreference,
        ScPOState,
        VerifiableReward,
        RLVRState,
        AgentMessage,
        AgentCoordinationChannel,
        MultiAgentCoordinationState,
        # V10 PRM/CAI
        ProcessRewardStep,
        PRMState,
        ConstitutionalPrinciple,
        ConstitutionalCritique,
        CAIState,
        ThinkingBudget,
        TestTimeComputeState,
        # V11 Speculative
        SpeculativeHypothesis,
        SpeculativeDecodingState,
        DraftStep,
        ChainOfDraftState,
        RetrievalDecision,
        AdaptiveRAGState,
        RewardHackingSignal,
        RewardHackingDetectorState,
        MetaJudgment,
        MetaRewardState,
        CausalIntervention,
        ImprovementAttributionState,
    )
    V36_MODULES_AVAILABLE = True
except ImportError:
    V36_MODULES_AVAILABLE = False

# Core classes: RalphLoop from original ralph_loop.py (main loop logic)
# LoopState and IterationResult can come from either modular or original
try:
    from ..ralph_loop import RalphLoop
    RALPH_AVAILABLE = True

    # If V36 modules not available, also import all classes from original
    if not V36_MODULES_AVAILABLE:
        from ..ralph_loop import (
            LoopState,
            IterationResult,
            # V4 Reflexion
            Reflection,
            DebatePosition,
            ProceduralSkill,
            # V5 Consistency
            ConsistencyPath,
            VerificationStep,
            OODAState,
            RISEAttempt,
            # V6 Scheduling
            StrategyArm,
            ConvergenceState,
            IterationMomentum,
            MetaIterationState,
            # V7 Curriculum
            CurriculumState,
            ExperienceReplay,
            STOPState,
            HierarchicalLoopState,
            # V8 MCTS
            MCTSNode,
            MCTSState,
            SelfPlayAgent,
            SelfPlayState,
            StrategistState,
            # V9 ScPO/RLVR
            ConsistencyPreference,
            ScPOState,
            VerifiableReward,
            RLVRState,
            AgentMessage,
            AgentCoordinationChannel,
            MultiAgentCoordinationState,
            # V10 PRM/CAI
            ProcessRewardStep,
            PRMState,
            ConstitutionalPrinciple,
            ConstitutionalCritique,
            CAIState,
            ThinkingBudget,
            TestTimeComputeState,
            # V11 Speculative
            SpeculativeHypothesis,
            SpeculativeDecodingState,
            DraftStep,
            ChainOfDraftState,
            RetrievalDecision,
            AdaptiveRAGState,
            RewardHackingSignal,
            RewardHackingDetectorState,
            MetaJudgment,
            MetaRewardState,
            CausalIntervention,
            ImprovementAttributionState,
        )
    else:
        # V36 modules available, import LoopState/IterationResult from state.py
        try:
            from .state import LoopState, IterationResult
        except ImportError:
            # Fallback to ralph_loop if state.py has issues
            from ..ralph_loop import LoopState, IterationResult

except ImportError as e:
    RALPH_AVAILABLE = False
    # Provide stub for when ralph_loop.py is not available
    RalphLoop = None
    if not V36_MODULES_AVAILABLE:
        LoopState = None
        IterationResult = None

# Production monitoring imports
try:
    from .production_monitoring import (
        ProductionConfig,
        ObservabilityProvider,
        DriftDetector,
        DriftSignal,
        ObservabilityBackend,
        OpikBackend,
        LangfuseBackend,
        PhoenixBackend,
        ConsoleBackend,
        RalphProductionMonitor,
        create_production_monitor,
    )
    PRODUCTION_MONITORING_AVAILABLE = True
except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False
    # Stubs
    ProductionConfig = None
    ObservabilityProvider = None
    DriftDetector = None
    DriftSignal = None
    RalphProductionMonitor = None
    create_production_monitor = None

# V37 Production Enhancements (new)
try:
    from .production import (
        # Logging
        StructuredLogFormatter,
        RalphLogger,
        configure_production_logging,
        correlation_context,
        get_correlation_id,
        set_correlation_id,
        # Metrics
        RalphMetrics,
        MetricType,
        MetricValue,
        get_metrics,
        # Rate Limiting
        RateLimitConfig,
        RateLimiter,
        TokenBucket,
        get_rate_limiter,
        rate_limited,
        # Shutdown
        ShutdownHandler,
        get_shutdown_handler,
        # Checkpointing
        CheckpointManager,
        CheckpointMetadata,
        # V11 Production Features
        HypothesisTracker,
        SpeculativeDecodingManager,
        ChainOfDraftManager,
        AdaptiveRAGManager,
        RewardHackingDetector as ProductionRewardHackingDetector,
        RewardHackingSignal as ProductionRewardHackingSignal,
        # Configuration
        ProductionConfig as V37ProductionConfig,
        initialize_production,
    )
    V37_PRODUCTION_AVAILABLE = True
except ImportError:
    V37_PRODUCTION_AVAILABLE = False

# V37 Production Loop
try:
    from .production_loop import (
        ProductionRalphLoop,
        ProductionIterationResult,
        create_production_loop,
    )
    PRODUCTION_LOOP_AVAILABLE = True
except ImportError:
    PRODUCTION_LOOP_AVAILABLE = False
    ProductionRalphLoop = None
    ProductionIterationResult = None
    create_production_loop = None

__all__ = [
    # Availability flags
    "RALPH_AVAILABLE",
    "V36_MODULES_AVAILABLE",
    "V37_PRODUCTION_AVAILABLE",
    "PRODUCTION_LOOP_AVAILABLE",
    # Core classes
    "RalphLoop",
    "LoopState",
    "IterationResult",
    # V4 Reflexion
    "Reflection",
    "DebatePosition",
    "ProceduralSkill",
    # V5 Consistency
    "ConsistencyPath",
    "VerificationStep",
    "OODAState",
    "RISEAttempt",
    # V6 Scheduling
    "StrategyArm",
    "ConvergenceState",
    "IterationMomentum",
    "MetaIterationState",
    # V7 Curriculum
    "CurriculumState",
    "ExperienceReplay",
    "STOPState",
    "HierarchicalLoopState",
    # V8 MCTS
    "MCTSNode",
    "MCTSState",
    "SelfPlayAgent",
    "SelfPlayState",
    "StrategistState",
    # V9 ScPO/RLVR
    "ConsistencyPreference",
    "ScPOState",
    "VerifiableReward",
    "RLVRState",
    "AgentMessage",
    "AgentCoordinationChannel",
    "MultiAgentCoordinationState",
    # V10 PRM/CAI
    "ProcessRewardStep",
    "PRMState",
    "ConstitutionalPrinciple",
    "ConstitutionalCritique",
    "CAIState",
    "ThinkingBudget",
    "TestTimeComputeState",
    # V11 Speculative
    "SpeculativeHypothesis",
    "SpeculativeDecodingState",
    "DraftStep",
    "ChainOfDraftState",
    "RetrievalDecision",
    "AdaptiveRAGState",
    "RewardHackingSignal",
    "RewardHackingDetectorState",
    "MetaJudgment",
    "MetaRewardState",
    "CausalIntervention",
    "ImprovementAttributionState",
    # Production Monitoring
    "PRODUCTION_MONITORING_AVAILABLE",
    "ProductionConfig",
    "ObservabilityProvider",
    "DriftDetector",
    "DriftSignal",
    "ObservabilityBackend",
    "OpikBackend",
    "LangfuseBackend",
    "PhoenixBackend",
    "ConsoleBackend",
    "RalphProductionMonitor",
    "create_production_monitor",
    # V37 Production Enhancements
    "StructuredLogFormatter",
    "RalphLogger",
    "configure_production_logging",
    "correlation_context",
    "get_correlation_id",
    "set_correlation_id",
    "RalphMetrics",
    "MetricType",
    "MetricValue",
    "get_metrics",
    "RateLimitConfig",
    "RateLimiter",
    "TokenBucket",
    "get_rate_limiter",
    "rate_limited",
    "ShutdownHandler",
    "get_shutdown_handler",
    "CheckpointManager",
    "CheckpointMetadata",
    "HypothesisTracker",
    "SpeculativeDecodingManager",
    "ChainOfDraftManager",
    "AdaptiveRAGManager",
    "ProductionRewardHackingDetector",
    "ProductionRewardHackingSignal",
    "V37ProductionConfig",
    "initialize_production",
    # V37 Production Loop
    "ProductionRalphLoop",
    "ProductionIterationResult",
    "create_production_loop",
]
