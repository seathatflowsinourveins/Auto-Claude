"""
Ralph Loop Strategies - V36 Modular Architecture

This module contains all strategy implementations extracted from ralph_loop.py.
Organized by version feature set (V4-V11).
"""

from .reflexion import (
    Reflection,
    DebatePosition,
    ProceduralSkill,
)
from .consistency import (
    ConsistencyPath,
    VerificationStep,
    OODAState,
    RISEAttempt,
)
from .scheduling import (
    StrategyArm,
    ConvergenceState,
    IterationMomentum,
    MetaIterationState,
)
from .curriculum import (
    CurriculumState,
    ExperienceReplay,
    STOPState,
    HierarchicalLoopState,
)
from .mcts import (
    MCTSNode,
    MCTSState,
    SelfPlayAgent,
    SelfPlayState,
    StrategistState,
)
from .scpo_rlvr import (
    ConsistencyPreference,
    ScPOState,
    VerifiableReward,
    RLVRState,
    AgentMessage,
    AgentCoordinationChannel,
    MultiAgentCoordinationState,
)
from .prm_cai import (
    ProcessRewardStep,
    PRMState,
    ConstitutionalPrinciple,
    ConstitutionalCritique,
    CAIState,
    ThinkingBudget,
    TestTimeComputeState,
)
from .speculative import (
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

__all__ = [
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
]
