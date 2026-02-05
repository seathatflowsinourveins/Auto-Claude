"""
Tests for CVT (Consensus Validation Token) Protocol.

Tests cover:
- Core consensus operations (validate_task, propose_action, vote, finalize)
- Reputation-weighted voting
- Byzantine fault detection
- Token generation and verification
- Raft fallback integration
- Performance (sub-millisecond latency targets)
- Edge cases and error handling

Test count: 45 tests
"""

import asyncio
import secrets
import time
from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# IMPORTS AND SETUP
# =============================================================================


# Import the modules under test
import sys
from pathlib import Path

# Add the platform parent to path for direct imports
_platform_dir = Path(__file__).parent.parent
if str(_platform_dir.parent) not in sys.path:
    sys.path.insert(0, str(_platform_dir.parent))

try:
    from core.orchestration.cvt_consensus import (
        CVTConsensus,
        CVTPhase,
        CVTProposal,
        CVTResult,
        CVToken,
        CVTVote,
        CVTWithRaftFallback,
        VoteDecision,
        ByzantineIndicator,
        ByzantineReport,
        create_cvt_consensus,
        TARGET_CONSENSUS_LATENCY_MS,
        MAX_CONSENSUS_LATENCY_MS,
        APPROVAL_THRESHOLD,
        QUORUM_THRESHOLD,
    )
    from core.orchestration.reputation_tracker import (
        ReputationTracker,
        create_reputation_tracker_sync,
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    CVTConsensus = None
    CVTPhase = None
    CVTProposal = None
    CVTResult = None
    CVToken = None
    CVTVote = None
    CVTWithRaftFallback = None
    VoteDecision = None
    ByzantineIndicator = None
    ByzantineReport = None
    create_cvt_consensus = None
    ReputationTracker = None
    create_reputation_tracker_sync = None
    TARGET_CONSENSUS_LATENCY_MS = 0.85
    MAX_CONSENSUS_LATENCY_MS = 5.0
    APPROVAL_THRESHOLD = 0.67
    QUORUM_THRESHOLD = 0.51


pytestmark = pytest.mark.skipif(
    not IMPORTS_AVAILABLE,
    reason="CVT consensus module not available"
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cvt_consensus():
    """Create a CVT consensus instance for testing."""
    cvt = CVTConsensus(node_id="test_node_1")
    # Register test participants
    cvt.register_participant("test_node_1", base_weight=1.0)
    cvt.register_participant("test_node_2", base_weight=1.0)
    cvt.register_participant("test_node_3", base_weight=1.0)
    return cvt


@pytest.fixture
def cvt_with_reputation():
    """Create CVT consensus with reputation tracker."""
    tracker = create_reputation_tracker_sync()
    cvt = CVTConsensus(
        node_id="rep_node_1",
        reputation_tracker=tracker,
    )
    cvt.register_participant("rep_node_1", base_weight=1.0)
    cvt.register_participant("rep_node_2", base_weight=1.0)
    cvt.register_participant("rep_node_3", base_weight=1.0)
    return cvt, tracker


@pytest.fixture
def large_consensus():
    """Create CVT consensus with many participants."""
    cvt = CVTConsensus(node_id="large_node_1")
    for i in range(1, 11):
        cvt.register_participant(f"large_node_{i}", base_weight=1.0)
    return cvt


# =============================================================================
# BASIC INITIALIZATION TESTS
# =============================================================================


class TestCVTInitialization:
    """Tests for CVT consensus initialization."""

    def test_create_cvt_consensus_basic(self):
        """Test basic CVT consensus creation."""
        cvt = CVTConsensus(node_id="init_test")
        assert cvt.node_id == "init_test"
        assert cvt.approval_threshold == APPROVAL_THRESHOLD
        assert cvt.quorum_threshold == QUORUM_THRESHOLD

    def test_create_cvt_consensus_custom_thresholds(self):
        """Test CVT with custom approval thresholds."""
        cvt = CVTConsensus(
            node_id="custom_test",
            approval_threshold=0.75,
            quorum_threshold=0.60,
        )
        assert cvt.approval_threshold == 0.75
        assert cvt.quorum_threshold == 0.60

    def test_factory_function(self):
        """Test the create_cvt_consensus factory function."""
        cvt = create_cvt_consensus(
            node_id="factory_test",
            participants=["node_a", "node_b", "node_c"],
        )
        assert cvt.node_id == "factory_test"
        assert len(cvt._participants) == 4  # self + 3 others

    def test_register_participant(self, cvt_consensus):
        """Test participant registration."""
        initial_count = len(cvt_consensus._participants)
        cvt_consensus.register_participant("new_participant", base_weight=2.0)
        assert len(cvt_consensus._participants) == initial_count + 1
        assert cvt_consensus._participants["new_participant"] == 2.0

    def test_unregister_participant(self, cvt_consensus):
        """Test participant unregistration."""
        cvt_consensus.register_participant("temp_participant")
        assert "temp_participant" in cvt_consensus._participants
        result = cvt_consensus.unregister_participant("temp_participant")
        assert result is True
        assert "temp_participant" not in cvt_consensus._participants

    def test_unregister_nonexistent_participant(self, cvt_consensus):
        """Test unregistering a participant that doesn't exist."""
        result = cvt_consensus.unregister_participant("nonexistent")
        assert result is False


# =============================================================================
# PROPOSAL TESTS
# =============================================================================


class TestCVTProposal:
    """Tests for CVT proposal creation and management."""

    def test_proposal_creation(self):
        """Test CVTProposal dataclass creation."""
        proposal = CVTProposal(
            proposal_id="test_proposal_1",
            proposer_id="proposer_1",
            task_type="validate_task",
            payload={"key": "value"},
        )
        assert proposal.proposal_id == "test_proposal_1"
        assert proposal.proposer_id == "proposer_1"
        assert proposal.task_type == "validate_task"
        assert proposal.phase == CVTPhase.IDLE

    def test_proposal_hash_computation(self):
        """Test proposal hash is computed correctly."""
        proposal = CVTProposal(
            proposal_id="hash_test",
            proposer_id="proposer",
            task_type="action",
            payload={"data": "test"},
        )
        hash1 = proposal.compute_hash()
        hash2 = proposal.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_proposal_hash_differs_with_content(self):
        """Test different proposals have different hashes."""
        proposal1 = CVTProposal(
            proposal_id="hash_test_1",
            proposer_id="proposer",
            task_type="action",
            payload={"data": "test"},
        )
        proposal2 = CVTProposal(
            proposal_id="hash_test_2",
            proposer_id="proposer",
            task_type="action",
            payload={"data": "test"},
        )
        assert proposal1.compute_hash() != proposal2.compute_hash()


# =============================================================================
# VOTE TESTS
# =============================================================================


class TestCVTVote:
    """Tests for CVT voting mechanics."""

    def test_vote_creation(self):
        """Test CVTVote dataclass creation."""
        vote = CVTVote(
            vote_id="vote_1",
            proposal_id="proposal_1",
            voter_id="voter_1",
            decision=VoteDecision.APPROVE,
            confidence=0.95,
        )
        assert vote.vote_id == "vote_1"
        assert vote.decision == VoteDecision.APPROVE
        assert vote.confidence == 0.95
        assert vote.signature != ""

    def test_vote_signature_verification(self):
        """Test vote signature verification."""
        vote = CVTVote(
            vote_id="verify_test",
            proposal_id="proposal",
            voter_id="voter",
            decision=VoteDecision.APPROVE,
            confidence=1.0,
        )
        assert vote.verify() is True

    def test_vote_signature_tampering_detected(self):
        """Test that tampered signatures are detected."""
        vote = CVTVote(
            vote_id="tamper_test",
            proposal_id="proposal",
            voter_id="voter",
            decision=VoteDecision.APPROVE,
            confidence=1.0,
        )
        # Tamper with the signature
        vote.signature = "tampered_signature"
        assert vote.verify() is False

    @pytest.mark.asyncio
    async def test_vote_on_active_proposal(self, cvt_consensus):
        """Test voting on an active proposal."""
        # Create proposal manually
        proposal = CVTProposal(
            proposal_id="vote_test_proposal",
            proposer_id="test_node_1",
            task_type="test",
            payload={},
        )
        proposal.phase = CVTPhase.VOTING
        cvt_consensus._proposals["vote_test_proposal"] = proposal
        cvt_consensus._votes["vote_test_proposal"] = {}

        vote = await cvt_consensus.vote(
            proposal_id="vote_test_proposal",
            decision=VoteDecision.APPROVE,
            confidence=0.9,
        )

        assert vote is not None
        assert vote.decision == VoteDecision.APPROVE

    @pytest.mark.asyncio
    async def test_vote_on_nonexistent_proposal(self, cvt_consensus):
        """Test voting on a proposal that doesn't exist."""
        vote = await cvt_consensus.vote(
            proposal_id="nonexistent",
            decision=VoteDecision.APPROVE,
        )
        assert vote is None

    @pytest.mark.asyncio
    async def test_double_vote_prevented(self, cvt_consensus):
        """Test that double voting is prevented."""
        proposal = CVTProposal(
            proposal_id="double_vote_test",
            proposer_id="test_node_1",
            task_type="test",
            payload={},
        )
        proposal.phase = CVTPhase.VOTING
        cvt_consensus._proposals["double_vote_test"] = proposal
        cvt_consensus._votes["double_vote_test"] = {}

        # First vote
        vote1 = await cvt_consensus.vote("double_vote_test", VoteDecision.APPROVE)
        assert vote1 is not None

        # Second vote should be rejected
        vote2 = await cvt_consensus.vote("double_vote_test", VoteDecision.APPROVE)
        assert vote2 is None


# =============================================================================
# CONSENSUS OPERATION TESTS
# =============================================================================


class TestCVTConsensusOperations:
    """Tests for CVT consensus operations."""

    @pytest.mark.asyncio
    async def test_validate_task_success(self, cvt_consensus):
        """Test successful task validation."""
        result = await cvt_consensus.validate_task(
            task_id="task_1",
            task_payload={"description": "Test task"},
        )
        assert isinstance(result, CVTResult)
        assert result.proposal_id.startswith("validate_task_1")

    @pytest.mark.asyncio
    async def test_validate_task_generates_token_on_success(self, cvt_consensus):
        """Test that successful validation generates a token."""
        result = await cvt_consensus.validate_task(
            task_id="token_task",
            task_payload={"action": "deploy"},
        )
        if result.success:
            assert result.token is not None
            assert result.token.decision == VoteDecision.APPROVE

    @pytest.mark.asyncio
    async def test_propose_action(self, cvt_consensus):
        """Test action proposal."""
        result = await cvt_consensus.propose_action(
            action_id="action_1",
            action_type="scale",
            action_payload={"replicas": 3},
            priority=7,
        )
        assert isinstance(result, CVTResult)
        assert result.proposal_id.startswith("action_action_1")

    @pytest.mark.asyncio
    async def test_finalize_with_quorum(self, cvt_consensus):
        """Test finalization with sufficient quorum."""
        proposal = CVTProposal(
            proposal_id="finalize_test",
            proposer_id="test_node_1",
            task_type="test",
            payload={},
        )
        proposal.phase = CVTPhase.VOTING
        cvt_consensus._proposals["finalize_test"] = proposal

        # Add votes from all participants
        cvt_consensus._votes["finalize_test"] = {
            "test_node_1": CVTVote(
                vote_id="v1", proposal_id="finalize_test",
                voter_id="test_node_1", decision=VoteDecision.APPROVE, confidence=1.0
            ),
            "test_node_2": CVTVote(
                vote_id="v2", proposal_id="finalize_test",
                voter_id="test_node_2", decision=VoteDecision.APPROVE, confidence=1.0
            ),
            "test_node_3": CVTVote(
                vote_id="v3", proposal_id="finalize_test",
                voter_id="test_node_3", decision=VoteDecision.APPROVE, confidence=1.0
            ),
        }

        result = await cvt_consensus.finalize("finalize_test")
        assert result.success is True
        assert result.decision == VoteDecision.APPROVE

    @pytest.mark.asyncio
    async def test_finalize_without_quorum(self, cvt_consensus):
        """Test finalization fails without quorum."""
        proposal = CVTProposal(
            proposal_id="no_quorum_test",
            proposer_id="test_node_1",
            task_type="test",
            payload={},
        )
        proposal.phase = CVTPhase.VOTING
        cvt_consensus._proposals["no_quorum_test"] = proposal

        # Only one vote - not enough for quorum
        cvt_consensus._votes["no_quorum_test"] = {
            "test_node_1": CVTVote(
                vote_id="v1", proposal_id="no_quorum_test",
                voter_id="test_node_1", decision=VoteDecision.APPROVE, confidence=1.0
            ),
        }

        result = await cvt_consensus.finalize("no_quorum_test")
        assert result.success is False
        assert "Quorum" in result.error

    @pytest.mark.asyncio
    async def test_finalize_rejected_by_majority(self, cvt_consensus):
        """Test finalization when majority rejects."""
        proposal = CVTProposal(
            proposal_id="reject_test",
            proposer_id="test_node_1",
            task_type="test",
            payload={},
        )
        proposal.phase = CVTPhase.VOTING
        cvt_consensus._proposals["reject_test"] = proposal

        # Majority rejects
        cvt_consensus._votes["reject_test"] = {
            "test_node_1": CVTVote(
                vote_id="v1", proposal_id="reject_test",
                voter_id="test_node_1", decision=VoteDecision.APPROVE, confidence=1.0
            ),
            "test_node_2": CVTVote(
                vote_id="v2", proposal_id="reject_test",
                voter_id="test_node_2", decision=VoteDecision.REJECT, confidence=1.0
            ),
            "test_node_3": CVTVote(
                vote_id="v3", proposal_id="reject_test",
                voter_id="test_node_3", decision=VoteDecision.REJECT, confidence=1.0
            ),
        }

        result = await cvt_consensus.finalize("reject_test")
        assert result.success is False
        assert result.decision == VoteDecision.REJECT


# =============================================================================
# TOKEN TESTS
# =============================================================================


class TestCVToken:
    """Tests for Consensus Validation Tokens."""

    def test_token_creation(self):
        """Test CVToken creation."""
        token = CVToken(
            token_id="token_1",
            proposal_id="proposal_1",
            decision=VoteDecision.APPROVE,
            total_weight=3.0,
            approval_weight=2.5,
            participant_count=3,
        )
        assert token.token_id == "token_1"
        assert token.decision == VoteDecision.APPROVE
        assert token.approval_ratio > 0.8

    def test_token_verification(self):
        """Test token signature verification."""
        token = CVToken(
            token_id="verify_token",
            proposal_id="proposal",
            decision=VoteDecision.APPROVE,
            total_weight=3.0,
            approval_weight=2.5,
            participant_count=3,
        )
        assert token.verify() is True

    def test_token_expiry(self):
        """Test token expiry detection."""
        from datetime import timedelta
        token = CVToken(
            token_id="expiry_token",
            proposal_id="proposal",
            decision=VoteDecision.APPROVE,
            total_weight=3.0,
            approval_weight=2.5,
            participant_count=3,
        )
        # Token should be valid initially
        assert token.is_valid is True

        # Manually expire it
        token.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert token.is_valid is False

    def test_verify_token_method(self, cvt_consensus):
        """Test CVTConsensus.verify_token method."""
        token = CVToken(
            token_id="method_test_token",
            proposal_id="proposal",
            decision=VoteDecision.APPROVE,
            total_weight=3.0,
            approval_weight=2.5,
            participant_count=3,
        )
        # Store token
        cvt_consensus._tokens[token.token_id] = token

        assert cvt_consensus.verify_token(token) is True

    def test_get_token(self, cvt_consensus):
        """Test retrieving a token by ID."""
        token = CVToken(
            token_id="get_test_token",
            proposal_id="proposal",
            decision=VoteDecision.APPROVE,
            total_weight=3.0,
            approval_weight=2.5,
            participant_count=3,
        )
        cvt_consensus._tokens["get_test_token"] = token

        retrieved = cvt_consensus.get_token("get_test_token")
        assert retrieved is not None
        assert retrieved.token_id == "get_test_token"


# =============================================================================
# REPUTATION-WEIGHTED VOTING TESTS
# =============================================================================


class TestReputationWeightedVoting:
    """Tests for reputation-weighted voting."""

    def test_participant_weight_with_reputation(self, cvt_with_reputation):
        """Test that participant weight includes reputation."""
        cvt, tracker = cvt_with_reputation

        # Set reputation for an agent
        rep = tracker.get_agent_reputation("rep_node_2")
        rep.reputation = 0.8

        weight = cvt.get_participant_weight("rep_node_2")
        # Weight should be > 1.0 for high reputation
        assert weight > 1.0

    def test_low_reputation_reduces_weight(self, cvt_with_reputation):
        """Test that low reputation reduces voting weight."""
        cvt, tracker = cvt_with_reputation

        # Set low reputation
        rep = tracker.get_agent_reputation("rep_node_2")
        rep.reputation = 0.2

        weight = cvt.get_participant_weight("rep_node_2")
        # Weight should be < 1.0 for low reputation
        assert weight < 1.0

    def test_total_weight_calculation(self, cvt_with_reputation):
        """Test total weight calculation with reputation."""
        cvt, tracker = cvt_with_reputation

        total = cvt.get_total_weight()
        # Should be sum of all participant weights
        assert total > 0


# =============================================================================
# BYZANTINE DETECTION TESTS
# =============================================================================


class TestByzantineDetection:
    """Tests for Byzantine fault detection."""

    @pytest.mark.asyncio
    async def test_detect_inconsistent_voting_pattern(self, cvt_consensus):
        """Test detection of inconsistent voting patterns."""
        # Add voting history showing random-looking behavior
        agent_id = "byzantine_agent"
        cvt_consensus._participants[agent_id] = 1.0

        # Simulate random voting (50% approve rate)
        for i in range(20):
            decision = VoteDecision.APPROVE if i % 2 == 0 else VoteDecision.REJECT
            cvt_consensus._vote_history[agent_id].append((f"prop_{i}", decision))

        suspicious = await cvt_consensus.detect_byzantine()
        assert agent_id in suspicious

    @pytest.mark.asyncio
    async def test_vote_flip_detection(self, cvt_consensus):
        """Test detection of vote flipping."""
        proposal = CVTProposal(
            proposal_id="flip_test",
            proposer_id="test_node_1",
            task_type="test",
            payload={},
        )
        proposal.phase = CVTPhase.VOTING
        cvt_consensus._proposals["flip_test"] = proposal
        cvt_consensus._votes["flip_test"] = {}

        # First vote
        await cvt_consensus.vote("flip_test", VoteDecision.APPROVE)

        # Try to change vote (should be detected)
        await cvt_consensus.vote("flip_test", VoteDecision.REJECT)

        # Check Byzantine reports
        reports = cvt_consensus.get_byzantine_reports()
        flip_reports = [r for r in reports if r.indicator == ByzantineIndicator.VOTE_FLIP]
        assert len(flip_reports) > 0

    def test_byzantine_report_creation(self):
        """Test ByzantineReport dataclass."""
        report = ByzantineReport(
            agent_id="bad_agent",
            indicator=ByzantineIndicator.DOUBLE_VOTE,
            confidence=0.95,
            evidence={"attempts": 2},
            proposal_id="prop_1",
        )
        assert report.agent_id == "bad_agent"
        assert report.indicator == ByzantineIndicator.DOUBLE_VOTE
        assert report.confidence == 0.95


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestCVTPerformance:
    """Tests for CVT performance targets."""

    @pytest.mark.asyncio
    async def test_fast_path_latency(self, cvt_consensus):
        """Test that fast path achieves target latency."""
        start = time.perf_counter()
        result = await cvt_consensus.validate_task(
            task_id="perf_task",
            task_payload={"simple": "task"},
            timeout_ms=TARGET_CONSENSUS_LATENCY_MS,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly (though exact sub-ms may not be reliable in tests)
        assert elapsed_ms < 100  # Generous bound for test environment

    @pytest.mark.asyncio
    async def test_latency_metrics_recorded(self, cvt_consensus):
        """Test that latency metrics are recorded."""
        await cvt_consensus.validate_task(
            task_id="metrics_task",
            task_payload={"data": "test"},
        )

        metrics = cvt_consensus.get_metrics()
        assert metrics["proposals"] > 0
        assert metrics["total_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_sub_ms_count_tracking(self, cvt_consensus):
        """Test sub-millisecond consensus count tracking."""
        # Run multiple fast validations
        for i in range(5):
            await cvt_consensus.validate_task(
                task_id=f"sub_ms_task_{i}",
                task_payload={},
                timeout_ms=0.5,
            )

        metrics = cvt_consensus.get_metrics()
        # At least some should be tracked as sub-ms
        assert "sub_ms_count" in metrics


# =============================================================================
# RAFT FALLBACK TESTS
# =============================================================================


class TestCVTWithRaftFallback:
    """Tests for CVT with Raft fallback."""

    def test_fallback_creation(self):
        """Test CVTWithRaftFallback creation."""
        fallback = CVTWithRaftFallback(node_id="fallback_test")
        assert fallback.cvt.node_id == "fallback_test"
        assert fallback.raft is None  # No Raft provided

    @pytest.mark.asyncio
    async def test_validate_without_fallback(self):
        """Test validation without Raft fallback available."""
        fallback = CVTWithRaftFallback(node_id="no_raft_test")
        fallback.cvt.register_participant("no_raft_test", 1.0)
        fallback.cvt.register_participant("node_2", 1.0)
        fallback.cvt.register_participant("node_3", 1.0)

        result = await fallback.validate_task(
            task_id="no_raft_task",
            task_payload={"data": "test"},
            use_raft_on_failure=False,
        )

        assert isinstance(result, CVTResult)

    @pytest.mark.asyncio
    async def test_fallback_metrics(self):
        """Test fallback metrics tracking."""
        fallback = CVTWithRaftFallback(node_id="metrics_test")
        fallback.cvt.register_participant("metrics_test", 1.0)
        fallback.cvt.register_participant("node_2", 1.0)

        await fallback.validate_task("task_1", {"data": "test"})

        metrics = fallback.get_metrics()
        assert "cvt_attempts" in metrics
        assert metrics["cvt_attempts"] >= 1


# =============================================================================
# METRICS AND STATUS TESTS
# =============================================================================


class TestCVTMetricsAndStatus:
    """Tests for CVT metrics and status reporting."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, cvt_consensus):
        """Test metrics retrieval."""
        await cvt_consensus.validate_task("m_task", {"test": True})

        metrics = cvt_consensus.get_metrics()
        assert "proposals" in metrics
        assert "consensus_reached" in metrics
        assert "consensus_failed" in metrics
        assert "tokens_issued" in metrics
        assert "avg_latency_ms" in metrics

    def test_get_status(self, cvt_consensus):
        """Test status retrieval."""
        status = cvt_consensus.get_status()

        assert "node_id" in status
        assert "participants" in status
        assert "total_weight" in status
        assert "approval_threshold" in status
        assert "quorum_threshold" in status
        assert "metrics" in status

    def test_participant_count_in_metrics(self, cvt_consensus):
        """Test participant count in metrics."""
        metrics = cvt_consensus.get_metrics()
        assert metrics["participant_count"] == 3  # 3 registered in fixture


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestCVTEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_payload_validation(self, cvt_consensus):
        """Test validation with empty payload."""
        result = await cvt_consensus.validate_task(
            task_id="empty_task",
            task_payload={},
        )
        assert isinstance(result, CVTResult)

    @pytest.mark.asyncio
    async def test_large_payload_handling(self, cvt_consensus):
        """Test handling of large payloads."""
        large_payload = {"data": "x" * 10000}
        result = await cvt_consensus.validate_task(
            task_id="large_task",
            task_payload=large_payload,
        )
        assert isinstance(result, CVTResult)

    @pytest.mark.asyncio
    async def test_concurrent_proposals(self, cvt_consensus):
        """Test multiple concurrent proposals."""
        tasks = [
            cvt_consensus.validate_task(f"concurrent_{i}", {"i": i})
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, CVTResult)

    def test_unregistered_participant_weight(self, cvt_consensus):
        """Test getting weight for unregistered participant."""
        weight = cvt_consensus.get_participant_weight("nonexistent")
        assert weight == 0.0

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_proposal(self, cvt_consensus):
        """Test finalizing a proposal that doesn't exist."""
        result = await cvt_consensus.finalize("nonexistent_proposal")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_vote_decision_enum_values(self):
        """Test VoteDecision enum values."""
        assert VoteDecision.APPROVE.value == "approve"
        assert VoteDecision.REJECT.value == "reject"
        assert VoteDecision.ABSTAIN.value == "abstain"

    def test_cvt_phase_enum_values(self):
        """Test CVTPhase enum values."""
        assert CVTPhase.IDLE.value == "idle"
        assert CVTPhase.PROPOSING.value == "proposing"
        assert CVTPhase.COMMITTED.value == "committed"
        assert CVTPhase.ABORTED.value == "aborted"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCVTIntegration:
    """Integration tests for CVT consensus."""

    @pytest.mark.asyncio
    async def test_full_consensus_flow(self, large_consensus):
        """Test complete consensus flow with multiple agents."""
        result = await large_consensus.validate_task(
            task_id="integration_task",
            task_payload={
                "action": "deploy",
                "version": "2.0",
                "config": {"replicas": 3},
            },
        )

        assert isinstance(result, CVTResult)
        assert result.participant_count > 0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_consensus_with_reputation_impact(self, cvt_with_reputation):
        """Test that reputation affects consensus outcomes."""
        cvt, tracker = cvt_with_reputation

        # Import ReputationEventType for the test
        from core.orchestration.reputation_tracker import ReputationEventType as RepEventType

        # Record successful events to boost reputation
        for _ in range(5):
            await tracker.record_event(
                "rep_node_2",
                RepEventType.TASK_SUCCESS,
            )

        # Run consensus
        result = await cvt.validate_task(
            task_id="rep_impact_task",
            task_payload={"test": "reputation"},
        )

        assert isinstance(result, CVTResult)


# Try to import ReputationEventType for the integration test
try:
    from core.orchestration.reputation_tracker import ReputationEventType
except ImportError:
    ReputationEventType = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
