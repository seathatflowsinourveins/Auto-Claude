"""
Tests for Anthropic Message Batches API Adapter (V66)
======================================================

Comprehensive test suite covering:
- Adapter initialization and configuration
- Batch creation and validation
- Status polling and retrieval
- Results processing
- Cancellation handling
- Error handling and resilience
- Edge cases and input validation

Test count: 45+ tests
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Fix imports for test environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.anthropic_batch_adapter import (
    AnthropicBatchAdapter,
    BatchProcessingStatus,
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchResultType,
    MAX_BATCH_REQUESTS,
    MAX_BATCH_SIZE_BYTES,
    MessageBatch,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_batch_request():
    """Sample batch request for testing."""
    return {
        "custom_id": "test-request-1",
        "params": {
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello, world"}],
        }
    }


@pytest.fixture
def sample_batch_requests():
    """List of sample batch requests."""
    return [
        {
            "custom_id": f"test-request-{i}",
            "params": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": f"Test message {i}"}],
            }
        }
        for i in range(3)
    ]


@pytest.fixture
def sample_batch_response():
    """Sample batch API response."""
    return {
        "id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": {
            "processing": 3,
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "ended_at": None,
        "created_at": "2024-09-24T18:37:24.100435Z",
        "expires_at": "2024-09-25T18:37:24.100435Z",
        "cancel_initiated_at": None,
        "results_url": None,
    }


@pytest.fixture
def sample_completed_batch_response():
    """Sample completed batch API response."""
    return {
        "id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
        "type": "message_batch",
        "processing_status": "ended",
        "request_counts": {
            "processing": 0,
            "succeeded": 2,
            "errored": 1,
            "canceled": 0,
            "expired": 0,
        },
        "ended_at": "2024-09-24T19:37:24.100435Z",
        "created_at": "2024-09-24T18:37:24.100435Z",
        "expires_at": "2024-09-25T18:37:24.100435Z",
        "cancel_initiated_at": None,
        "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d/results",
    }


@pytest.fixture
def sample_results_jsonl():
    """Sample JSONL results content."""
    results = [
        {
            "custom_id": "test-request-0",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_01abc",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "Hello! Response 0."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                }
            }
        },
        {
            "custom_id": "test-request-1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_01def",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "Hello! Response 1."}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 25},
                }
            }
        },
        {
            "custom_id": "test-request-2",
            "result": {
                "type": "errored",
                "error": {
                    "type": "invalid_request",
                    "message": "Invalid model specified",
                }
            }
        },
    ]
    return "\n".join(json.dumps(r) for r in results)


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


# =============================================================================
# Data Class Tests
# =============================================================================

class TestBatchProcessingStatus:
    """Tests for BatchProcessingStatus enum."""

    def test_status_values(self):
        """Test all status values are defined."""
        assert BatchProcessingStatus.IN_PROGRESS.value == "in_progress"
        assert BatchProcessingStatus.CANCELING.value == "canceling"
        assert BatchProcessingStatus.ENDED.value == "ended"

    def test_status_from_string(self):
        """Test creating status from string."""
        status = BatchProcessingStatus("in_progress")
        assert status == BatchProcessingStatus.IN_PROGRESS


class TestBatchResultType:
    """Tests for BatchResultType enum."""

    def test_result_type_values(self):
        """Test all result type values are defined."""
        assert BatchResultType.SUCCEEDED.value == "succeeded"
        assert BatchResultType.ERRORED.value == "errored"
        assert BatchResultType.CANCELED.value == "canceled"
        assert BatchResultType.EXPIRED.value == "expired"


class TestBatchRequestCounts:
    """Tests for BatchRequestCounts dataclass."""

    def test_empty_counts(self):
        """Test default empty counts."""
        counts = BatchRequestCounts()
        assert counts.processing == 0
        assert counts.succeeded == 0
        assert counts.total == 0
        assert counts.completed == 0

    def test_total_calculation(self):
        """Test total calculation."""
        counts = BatchRequestCounts(processing=5, succeeded=10, errored=2, canceled=1, expired=0)
        assert counts.total == 18
        assert counts.completed == 13

    def test_to_dict(self):
        """Test conversion to dictionary."""
        counts = BatchRequestCounts(processing=5, succeeded=10)
        d = counts.to_dict()
        assert d["processing"] == 5
        assert d["succeeded"] == 10
        assert d["total"] == 15
        assert d["completed"] == 10


class TestBatchRequest:
    """Tests for BatchRequest dataclass."""

    def test_batch_request_creation(self):
        """Test creating a batch request."""
        req = BatchRequest(
            custom_id="test-1",
            params={"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}
        )
        assert req.custom_id == "test-1"
        assert req.params["model"] == "claude-sonnet-4-5"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        req = BatchRequest(
            custom_id="test-1",
            params={"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}
        )
        d = req.to_dict()
        assert d["custom_id"] == "test-1"
        assert d["params"]["model"] == "claude-sonnet-4-5"


class TestMessageBatch:
    """Tests for MessageBatch dataclass."""

    def test_from_dict(self, sample_batch_response):
        """Test creating MessageBatch from API response."""
        batch = MessageBatch.from_dict(sample_batch_response)
        assert batch.id == "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
        assert batch.processing_status == BatchProcessingStatus.IN_PROGRESS
        assert batch.request_counts.processing == 3

    def test_to_dict(self, sample_batch_response):
        """Test converting MessageBatch to dictionary."""
        batch = MessageBatch.from_dict(sample_batch_response)
        d = batch.to_dict()
        assert d["id"] == "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
        assert d["processing_status"] == "in_progress"
        assert d["is_complete"] is False

    def test_completed_batch(self, sample_completed_batch_response):
        """Test completed batch status."""
        batch = MessageBatch.from_dict(sample_completed_batch_response)
        assert batch.processing_status == BatchProcessingStatus.ENDED
        assert batch.results_url is not None
        assert batch.to_dict()["is_complete"] is True


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_succeeded_result(self):
        """Test creating succeeded result."""
        data = {
            "custom_id": "test-1",
            "result": {
                "type": "succeeded",
                "message": {"id": "msg_abc", "content": [{"type": "text", "text": "Hi"}]}
            }
        }
        result = BatchResult.from_dict(data)
        assert result.result_type == BatchResultType.SUCCEEDED
        assert result.message is not None
        assert result.error is None

    def test_errored_result(self):
        """Test creating errored result."""
        data = {
            "custom_id": "test-1",
            "result": {
                "type": "errored",
                "error": {"type": "invalid_request", "message": "Bad request"}
            }
        }
        result = BatchResult.from_dict(data)
        assert result.result_type == BatchResultType.ERRORED
        assert result.message is None
        assert result.error is not None

    def test_to_dict_success(self):
        """Test to_dict for successful result."""
        data = {
            "custom_id": "test-1",
            "result": {"type": "succeeded", "message": {"id": "msg_abc"}}
        }
        result = BatchResult.from_dict(data)
        d = result.to_dict()
        assert d["success"] is True
        assert d["result_type"] == "succeeded"

    def test_to_dict_error(self):
        """Test to_dict for errored result."""
        data = {
            "custom_id": "test-1",
            "result": {"type": "errored", "error": {"message": "Error"}}
        }
        result = BatchResult.from_dict(data)
        d = result.to_dict()
        assert d["success"] is False
        assert d["result_type"] == "errored"


# =============================================================================
# Adapter Initialization Tests
# =============================================================================

class TestAdapterInitialization:
    """Tests for adapter initialization."""

    def test_adapter_properties(self):
        """Test adapter basic properties."""
        adapter = AnthropicBatchAdapter()
        assert adapter.sdk_name == "anthropic_batch"
        assert adapter.available is False  # Not initialized yet

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        adapter = AnthropicBatchAdapter()

        # Clear environment variable if set
        with patch.dict(os.environ, {}, clear=True):
            result = await adapter.initialize({})
            assert result.success is False
            assert "ANTHROPIC_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_initialize_with_config_api_key(self, mock_httpx_client):
        """Test initialization with API key in config."""
        adapter = AnthropicBatchAdapter()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_httpx_client.get.return_value = mock_response

        with patch("adapters.anthropic_batch_adapter.httpx.AsyncClient", return_value=mock_httpx_client):
            result = await adapter.initialize({"api_key": "test-key"})
            assert result.success is True
            assert adapter.available is True
            assert "50% cost savings" in result.data["features"]

    @pytest.mark.asyncio
    async def test_initialize_with_env_api_key(self, mock_httpx_client):
        """Test initialization with API key from environment."""
        adapter = AnthropicBatchAdapter()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_httpx_client.get.return_value = mock_response

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-test-key"}):
            with patch("adapters.anthropic_batch_adapter.httpx.AsyncClient", return_value=mock_httpx_client):
                result = await adapter.initialize({})
                assert result.success is True

    @pytest.mark.asyncio
    async def test_initialize_with_custom_base_url(self, mock_httpx_client):
        """Test initialization with custom base URL."""
        adapter = AnthropicBatchAdapter()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_httpx_client.get.return_value = mock_response

        with patch("adapters.anthropic_batch_adapter.httpx.AsyncClient", return_value=mock_httpx_client):
            result = await adapter.initialize({
                "api_key": "test-key",
                "base_url": "https://custom.api.com/v1"
            })
            assert result.success is True
            assert result.data["base_url"] == "https://custom.api.com/v1"


# =============================================================================
# Batch Creation Tests
# =============================================================================

class TestBatchCreation:
    """Tests for batch creation operation."""

    @pytest.mark.asyncio
    async def test_create_batch_success(self, mock_httpx_client, sample_batch_requests, sample_batch_response):
        """Test successful batch creation."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_batch_response
        mock_httpx_client.post.return_value = mock_response

        result = await adapter.execute("create_batch", requests=sample_batch_requests)

        assert result.success is True
        assert result.data["batch_id"] == "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"
        assert result.data["request_count"] == 3

    @pytest.mark.asyncio
    async def test_create_batch_empty_requests(self, mock_httpx_client):
        """Test batch creation fails with empty requests."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("create_batch", requests=[])
        assert result.success is False
        assert "requests parameter required" in result.error

    @pytest.mark.asyncio
    async def test_create_batch_missing_custom_id(self, mock_httpx_client):
        """Test batch creation fails without custom_id."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("create_batch", requests=[
            {"params": {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}}
        ])
        assert result.success is False
        assert "missing custom_id" in result.error

    @pytest.mark.asyncio
    async def test_create_batch_duplicate_custom_id(self, mock_httpx_client):
        """Test batch creation fails with duplicate custom_id."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("create_batch", requests=[
            {"custom_id": "same-id", "params": {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}},
            {"custom_id": "same-id", "params": {"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}},
        ])
        assert result.success is False
        assert "Duplicate custom_id" in result.error

    @pytest.mark.asyncio
    async def test_create_batch_missing_model(self, mock_httpx_client):
        """Test batch creation fails without model in params."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("create_batch", requests=[
            {"custom_id": "test-1", "params": {"messages": [], "max_tokens": 100}}
        ])
        assert result.success is False
        assert "missing model" in result.error

    @pytest.mark.asyncio
    async def test_create_batch_missing_messages(self, mock_httpx_client):
        """Test batch creation fails without messages in params."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("create_batch", requests=[
            {"custom_id": "test-1", "params": {"model": "claude-sonnet-4-5", "max_tokens": 100}}
        ])
        assert result.success is False
        assert "missing messages" in result.error

    @pytest.mark.asyncio
    async def test_create_batch_missing_max_tokens(self, mock_httpx_client):
        """Test batch creation fails without max_tokens in params."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("create_batch", requests=[
            {"custom_id": "test-1", "params": {"model": "claude-sonnet-4-5", "messages": []}}
        ])
        assert result.success is False
        assert "missing max_tokens" in result.error

    @pytest.mark.asyncio
    async def test_create_batch_with_batch_request_objects(self, mock_httpx_client, sample_batch_response):
        """Test batch creation with BatchRequest objects."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_batch_response
        mock_httpx_client.post.return_value = mock_response

        requests = [
            BatchRequest(
                custom_id="test-1",
                params={"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 100}
            )
        ]
        result = await adapter.execute("create_batch", requests=requests)
        assert result.success is True


# =============================================================================
# Batch Status Tests
# =============================================================================

class TestBatchStatus:
    """Tests for batch status operations."""

    @pytest.mark.asyncio
    async def test_get_batch_status_in_progress(self, mock_httpx_client, sample_batch_response):
        """Test getting status of in-progress batch."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_batch_response
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute("get_batch_status", batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")

        assert result.success is True
        assert result.data["processing_status"] == "in_progress"
        assert result.data["is_complete"] is False
        assert result.data["request_counts"]["processing"] == 3

    @pytest.mark.asyncio
    async def test_get_batch_status_completed(self, mock_httpx_client, sample_completed_batch_response):
        """Test getting status of completed batch."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_completed_batch_response
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute("get_batch_status", batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")

        assert result.success is True
        assert result.data["processing_status"] == "ended"
        assert result.data["is_complete"] is True

    @pytest.mark.asyncio
    async def test_get_batch_status_missing_id(self, mock_httpx_client):
        """Test get_batch_status fails without batch_id."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("get_batch_status")
        assert result.success is False
        assert "batch_id required" in result.error

    @pytest.mark.asyncio
    async def test_get_batch_status_not_found(self, mock_httpx_client):
        """Test get_batch_status handles 404."""
        import httpx

        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.status_code = 404
        error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=mock_response)
        mock_httpx_client.get.side_effect = error

        result = await adapter.execute("get_batch_status", batch_id="invalid-id")
        assert result.success is False
        assert "not found" in result.error.lower()


# =============================================================================
# Batch Results Tests
# =============================================================================

class TestBatchResults:
    """Tests for batch results retrieval."""

    @pytest.mark.asyncio
    async def test_get_batch_results_success(
        self, mock_httpx_client, sample_completed_batch_response, sample_results_jsonl
    ):
        """Test successful results retrieval."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        # Mock status response
        status_response = MagicMock()
        status_response.raise_for_status = MagicMock()
        status_response.json.return_value = sample_completed_batch_response

        # Mock results response
        results_response = MagicMock()
        results_response.raise_for_status = MagicMock()
        results_response.text = sample_results_jsonl

        mock_httpx_client.get.side_effect = [status_response, results_response]

        result = await adapter.execute("get_batch_results", batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")

        assert result.success is True
        assert result.data["total_count"] == 3
        assert result.data["summary"]["succeeded"] == 2
        assert result.data["summary"]["errored"] == 1
        assert result.data["summary"]["success_rate"] == pytest.approx(66.67, rel=0.1)

    @pytest.mark.asyncio
    async def test_get_batch_results_not_complete(self, mock_httpx_client, sample_batch_response):
        """Test results retrieval fails if batch not complete."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_batch_response  # in_progress
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute("get_batch_results", batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")

        assert result.success is False
        assert "not complete" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_batch_results_with_raw_jsonl(
        self, mock_httpx_client, sample_completed_batch_response, sample_results_jsonl
    ):
        """Test results retrieval with raw JSONL included."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        status_response = MagicMock()
        status_response.raise_for_status = MagicMock()
        status_response.json.return_value = sample_completed_batch_response

        results_response = MagicMock()
        results_response.raise_for_status = MagicMock()
        results_response.text = sample_results_jsonl

        mock_httpx_client.get.side_effect = [status_response, results_response]

        result = await adapter.execute(
            "get_batch_results",
            batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
            include_raw=True
        )

        assert result.success is True
        assert "raw_jsonl" in result.data
        assert len(result.data["raw_jsonl"]) > 0


# =============================================================================
# Batch Cancellation Tests
# =============================================================================

class TestBatchCancellation:
    """Tests for batch cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_batch_success(self, mock_httpx_client):
        """Test successful batch cancellation."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        cancel_response = {
            "id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
            "type": "message_batch",
            "processing_status": "canceling",
            "request_counts": {"processing": 2, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
            "created_at": "2024-09-24T18:37:24.100435Z",
            "expires_at": "2024-09-25T18:37:24.100435Z",
            "cancel_initiated_at": "2024-09-24T18:39:03.114875Z",
            "results_url": None,
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = cancel_response
        mock_httpx_client.post.return_value = mock_response

        result = await adapter.execute("cancel_batch", batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d")

        assert result.success is True
        assert result.data["processing_status"] == "canceling"
        assert result.data["cancel_initiated"] is True

    @pytest.mark.asyncio
    async def test_cancel_batch_missing_id(self, mock_httpx_client):
        """Test cancel_batch fails without batch_id."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("cancel_batch")
        assert result.success is False
        assert "batch_id required" in result.error

    @pytest.mark.asyncio
    async def test_cancel_batch_already_ended(self, mock_httpx_client):
        """Test cancel_batch handles already ended batch."""
        import httpx

        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.status_code = 409
        error = httpx.HTTPStatusError("Conflict", request=MagicMock(), response=mock_response)
        mock_httpx_client.post.side_effect = error

        result = await adapter.execute("cancel_batch", batch_id="msgbatch_ended")
        assert result.success is False
        assert "cannot be canceled" in result.error.lower()


# =============================================================================
# List Batches Tests
# =============================================================================

class TestListBatches:
    """Tests for listing batches."""

    @pytest.mark.asyncio
    async def test_list_batches_success(self, mock_httpx_client, sample_batch_response):
        """Test successful batch listing."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        list_response = {
            "data": [sample_batch_response],
            "has_more": False,
            "first_id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
            "last_id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = list_response
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute("list_batches", limit=20)

        assert result.success is True
        assert result.data["count"] == 1
        assert result.data["has_more"] is False

    @pytest.mark.asyncio
    async def test_list_batches_with_pagination(self, mock_httpx_client, sample_batch_response):
        """Test batch listing with pagination."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        list_response = {
            "data": [sample_batch_response],
            "has_more": True,
            "first_id": "msgbatch_first",
            "last_id": "msgbatch_last",
        }

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = list_response
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute("list_batches", after_id="msgbatch_previous")

        assert result.success is True
        assert result.data["has_more"] is True
        mock_httpx_client.get.assert_called_once()


# =============================================================================
# Polling Tests
# =============================================================================

class TestBatchPolling:
    """Tests for batch polling."""

    @pytest.mark.asyncio
    async def test_poll_until_complete_immediate(self, mock_httpx_client, sample_completed_batch_response):
        """Test polling that completes immediately."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_completed_batch_response
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute(
            "poll_until_complete",
            batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
            poll_interval=0.1
        )

        assert result.success is True
        assert result.data["completed"] is True
        assert result.data["poll_count"] == 0

    @pytest.mark.asyncio
    async def test_poll_until_complete_multiple_polls(self, mock_httpx_client, sample_batch_response, sample_completed_batch_response):
        """Test polling that requires multiple polls."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        # First call returns in_progress, second returns ended
        response1 = MagicMock()
        response1.raise_for_status = MagicMock()
        response1.json.return_value = sample_batch_response

        response2 = MagicMock()
        response2.raise_for_status = MagicMock()
        response2.json.return_value = sample_completed_batch_response

        mock_httpx_client.get.side_effect = [response1, response2]

        result = await adapter.execute(
            "poll_until_complete",
            batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
            poll_interval=0.1
        )

        assert result.success is True
        assert result.data["completed"] is True
        assert result.data["poll_count"] == 1

    @pytest.mark.asyncio
    async def test_poll_until_complete_timeout(self, mock_httpx_client, sample_batch_response):
        """Test polling that times out."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = sample_batch_response  # Always in_progress
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.execute(
            "poll_until_complete",
            batch_id="msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
            poll_interval=0.05,
            max_wait=0.1
        )

        assert result.success is False
        assert "timeout" in result.error.lower()


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization(self):
        """Test execute fails when not initialized."""
        adapter = AnthropicBatchAdapter()
        result = await adapter.execute("list_batches")
        assert result.success is False
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mock_httpx_client):
        """Test handling unknown operation."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.execute("unknown_operation")
        assert result.success is False
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_operation_timeout(self, mock_httpx_client):
        """Test operation timeout handling."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(10)
            return MagicMock()

        mock_httpx_client.get = slow_operation

        result = await adapter.execute("list_batches", timeout=0.1)
        assert result.success is False
        assert "timed out" in result.error.lower()


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_httpx_client):
        """Test health check when healthy."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True
        adapter._base_url = "https://api.anthropic.com/v1"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_httpx_client.get.return_value = mock_response

        result = await adapter.health_check()

        assert result.success is True
        assert result.data["status"] == "healthy"
        assert "50% cost savings" in result.data["features"]

    @pytest.mark.asyncio
    async def test_health_check_without_client(self):
        """Test health check without initialized client."""
        adapter = AnthropicBatchAdapter()
        result = await adapter.health_check()
        assert result.success is False
        assert "not initialized" in result.error.lower()


# =============================================================================
# Shutdown Tests
# =============================================================================

class TestShutdown:
    """Tests for adapter shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_httpx_client):
        """Test adapter shutdown."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        result = await adapter.shutdown()

        assert result.success is True
        assert result.data["status"] == "shutdown"
        assert adapter.available is False
        mock_httpx_client.aclose.assert_called_once()


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_max_batch_requests(self):
        """Test MAX_BATCH_REQUESTS constant."""
        assert MAX_BATCH_REQUESTS == 100_000

    def test_max_batch_size_bytes(self):
        """Test MAX_BATCH_SIZE_BYTES constant."""
        assert MAX_BATCH_SIZE_BYTES == 256 * 1024 * 1024  # 256 MB


# =============================================================================
# Integration-style Tests (mocked)
# =============================================================================

class TestIntegrationScenarios:
    """Integration-style tests for common workflows."""

    @pytest.mark.asyncio
    async def test_full_batch_workflow(
        self, mock_httpx_client, sample_batch_requests, sample_batch_response,
        sample_completed_batch_response, sample_results_jsonl
    ):
        """Test complete batch workflow: create -> poll -> get results."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        # Step 1: Create batch
        create_response = MagicMock()
        create_response.raise_for_status = MagicMock()
        create_response.json.return_value = sample_batch_response
        mock_httpx_client.post.return_value = create_response

        create_result = await adapter.execute("create_batch", requests=sample_batch_requests)
        assert create_result.success is True
        batch_id = create_result.data["batch_id"]

        # Step 2: Poll until complete (simulate immediate completion)
        status_response = MagicMock()
        status_response.raise_for_status = MagicMock()
        status_response.json.return_value = sample_completed_batch_response

        results_response = MagicMock()
        results_response.raise_for_status = MagicMock()
        results_response.text = sample_results_jsonl

        mock_httpx_client.get.side_effect = [status_response, status_response, results_response]

        poll_result = await adapter.execute("poll_until_complete", batch_id=batch_id, poll_interval=0.01)
        assert poll_result.success is True

        # Step 3: Get results
        mock_httpx_client.get.side_effect = [status_response, results_response]
        results_result = await adapter.execute("get_batch_results", batch_id=batch_id)
        assert results_result.success is True
        assert results_result.data["total_count"] == 3

    @pytest.mark.asyncio
    async def test_batch_with_cancellation(self, mock_httpx_client, sample_batch_requests, sample_batch_response):
        """Test batch workflow with cancellation."""
        adapter = AnthropicBatchAdapter()
        adapter._client = mock_httpx_client
        adapter._available = True

        # Step 1: Create batch
        create_response = MagicMock()
        create_response.raise_for_status = MagicMock()
        create_response.json.return_value = sample_batch_response
        mock_httpx_client.post.return_value = create_response

        create_result = await adapter.execute("create_batch", requests=sample_batch_requests)
        assert create_result.success is True
        batch_id = create_result.data["batch_id"]

        # Step 2: Cancel batch
        cancel_response = {
            **sample_batch_response,
            "processing_status": "canceling",
            "cancel_initiated_at": "2024-09-24T18:39:03.114875Z",
        }
        cancel_mock = MagicMock()
        cancel_mock.raise_for_status = MagicMock()
        cancel_mock.json.return_value = cancel_response
        mock_httpx_client.post.return_value = cancel_mock

        cancel_result = await adapter.execute("cancel_batch", batch_id=batch_id)
        assert cancel_result.success is True
        assert cancel_result.data["processing_status"] == "canceling"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
