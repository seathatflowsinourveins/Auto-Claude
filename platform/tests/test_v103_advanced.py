#!/usr/bin/env python3
"""
Test Suite for V10.3 Advanced MCP Features

Tests comprehensive MCP support based on official documentation:
- MCP Elicitation: Form mode and URL mode for user input collection
- MCP Sampling: Server-initiated LLM calls with tool use support
- MCP Progress: Progress notifications for long-running operations
- MCP Capabilities: Client capability negotiation
- MCP Subscriptions: Resource subscription patterns
- Knowledge Graph: Entity-relation-observation storage

Run with: python -m pytest tests/test_v103_advanced.py -v
"""

import json
import sys
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # V10.3 Elicitation
    ElicitationMode,
    ElicitationAction,
    ElicitationRequest,
    ElicitationResponse,
    # V10.3 Sampling
    ToolChoiceMode,
    SamplingMessage,
    ModelPreferences,
    SamplingTool,
    SamplingRequest,
    SamplingResponse,
    # V10.3 Progress
    ProgressNotification,
    # V10.3 Capabilities
    MCPCapabilities,
    # V10.3 Subscriptions
    ResourceSubscription,
    SubscriptionManager,
    # V10.3 Knowledge Graph
    Entity,
    Relation,
    KnowledgeGraph,
)


# =============================================================================
# ELICITATION TESTS
# =============================================================================


class TestElicitationMode:
    """Test ElicitationMode enum values per MCP spec."""

    def test_mode_values(self):
        """Verify elicitation mode values."""
        assert ElicitationMode.FORM.value == "form"
        assert ElicitationMode.URL.value == "url"


class TestElicitationAction:
    """Test ElicitationAction enum values per MCP spec."""

    def test_action_values(self):
        """Verify elicitation action values."""
        assert ElicitationAction.ACCEPT.value == "accept"
        assert ElicitationAction.DECLINE.value == "decline"
        assert ElicitationAction.CANCEL.value == "cancel"


class TestElicitationRequest:
    """Test ElicitationRequest class for user input collection."""

    def test_form_mode_creation(self):
        """Test creating form mode elicitation request."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name"]
        }
        request = ElicitationRequest.form(
            message="Please provide your info",
            schema=schema
        )

        assert request.mode == ElicitationMode.FORM
        assert request.message == "Please provide your info"
        assert request.schema == schema
        assert request.timeout_ms == 600000  # Default 10 min

    def test_url_mode_creation(self):
        """Test creating URL mode elicitation request."""
        request = ElicitationRequest.url_mode(
            message="Complete OAuth login",
            url="https://auth.example.com/oauth",
            description="Login to your account"
        )

        assert request.mode == ElicitationMode.URL
        assert request.message == "Complete OAuth login"
        assert request.redirect_url == "https://auth.example.com/oauth"
        assert request.description == "Login to your account"

    def test_form_to_dict(self):
        """Test form mode serialization to MCP format."""
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        request = ElicitationRequest.form(
            message="Enter your age",
            schema=schema
        )
        result = request.to_dict()

        assert result["message"] == "Enter your age"
        assert result["requestedSchema"] == schema

    def test_url_to_dict(self):
        """Test URL mode serialization to MCP format."""
        request = ElicitationRequest.url_mode(
            message="Make payment",
            url="https://pay.example.com/checkout",
            description="Secure payment"
        )
        result = request.to_dict()

        assert result["message"] == "Make payment"
        assert result["url"] == "https://pay.example.com/checkout"
        assert result["description"] == "Secure payment"


class TestElicitationResponse:
    """Test ElicitationResponse class for user input results."""

    def test_accept_response(self):
        """Test parsing accepted elicitation response."""
        data = {
            "action": "accept",
            "content": {"name": "Alice", "email": "alice@example.com"}
        }
        response = ElicitationResponse.from_dict(data)

        assert response.action == ElicitationAction.ACCEPT
        assert response.content == {"name": "Alice", "email": "alice@example.com"}
        assert response.accepted is True
        assert response.declined is False
        assert response.cancelled is False

    def test_decline_response(self):
        """Test parsing declined elicitation response."""
        data = {"action": "decline"}
        response = ElicitationResponse.from_dict(data)

        assert response.action == ElicitationAction.DECLINE
        assert response.content is None
        assert response.accepted is False
        assert response.declined is True
        assert response.cancelled is False

    def test_cancel_response(self):
        """Test parsing cancelled elicitation response."""
        data = {"action": "cancel"}
        response = ElicitationResponse.from_dict(data)

        assert response.action == ElicitationAction.CANCEL
        assert response.content is None
        assert response.cancelled is True


# =============================================================================
# SAMPLING TESTS
# =============================================================================


class TestToolChoiceMode:
    """Test ToolChoiceMode enum values per MCP spec."""

    def test_tool_choice_values(self):
        """Verify tool choice mode values."""
        assert ToolChoiceMode.AUTO.value == "auto"
        assert ToolChoiceMode.REQUIRED.value == "required"
        assert ToolChoiceMode.NONE.value == "none"


class TestSamplingMessage:
    """Test SamplingMessage class for LLM sampling requests."""

    def test_user_message(self):
        """Test creating user message."""
        msg = SamplingMessage.user("Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"

    def test_assistant_message(self):
        """Test creating assistant message."""
        msg = SamplingMessage.assistant("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_to_dict_text_content(self):
        """Test serialization with text content."""
        msg = SamplingMessage.user("What is 2+2?")
        result = msg.to_dict()

        assert result["role"] == "user"
        assert result["content"] == {"type": "text", "text": "What is 2+2?"}

    def test_to_dict_complex_content(self):
        """Test serialization with pre-formatted content."""
        content = {"type": "image", "data": "base64...", "mimeType": "image/png"}
        msg = SamplingMessage(role="user", content=content)
        result = msg.to_dict()

        assert result["content"] == content


class TestModelPreferences:
    """Test ModelPreferences class for model selection hints."""

    def test_default_preferences(self):
        """Test default model preferences."""
        prefs = ModelPreferences()
        result = prefs.to_dict()

        # Defaults shouldn't be included when equal to 0.5
        assert "costPriority" not in result
        assert "speedPriority" not in result

    def test_prefer_claude(self):
        """Test Claude model hints."""
        prefs = ModelPreferences.prefer_claude("opus")
        result = prefs.to_dict()

        assert result["hints"] == [{"name": "claude-3-opus"}, {"name": "claude"}]

    def test_prefer_fast(self):
        """Test speed-prioritized preferences."""
        prefs = ModelPreferences.prefer_fast()

        assert prefs.speed_priority == 0.9
        assert prefs.intelligence_priority == 0.3

    def test_prefer_smart(self):
        """Test intelligence-prioritized preferences."""
        prefs = ModelPreferences.prefer_smart()

        assert prefs.intelligence_priority == 0.9
        assert prefs.speed_priority == 0.3


class TestSamplingTool:
    """Test SamplingTool class for tool definitions in sampling."""

    def test_tool_creation(self):
        """Test creating a sampling tool."""
        tool = SamplingTool(
            name="get_weather",
            description="Get weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        )

        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a city"

    def test_tool_to_dict(self):
        """Test tool serialization to MCP format."""
        tool = SamplingTool(
            name="calculate",
            description="Perform calculation",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}}
        )
        result = tool.to_dict()

        assert result["name"] == "calculate"
        assert result["description"] == "Perform calculation"
        assert "inputSchema" in result


class TestSamplingRequest:
    """Test SamplingRequest class for server-initiated LLM calls."""

    def test_basic_request(self):
        """Test basic sampling request without tools."""
        request = SamplingRequest(
            messages=[SamplingMessage.user("Write a haiku")],
            max_tokens=100
        )
        result = request.to_dict()

        assert result["maxTokens"] == 100
        assert len(result["messages"]) == 1
        assert "tools" not in result

    def test_request_with_tools(self):
        """Test sampling request with tool definitions."""
        tool = SamplingTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}}
        )
        request = SamplingRequest(
            messages=[SamplingMessage.user("Search for cats")],
            max_tokens=500,
            tools=[tool],
            tool_choice=ToolChoiceMode.AUTO
        )
        result = request.to_dict()

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search"
        assert result["toolChoice"] == {"mode": "auto"}

    def test_request_with_system_prompt(self):
        """Test sampling request with system prompt."""
        request = SamplingRequest(
            messages=[SamplingMessage.user("Hello")],
            max_tokens=200,
            system_prompt="You are a helpful assistant."
        )
        result = request.to_dict()

        assert result["systemPrompt"] == "You are a helpful assistant."

    def test_request_with_model_preferences(self):
        """Test sampling request with model preferences."""
        request = SamplingRequest(
            messages=[SamplingMessage.user("Complex task")],
            max_tokens=1000,
            model_preferences=ModelPreferences.prefer_smart()
        )
        result = request.to_dict()

        assert "modelPreferences" in result
        assert result["modelPreferences"]["intelligencePriority"] == 0.9


class TestSamplingResponse:
    """Test SamplingResponse class for sampling results."""

    def test_text_response(self):
        """Test parsing text response."""
        data = {
            "role": "assistant",
            "content": {"type": "text", "text": "Here's a haiku..."},
            "model": "claude-3-sonnet",
            "stopReason": "endTurn"
        }
        response = SamplingResponse.from_dict(data)

        assert response.role == "assistant"
        assert response.model == "claude-3-sonnet"
        assert response.stop_reason == "endTurn"
        assert response.is_tool_use is False
        assert "haiku" in response.text

    def test_tool_use_response(self):
        """Test parsing tool use response."""
        data = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"city": "Paris"}}
            ],
            "model": "claude-3-sonnet",
            "stopReason": "toolUse"
        }
        response = SamplingResponse.from_dict(data)

        assert response.is_tool_use is True
        tool_uses = response.get_tool_uses()
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "get_weather"

    def test_string_content_response(self):
        """Test parsing string content response."""
        data = {
            "role": "assistant",
            "content": "Simple text response",
            "model": "claude-3-haiku",
            "stopReason": "endTurn"
        }
        response = SamplingResponse.from_dict(data)

        assert response.text == "Simple text response"


# =============================================================================
# PROGRESS TESTS
# =============================================================================


class TestProgressNotification:
    """Test ProgressNotification class for long-running operations."""

    def test_progress_creation(self):
        """Test creating progress notification."""
        progress = ProgressNotification(
            progress_token="op_12345",
            progress=50,
            total=100,
            message="Processing file 5 of 10"
        )

        assert progress.progress_token == "op_12345"
        assert progress.progress == 50
        assert progress.total == 100
        assert progress.percentage == 50.0

    def test_to_notification(self):
        """Test serialization to MCP notification format."""
        progress = ProgressNotification(
            progress_token="op_abc",
            progress=75,
            total=100,
            message="Almost done..."
        )
        result = progress.to_notification()

        assert result["method"] == "notifications/progress"
        assert result["params"]["progressToken"] == "op_abc"
        assert result["params"]["progress"] == 75
        assert result["params"]["total"] == 100
        assert result["params"]["message"] == "Almost done..."

    def test_percentage_without_total(self):
        """Test percentage when total is None."""
        progress = ProgressNotification(
            progress_token="op_xyz",
            progress=100
        )

        assert progress.percentage is None


# =============================================================================
# CAPABILITIES TESTS
# =============================================================================


class TestMCPCapabilities:
    """Test MCPCapabilities class for feature negotiation."""

    def test_no_capabilities(self):
        """Test parsing empty capabilities."""
        caps = MCPCapabilities.from_dict({})

        assert caps.sampling is False
        assert caps.sampling_tools is False
        assert caps.elicitation is False
        assert caps.roots is False

    def test_basic_sampling(self):
        """Test parsing basic sampling capability."""
        data = {"sampling": {}}
        caps = MCPCapabilities.from_dict(data)

        assert caps.sampling is True
        assert caps.supports_sampling() is True
        assert caps.sampling_tools is False

    def test_sampling_with_tools(self):
        """Test parsing sampling with tools capability."""
        data = {"sampling": {"tools": {}}}
        caps = MCPCapabilities.from_dict(data)

        assert caps.sampling is True
        assert caps.sampling_tools is True
        assert caps.supports_sampling_tools() is True

    def test_elicitation_capability(self):
        """Test parsing elicitation capability."""
        data = {"elicitation": {}}
        caps = MCPCapabilities.from_dict(data)

        assert caps.elicitation is True
        assert caps.supports_elicitation() is True

    def test_full_capabilities(self):
        """Test parsing all capabilities."""
        data = {
            "sampling": {"tools": {}},
            "elicitation": {},
            "roots": {"listChanged": True}
        }
        caps = MCPCapabilities.from_dict(data)

        assert caps.sampling is True
        assert caps.sampling_tools is True
        assert caps.elicitation is True
        assert caps.roots is True


# =============================================================================
# SUBSCRIPTION TESTS
# =============================================================================


class TestResourceSubscription:
    """Test ResourceSubscription class for resource update patterns."""

    def test_subscription_creation(self):
        """Test creating resource subscription."""
        sub = ResourceSubscription(uri="file:///project/config.json")

        assert sub.uri == "file:///project/config.json"

    def test_subscribe_request(self):
        """Test generating subscribe request."""
        sub = ResourceSubscription(uri="file:///data.json")
        result = sub.to_subscribe_request()

        assert result["method"] == "resources/subscribe"
        assert result["params"]["uri"] == "file:///data.json"

    def test_unsubscribe_request(self):
        """Test generating unsubscribe request."""
        sub = ResourceSubscription(uri="file:///data.json")
        result = sub.to_unsubscribe_request()

        assert result["method"] == "resources/unsubscribe"
        assert result["params"]["uri"] == "file:///data.json"


class TestSubscriptionManager:
    """Test SubscriptionManager class for managing subscriptions."""

    def test_subscribe(self):
        """Test subscribing to a resource."""
        manager = SubscriptionManager()
        manager.subscribe("file:///config.json", "session_1")

        assert manager.has_subscribers("file:///config.json")
        assert "session_1" in manager.get_subscribers("file:///config.json")

    def test_multiple_subscribers(self):
        """Test multiple sessions subscribing to same resource."""
        manager = SubscriptionManager()
        manager.subscribe("file:///data.json", "session_1")
        manager.subscribe("file:///data.json", "session_2")

        subscribers = manager.get_subscribers("file:///data.json")
        assert len(subscribers) == 2
        assert "session_1" in subscribers
        assert "session_2" in subscribers

    def test_unsubscribe(self):
        """Test unsubscribing from a resource."""
        manager = SubscriptionManager()
        manager.subscribe("file:///config.json", "session_1")
        manager.unsubscribe("file:///config.json", "session_1")

        assert not manager.has_subscribers("file:///config.json")

    def test_get_subscribed_uris(self):
        """Test getting all subscribed URIs for a session."""
        manager = SubscriptionManager()
        manager.subscribe("file:///a.json", "session_1")
        manager.subscribe("file:///b.json", "session_1")
        manager.subscribe("file:///c.json", "session_2")

        uris = manager.get_subscribed_uris("session_1")
        assert "file:///a.json" in uris
        assert "file:///b.json" in uris
        assert "file:///c.json" not in uris

    def test_update_notification(self):
        """Test creating update notification."""
        manager = SubscriptionManager()
        notification = manager.create_update_notification("file:///changed.json")

        assert notification["method"] == "notifications/resources/updated"
        assert notification["params"]["uri"] == "file:///changed.json"


# =============================================================================
# KNOWLEDGE GRAPH TESTS
# =============================================================================


class TestEntity:
    """Test Entity class for knowledge graph nodes."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            name="Claude",
            entity_type="ai_assistant",
            observations=["Created by Anthropic", "Released in 2023"]
        )

        assert entity.name == "Claude"
        assert entity.entity_type == "ai_assistant"
        assert len(entity.observations) == 2

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(name="Python", entity_type="language", observations=["Popular"])
        result = entity.to_dict()

        assert result["type"] == "entity"
        assert result["name"] == "Python"
        assert result["entityType"] == "language"

    def test_entity_from_dict(self):
        """Test entity deserialization."""
        data = {
            "name": "JavaScript",
            "entityType": "language",
            "observations": ["Web language", "Created in 1995"]
        }
        entity = Entity.from_dict(data)

        assert entity.name == "JavaScript"
        assert entity.entity_type == "language"
        assert "Web language" in entity.observations


class TestRelation:
    """Test Relation class for knowledge graph edges."""

    def test_relation_creation(self):
        """Test creating a relation."""
        relation = Relation(
            from_entity="Claude",
            to_entity="Anthropic",
            relation_type="created_by"
        )

        assert relation.from_entity == "Claude"
        assert relation.to_entity == "Anthropic"
        assert relation.relation_type == "created_by"

    def test_relation_to_dict(self):
        """Test relation serialization."""
        relation = Relation("Python", "Guido", "invented_by")
        result = relation.to_dict()

        assert result["type"] == "relation"
        assert result["from"] == "Python"
        assert result["to"] == "Guido"
        assert result["relationType"] == "invented_by"

    def test_relation_from_dict(self):
        """Test relation deserialization."""
        data = {
            "from": "Flask",
            "to": "Python",
            "relationType": "built_with"
        }
        relation = Relation.from_dict(data)

        assert relation.from_entity == "Flask"
        assert relation.to_entity == "Python"


class TestKnowledgeGraph:
    """Test KnowledgeGraph class for entity-relation storage."""

    def test_add_entity(self):
        """Test adding entities to graph."""
        graph = KnowledgeGraph()
        entity = Entity("Claude", "ai", ["Helpful assistant"])

        assert graph.add_entity(entity) is True
        assert len(graph.entities) == 1

        # Should not add duplicate
        assert graph.add_entity(entity) is False
        assert len(graph.entities) == 1

    def test_add_relation(self):
        """Test adding relations to graph."""
        graph = KnowledgeGraph()
        relation = Relation("A", "B", "connects_to")

        assert graph.add_relation(relation) is True
        assert len(graph.relations) == 1

        # Should not add duplicate
        assert graph.add_relation(relation) is False
        assert len(graph.relations) == 1

    def test_add_observation(self):
        """Test adding observation to entity."""
        graph = KnowledgeGraph()
        entity = Entity("Claude", "ai", ["Helpful"])
        graph.add_entity(entity)

        assert graph.add_observation("Claude", "Safe") is True
        assert "Safe" in graph.entities[0].observations

        # Should not add duplicate observation
        assert graph.add_observation("Claude", "Safe") is False

        # Should return False for non-existent entity
        assert graph.add_observation("GPT", "Smart") is False

    def test_search(self):
        """Test searching the knowledge graph."""
        graph = KnowledgeGraph()
        graph.add_entity(Entity("Claude", "ai", ["Made by Anthropic"]))
        graph.add_entity(Entity("GPT", "ai", ["Made by OpenAI"]))
        graph.add_relation(Relation("Claude", "GPT", "competes_with"))

        # Search by name
        results = graph.search("Claude")
        assert len(results.entities) == 1
        assert results.entities[0].name == "Claude"

        # Search by observation
        results = graph.search("Anthropic")
        assert len(results.entities) == 1

        # Search for both entities
        results = graph.search("ai")
        assert len(results.entities) == 2

    def test_jsonl_serialization(self):
        """Test JSONL serialization and deserialization."""
        graph = KnowledgeGraph()
        graph.add_entity(Entity("A", "type1", ["observation1"]))
        graph.add_entity(Entity("B", "type2", ["observation2"]))
        graph.add_relation(Relation("A", "B", "connects"))

        # Serialize to JSONL
        jsonl = graph.to_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) == 3  # 2 entities + 1 relation

        # Deserialize from JSONL
        restored = KnowledgeGraph.from_jsonl(jsonl)
        assert len(restored.entities) == 2
        assert len(restored.relations) == 1
        assert restored.entities[0].name == "A"
        assert restored.relations[0].relation_type == "connects"

    def test_empty_jsonl(self):
        """Test handling empty JSONL."""
        graph = KnowledgeGraph.from_jsonl("")
        assert len(graph.entities) == 0
        assert len(graph.relations) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
