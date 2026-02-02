#!/usr/bin/env python3
"""
V10.11 MCP Infrastructure Tests

Tests for FastMCP, Grafana, Qdrant, Postgres MCP Pro, and Enhanced Sequential Thinking patterns.
"""

import pytest
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import sys
import os

# Add the hooks directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.hook_utils import (
    # FastMCP Server Patterns
    FastMCPServer,
    FastMCPTool,
    FastMCPResource,
    LifespanContext,
    MCPContext,
    # Grafana Observability
    DashboardPanelType,
    GrafanaPanel,
    GrafanaDashboard,
    PrometheusQuery,
    LokiQuery,
    AlertState,
    GrafanaAlert,
    IncidentSeverity,
    IncidentStatus,
    IncidentActivity,
    GrafanaIncident,
    # Qdrant Vector Patterns
    DistanceMetric,
    VectorConfig,
    QdrantCollection,
    QdrantPoint,
    VectorSearchResult,
    EmbeddingProvider,
    EmbeddingModel,
    SemanticSearch,
    # Postgres MCP Pro
    DatabaseAccessMode,
    QueryType,
    QueryExecution,
    PgIndexType,
    IndexRecommendation,
    HealthStatus,
    DatabaseHealth,
    ExplainFormat,
    ExplainAnalysis,
    # Enhanced Sequential Thinking
    ThoughtType,
    ThoughtBranch,
    EnhancedThought,
    BranchingThinkingSession,
)


# =============================================================================
# FastMCP Server Pattern Tests
# =============================================================================

class TestFastMCPServer:
    """Tests for FastMCP server implementation."""

    def test_server_creation(self):
        """Test FastMCP server creation."""
        server = FastMCPServer(
            name="test-server",
            version="1.0.0",
            description="Test server"
        )
        assert server.name == "test-server"
        assert server.version == "1.0.0"
        assert server.description == "Test server"

    def test_tool_decorator(self):
        """Test tool registration via decorator."""
        server = FastMCPServer("test-server")

        @server.tool()
        def my_tool(x: int) -> str:
            return f"Result: {x}"

        assert "my_tool" in server.list_tools()
        assert server._tools["my_tool"].name == "my_tool"

    def test_tool_with_custom_name(self):
        """Test tool with custom name."""
        server = FastMCPServer("test-server")

        @server.tool(name="custom_tool", description="Custom description")
        def original_name():
            pass

        assert "custom_tool" in server.list_tools()
        assert server._tools["custom_tool"].description == "Custom description"

    def test_resource_decorator(self):
        """Test resource registration via decorator."""
        server = FastMCPServer("test-server")

        @server.resource(uri="file:///test.txt", name="test_file")
        def read_test():
            return "content"

        assert "file:///test.txt" in server.list_resources()

    def test_prompt_decorator(self):
        """Test prompt registration via decorator."""
        server = FastMCPServer("test-server")

        @server.prompt(name="greeting")
        def make_greeting(name: str):
            return f"Hello, {name}!"

        assert "greeting" in server.list_prompts()

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test calling a registered tool."""
        server = FastMCPServer("test-server")

        @server.tool()
        def add(a: int, b: int) -> int:
            return a + b

        result = await server.call_tool("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test calling unknown tool raises error."""
        server = FastMCPServer("test-server")

        with pytest.raises(ValueError, match="Unknown tool"):
            await server.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_call_async_tool(self):
        """Test calling async tool."""
        server = FastMCPServer("test-server")

        @server.tool()
        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)
            return a + b

        result = await server.call_tool("async_add", {"a": 5, "b": 7})
        assert result == 12


class TestFastMCPTool:
    """Tests for FastMCP tool definition."""

    def test_tool_creation(self):
        """Test tool creation."""
        tool = FastMCPTool(
            name="test_tool",
            description="A test tool",
            handler=lambda x: x * 2
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_to_schema(self):
        """Test schema generation."""
        tool = FastMCPTool(
            name="calculate",
            description="Calculate something",
            handler=lambda: None
        )
        schema = tool.to_schema()
        assert schema["name"] == "calculate"
        assert schema["description"] == "Calculate something"
        assert "inputSchema" in schema

    @pytest.mark.asyncio
    async def test_execute_sync_handler(self):
        """Test executing sync handler."""
        tool = FastMCPTool(
            name="sync_tool",
            description="Sync tool",
            handler=lambda x: x * 2
        )
        result = await tool.execute({"x": 5})
        assert result == 10


class TestFastMCPResource:
    """Tests for FastMCP resource definition."""

    def test_resource_creation(self):
        """Test resource creation."""
        resource = FastMCPResource(
            uri="file:///data.json",
            name="data",
            description="Data file",
            handler=lambda: '{"key": "value"}',
            mime_type="application/json"
        )
        assert resource.uri == "file:///data.json"
        assert resource.mime_type == "application/json"

    def test_to_schema(self):
        """Test schema generation."""
        resource = FastMCPResource(
            uri="file:///test.txt",
            name="test",
            description="Test file",
            handler=lambda: "content"
        )
        schema = resource.to_schema()
        assert schema["uri"] == "file:///test.txt"
        assert schema["mimeType"] == "text/plain"


class TestLifespanContext:
    """Tests for FastMCP lifespan context."""

    def test_state_management(self):
        """Test lifespan state management."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)

        lifespan.set("db", "connection")
        assert lifespan.get("db") == "connection"
        assert lifespan.get("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)

        async with lifespan as ctx:
            ctx.set("key", "value")
            assert ctx.get("key") == "value"

        # State cleared after exit
        assert lifespan.get("key") is None


class TestMCPContext:
    """Tests for MCP execution context."""

    def test_context_creation(self):
        """Test context creation."""
        server = FastMCPServer("test")
        context = MCPContext(
            request_id="req-123",
            server=server
        )
        assert context.request_id == "req-123"
        assert context.server == server

    def test_lifespan_state_access(self):
        """Test accessing lifespan state."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)
        lifespan.set("config", {"debug": True})

        context = MCPContext(
            request_id="req-456",
            server=server,
            lifespan=lifespan
        )
        assert context.get_lifespan_state("config") == {"debug": True}

    def test_lifespan_state_without_lifespan(self):
        """Test accessing state without lifespan."""
        server = FastMCPServer("test")
        context = MCPContext(request_id="req-789", server=server)
        assert context.get_lifespan_state("anything") is None


# =============================================================================
# Grafana Observability Tests
# =============================================================================

class TestDashboardPanelType:
    """Tests for Grafana panel types."""

    def test_panel_types(self):
        """Test panel type values."""
        assert DashboardPanelType.GRAPH.value == "graph"
        assert DashboardPanelType.TIMESERIES.value == "timeseries"
        assert DashboardPanelType.LOGS.value == "logs"


class TestGrafanaPanel:
    """Tests for Grafana dashboard panels."""

    def test_panel_creation(self):
        """Test panel creation."""
        panel = GrafanaPanel(
            id=1,
            title="CPU Usage",
            panel_type=DashboardPanelType.TIMESERIES
        )
        assert panel.id == 1
        assert panel.title == "CPU Usage"
        assert panel.panel_type == DashboardPanelType.TIMESERIES

    def test_to_json(self):
        """Test JSON serialization."""
        panel = GrafanaPanel(
            id=2,
            title="Memory",
            panel_type=DashboardPanelType.GAUGE,
            grid_pos={"x": 0, "y": 8, "w": 6, "h": 4}
        )
        json_data = panel.to_json()
        assert json_data["id"] == 2
        assert json_data["type"] == "gauge"
        assert json_data["gridPos"]["w"] == 6


class TestGrafanaDashboard:
    """Tests for Grafana dashboards."""

    def test_dashboard_creation(self):
        """Test dashboard creation."""
        dashboard = GrafanaDashboard(
            uid="abc123",
            title="System Metrics"
        )
        assert dashboard.uid == "abc123"
        assert dashboard.title == "System Metrics"

    def test_add_panel(self):
        """Test adding panels."""
        dashboard = GrafanaDashboard(uid="test", title="Test")
        panel = GrafanaPanel(id=1, title="Panel 1", panel_type=DashboardPanelType.GRAPH)
        dashboard.add_panel(panel)
        assert len(dashboard.panels) == 1

    def test_to_json(self):
        """Test JSON serialization with panels."""
        dashboard = GrafanaDashboard(
            uid="metrics",
            title="Metrics Dashboard",
            tags=["system", "monitoring"]
        )
        dashboard.add_panel(GrafanaPanel(id=1, title="CPU", panel_type=DashboardPanelType.STAT))

        json_data = dashboard.to_json()
        assert json_data["uid"] == "metrics"
        assert len(json_data["panels"]) == 1
        assert "system" in json_data["tags"]


class TestPrometheusQuery:
    """Tests for Prometheus query configuration."""

    def test_query_creation(self):
        """Test query creation."""
        query = PrometheusQuery(
            expr='rate(http_requests_total[5m])',
            legend_format="{{method}} {{path}}"
        )
        assert "rate" in query.expr
        assert query.legend_format == "{{method}} {{path}}"

    def test_to_target(self):
        """Test target format conversion."""
        query = PrometheusQuery(
            expr='up',
            ref_id="B",
            instant=True
        )
        target = query.to_target()
        assert target["expr"] == "up"
        assert target["refId"] == "B"
        assert target["instant"] is True
        assert target["datasource"]["type"] == "prometheus"


class TestLokiQuery:
    """Tests for Loki log queries."""

    def test_query_creation(self):
        """Test Loki query creation."""
        query = LokiQuery(
            expr='{job="api"} |= "error"',
            max_lines=500
        )
        assert query.max_lines == 500

    def test_to_target(self):
        """Test target format conversion."""
        query = LokiQuery(expr='{app="web"}', query_type="instant")
        target = query.to_target()
        assert target["datasource"]["type"] == "loki"
        assert target["queryType"] == "instant"


class TestGrafanaAlert:
    """Tests for Grafana alerting."""

    def test_alert_creation(self):
        """Test alert rule creation."""
        alert = GrafanaAlert(
            uid="alert-1",
            title="High CPU",
            condition="A",
            state=AlertState.OK
        )
        assert alert.uid == "alert-1"
        assert alert.state == AlertState.OK

    def test_to_json(self):
        """Test JSON serialization."""
        alert = GrafanaAlert(
            uid="alert-2",
            title="Memory Warning",
            condition="B",
            for_duration="10m",
            labels={"severity": "warning"}
        )
        json_data = alert.to_json()
        assert json_data["for"] == "10m"
        assert json_data["labels"]["severity"] == "warning"


class TestGrafanaIncident:
    """Tests for Grafana incident management."""

    def test_incident_creation(self):
        """Test incident creation."""
        incident = GrafanaIncident(
            id="inc-001",
            title="Database Outage",
            severity=IncidentSeverity.CRITICAL
        )
        assert incident.id == "inc-001"
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.status == IncidentStatus.OPEN

    def test_add_activity(self):
        """Test adding activity."""
        incident = GrafanaIncident(
            id="inc-002",
            title="API Slowdown",
            severity=IncidentSeverity.MAJOR
        )
        activity = IncidentActivity(
            id="act-001",
            incident_id="inc-002",
            activity_type="comment",
            body="Investigating...",
            created_at="2026-01-17T10:00:00Z"
        )
        incident.add_activity(activity)
        assert len(incident.activities) == 1

    def test_resolve(self):
        """Test resolving incident."""
        incident = GrafanaIncident(
            id="inc-003",
            title="Network Issue",
            severity=IncidentSeverity.MINOR
        )
        incident.resolve("Fixed network configuration")
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None


# =============================================================================
# Qdrant Vector Pattern Tests
# =============================================================================

class TestDistanceMetric:
    """Tests for Qdrant distance metrics."""

    def test_metric_values(self):
        """Test distance metric values."""
        assert DistanceMetric.COSINE.value == "Cosine"
        assert DistanceMetric.EUCLID.value == "Euclid"
        assert DistanceMetric.DOT.value == "Dot"


class TestVectorConfig:
    """Tests for vector configuration."""

    def test_config_creation(self):
        """Test vector config creation."""
        config = VectorConfig(
            size=768,
            distance=DistanceMetric.COSINE
        )
        assert config.size == 768
        assert config.distance == DistanceMetric.COSINE

    def test_to_json(self):
        """Test JSON serialization."""
        config = VectorConfig(size=384, distance=DistanceMetric.EUCLID, on_disk=True)
        json_data = config.to_json()
        assert json_data["size"] == 384
        assert json_data["distance"] == "Euclid"
        assert json_data["on_disk"] is True


class TestQdrantCollection:
    """Tests for Qdrant collection."""

    def test_collection_creation(self):
        """Test collection creation."""
        config = VectorConfig(size=1536, distance=DistanceMetric.COSINE)
        collection = QdrantCollection(name="embeddings", vectors_config=config)
        assert collection.name == "embeddings"
        assert collection.vectors_config.size == 1536

    def test_to_create_params(self):
        """Test creation parameters."""
        config = VectorConfig(size=768)
        collection = QdrantCollection(
            name="docs",
            vectors_config=config,
            shard_number=2,
            replication_factor=3
        )
        params = collection.to_create_params()
        assert params["shard_number"] == 2
        assert params["replication_factor"] == 3


class TestQdrantPoint:
    """Tests for Qdrant points."""

    def test_point_creation(self):
        """Test point creation."""
        point = QdrantPoint(
            id="doc-001",
            vector=[0.1, 0.2, 0.3],
            payload={"title": "Test Document"}
        )
        assert point.id == "doc-001"
        assert len(point.vector) == 3

    def test_to_json(self):
        """Test JSON serialization."""
        point = QdrantPoint(id=123, vector=[0.5, 0.5], payload={"type": "query"})
        json_data = point.to_json()
        assert json_data["id"] == 123
        assert json_data["payload"]["type"] == "query"


class TestVectorSearchResult:
    """Tests for vector search results."""

    def test_result_creation(self):
        """Test result creation."""
        result = VectorSearchResult(
            id="match-001",
            score=0.95,
            payload={"content": "Relevant document"}
        )
        assert result.score == 0.95

    def test_from_json(self):
        """Test creation from JSON."""
        data = {
            "id": "match-002",
            "score": 0.87,
            "payload": {"title": "Another doc"},
            "vector": [0.1, 0.2]
        }
        result = VectorSearchResult.from_json(data)
        assert result.id == "match-002"
        assert result.score == 0.87
        assert result.vector == [0.1, 0.2]


class TestEmbeddingModel:
    """Tests for embedding model configuration."""

    def test_model_creation(self):
        """Test model creation."""
        model = EmbeddingModel(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            dimensions=1536
        )
        assert model.provider == EmbeddingProvider.OPENAI
        assert model.dimensions == 1536

    def test_get_embedding_function(self):
        """Test getting embedding function."""
        model = EmbeddingModel(
            provider=EmbeddingProvider.FASTEMBED,
            model_name="BAAI/bge-small-en-v1.5",
            dimensions=384
        )
        embed_fn = model.get_embedding_function()
        result = embed_fn("test text")
        assert len(result) == 384


class TestSemanticSearch:
    """Tests for semantic search configuration."""

    def test_search_creation(self):
        """Test semantic search creation."""
        config = VectorConfig(size=768)
        collection = QdrantCollection(name="search", vectors_config=config)
        model = EmbeddingModel(
            provider=EmbeddingProvider.OLLAMA,
            model_name="nomic-embed-text",
            dimensions=768
        )
        search = SemanticSearch(
            collection=collection,
            embedding_model=model,
            top_k=5
        )
        assert search.top_k == 5

    def test_build_query(self):
        """Test building search query."""
        config = VectorConfig(size=3)
        collection = QdrantCollection(name="test", vectors_config=config)
        model = EmbeddingModel(
            provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            dimensions=3
        )
        search = SemanticSearch(
            collection=collection,
            embedding_model=model,
            top_k=10,
            score_threshold=0.7
        )
        query = search.build_query([0.1, 0.2, 0.3])
        assert query["limit"] == 10
        assert query["score_threshold"] == 0.7
        assert query["with_payload"] is True


# =============================================================================
# Postgres MCP Pro Tests
# =============================================================================

class TestDatabaseAccessMode:
    """Tests for database access modes."""

    def test_access_mode_values(self):
        """Test access mode values."""
        assert DatabaseAccessMode.UNRESTRICTED.value == "unrestricted"
        assert DatabaseAccessMode.READ_ONLY.value == "read_only"


class TestQueryType:
    """Tests for query types."""

    def test_query_type_values(self):
        """Test query type values."""
        assert QueryType.SELECT.value == "select"
        assert QueryType.DDL.value == "ddl"


class TestQueryExecution:
    """Tests for query execution results."""

    def test_successful_query(self):
        """Test successful query."""
        result = QueryExecution(
            query="SELECT * FROM users",
            query_type=QueryType.SELECT,
            rows=[{"id": 1, "name": "Alice"}],
            row_count=1,
            execution_time_ms=5.2
        )
        assert result.success is True
        assert result.row_count == 1

    def test_failed_query(self):
        """Test failed query."""
        result = QueryExecution(
            query="SELECT * FROM nonexistent",
            query_type=QueryType.SELECT,
            error="relation does not exist"
        )
        assert result.success is False


class TestPgIndexType:
    """Tests for PostgreSQL index types."""

    def test_index_type_values(self):
        """Test index type values."""
        assert PgIndexType.BTREE.value == "btree"
        assert PgIndexType.GIN.value == "gin"
        assert PgIndexType.BRIN.value == "brin"


class TestIndexRecommendation:
    """Tests for index recommendations."""

    def test_recommendation_creation(self):
        """Test recommendation creation."""
        rec = IndexRecommendation(
            table_name="orders",
            column_names=["customer_id", "created_at"],
            index_type=PgIndexType.BTREE,
            estimated_benefit=2.5,
            create_statement="CREATE INDEX idx_orders_customer ON orders(customer_id, created_at)",
            reason="Frequently filtered columns",
            priority=8
        )
        assert rec.estimated_benefit == 2.5
        assert rec.priority == 8

    def test_to_json(self):
        """Test JSON serialization."""
        rec = IndexRecommendation(
            table_name="products",
            column_names=["category"],
            index_type=PgIndexType.HASH,
            estimated_benefit=1.5,
            create_statement="CREATE INDEX...",
            reason="Hash lookup"
        )
        json_data = rec.to_json()
        assert json_data["indexType"] == "hash"
        assert "category" in json_data["columns"]


class TestDatabaseHealth:
    """Tests for database health status."""

    def test_health_creation(self):
        """Test health status creation."""
        health = DatabaseHealth(
            status=HealthStatus.HEALTHY,
            connection_count=50,
            max_connections=200,
            cache_hit_ratio=0.98
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.cache_hit_ratio == 0.98

    def test_connection_usage_percent(self):
        """Test connection usage calculation."""
        health = DatabaseHealth(
            status=HealthStatus.WARNING,
            connection_count=150,
            max_connections=200
        )
        assert health.connection_usage_percent == 75.0

    def test_to_json(self):
        """Test JSON serialization."""
        health = DatabaseHealth(
            status=HealthStatus.CRITICAL,
            connection_count=195,
            max_connections=200,
            blocked_queries=5,
            warnings=["High connection usage", "Blocked queries"]
        )
        json_data = health.to_json()
        assert json_data["status"] == "critical"
        assert len(json_data["warnings"]) == 2


class TestExplainAnalysis:
    """Tests for EXPLAIN ANALYZE results."""

    def test_analysis_creation(self):
        """Test analysis creation."""
        analysis = ExplainAnalysis(
            query="SELECT * FROM orders WHERE id = 1",
            plan={"Node Type": "Index Scan"},
            total_cost=5.0,
            actual_time_ms=2.5,
            rows_estimated=1,
            rows_actual=1
        )
        assert analysis.actual_time_ms == 2.5

    def test_estimation_accuracy(self):
        """Test row estimation accuracy."""
        analysis = ExplainAnalysis(
            query="SELECT...",
            plan={},
            rows_estimated=100,
            rows_actual=80
        )
        assert analysis.estimation_accuracy == 0.8

    def test_cache_hit_ratio(self):
        """Test buffer cache hit ratio."""
        analysis = ExplainAnalysis(
            query="SELECT...",
            plan={},
            shared_hit_blocks=950,
            shared_read_blocks=50
        )
        assert analysis.cache_hit_ratio == 0.95

    def test_get_slow_nodes(self):
        """Test finding slow nodes."""
        analysis = ExplainAnalysis(
            query="SELECT...",
            plan={
                "Node Type": "Seq Scan",
                "Actual Total Time": 150.0,
                "Actual Rows": 10000,
                "Plans": [
                    {"Node Type": "Sort", "Actual Total Time": 50.0, "Actual Rows": 100}
                ]
            }
        )
        slow = analysis.get_slow_nodes(threshold_ms=100.0)
        assert len(slow) == 1
        assert slow[0]["node_type"] == "Seq Scan"


# =============================================================================
# Enhanced Sequential Thinking Tests
# =============================================================================

class TestThoughtType:
    """Tests for thought types."""

    def test_thought_type_values(self):
        """Test thought type values."""
        assert ThoughtType.OBSERVATION.value == "observation"
        assert ThoughtType.REVISION.value == "revision"
        assert ThoughtType.BRANCH.value == "branch"


class TestThoughtBranch:
    """Tests for thought branches."""

    def test_branch_creation(self):
        """Test branch creation."""
        branch = ThoughtBranch(
            id="branch-001",
            parent_thought_id="thought-005",
            name="Alternative approach",
            created_at="2026-01-17T12:00:00Z"
        )
        assert branch.id == "branch-001"
        assert branch.merged is False


class TestEnhancedThought:
    """Tests for enhanced thoughts."""

    def test_thought_creation(self):
        """Test thought creation."""
        thought = EnhancedThought(
            id="thought-001",
            content="Initial observation about the problem",
            thought_type=ThoughtType.OBSERVATION,
            thought_number=1,
            total_thoughts=5
        )
        assert thought.id == "thought-001"
        assert thought.thought_type == ThoughtType.OBSERVATION

    def test_revision_thought(self):
        """Test revision thought."""
        revision = EnhancedThought(
            id="thought-002",
            content="Updated understanding",
            thought_type=ThoughtType.REVISION,
            thought_number=3,
            total_thoughts=5,
            is_revision=True,
            revises_thought_id="thought-001"
        )
        assert revision.is_revision is True
        assert revision.revises_thought_id == "thought-001"

    def test_to_json(self):
        """Test JSON serialization."""
        thought = EnhancedThought(
            id="t-003",
            content="Analysis complete",
            thought_type=ThoughtType.CONCLUSION,
            thought_number=5,
            total_thoughts=5,
            next_thought_needed=False,
            confidence=0.9
        )
        json_data = thought.to_json()
        assert json_data["thoughtType"] == "conclusion"
        assert json_data["nextThoughtNeeded"] is False
        assert json_data["confidence"] == 0.9


class TestBranchingThinkingSession:
    """Tests for branching thinking sessions."""

    def test_session_creation(self):
        """Test session creation."""
        session = BranchingThinkingSession(session_id="session-001")
        assert session.session_id == "session-001"
        assert len(session.thoughts) == 0
        assert len(session.branches) == 0

    def test_add_thought(self):
        """Test adding thoughts."""
        session = BranchingThinkingSession(session_id="session-002")
        thought = EnhancedThought(
            id="t-001",
            content="First observation",
            thought_type=ThoughtType.OBSERVATION,
            thought_number=1,
            total_thoughts=3
        )
        session.add_thought(thought)
        assert len(session.thoughts) == 1
        assert session.thoughts[0].branch_id is None

    def test_create_branch(self):
        """Test creating branches."""
        session = BranchingThinkingSession(session_id="session-003")
        thought = EnhancedThought(
            id="t-001",
            content="Initial thought",
            thought_type=ThoughtType.ANALYSIS,
            thought_number=1,
            total_thoughts=5
        )
        session.add_thought(thought)

        branch = session.create_branch("Alternative analysis", "t-001")
        assert branch.name == "Alternative analysis"
        assert branch.parent_thought_id == "t-001"
        assert len(session.branches) == 1

    def test_switch_branch(self):
        """Test switching branches."""
        session = BranchingThinkingSession(session_id="session-004")
        thought = EnhancedThought(
            id="t-001",
            content="Main thought",
            thought_type=ThoughtType.HYPOTHESIS,
            thought_number=1,
            total_thoughts=3
        )
        session.add_thought(thought)
        branch = session.create_branch("Side branch", "t-001")

        session.switch_branch(branch.id)
        assert session.current_branch_id == branch.id

    def test_get_branch_thoughts(self):
        """Test getting thoughts for a branch."""
        session = BranchingThinkingSession(session_id="session-005")

        # Add main branch thought
        thought1 = EnhancedThought(
            id="t-001",
            content="Main thought",
            thought_type=ThoughtType.OBSERVATION,
            thought_number=1,
            total_thoughts=5
        )
        session.add_thought(thought1)

        # Create and switch to branch
        branch = session.create_branch("Branch", "t-001")
        session.switch_branch(branch.id)

        # Add branch thought
        thought2 = EnhancedThought(
            id="t-002",
            content="Branch thought",
            thought_type=ThoughtType.ANALYSIS,
            thought_number=2,
            total_thoughts=5
        )
        session.add_thought(thought2)

        branch_thoughts = session.get_branch_thoughts()
        assert len(branch_thoughts) == 1
        assert branch_thoughts[0].id == "t-002"

    def test_revise_thought(self):
        """Test revising a thought."""
        session = BranchingThinkingSession(session_id="session-006")
        original = EnhancedThought(
            id="t-001",
            content="Original content",
            thought_type=ThoughtType.HYPOTHESIS,
            thought_number=1,
            total_thoughts=3
        )
        session.add_thought(original)

        revision = session.revise_thought(
            "t-001",
            "Revised content after more analysis",
            reason="Found additional evidence"
        )
        assert revision.is_revision is True
        assert revision.revises_thought_id == "t-001"
        assert len(session.thoughts) == 2

    def test_revise_nonexistent_thought(self):
        """Test revising nonexistent thought raises error."""
        session = BranchingThinkingSession(session_id="session-007")
        with pytest.raises(ValueError, match="not found"):
            session.revise_thought("nonexistent", "New content")

    def test_get_revision_chain(self):
        """Test getting revision chain."""
        session = BranchingThinkingSession(session_id="session-008")
        thought1 = EnhancedThought(
            id="t-001",
            content="First version",
            thought_type=ThoughtType.OBSERVATION,
            thought_number=1,
            total_thoughts=5
        )
        session.add_thought(thought1)

        session.revise_thought("t-001", "Second version")

        chain = session.get_revision_chain("t-001")
        assert len(chain) == 2
        assert chain[0].id == "t-001"

    def test_to_json(self):
        """Test JSON serialization."""
        session = BranchingThinkingSession(
            session_id="session-009",
            created_at="2026-01-17T10:00:00Z"
        )
        thought = EnhancedThought(
            id="t-001",
            content="Test",
            thought_type=ThoughtType.QUESTION,
            thought_number=1,
            total_thoughts=1
        )
        session.add_thought(thought)

        json_data = session.to_json()
        assert json_data["sessionId"] == "session-009"
        assert len(json_data["thoughts"]) == 1
        assert json_data["createdAt"] == "2026-01-17T10:00:00Z"


# =============================================================================
# Integration Tests
# =============================================================================

class TestV1011Integration:
    """Integration tests for V10.11 patterns."""

    def test_fastmcp_with_context(self):
        """Test FastMCP server with full context."""
        server = FastMCPServer("integration-test", version="2.0.0")
        lifespan = LifespanContext(server=server)
        lifespan.set("cache", {})

        context = MCPContext(
            request_id="int-001",
            server=server,
            lifespan=lifespan
        )

        @server.tool()
        def cached_lookup(key: str) -> str:
            cache = context.get_lifespan_state("cache")
            return cache.get(key, "not found")

        assert "cached_lookup" in server.list_tools()

    def test_grafana_dashboard_with_queries(self):
        """Test Grafana dashboard with Prometheus and Loki queries."""
        dashboard = GrafanaDashboard(uid="int-dash", title="Integration Dashboard")

        # CPU panel with Prometheus
        cpu_query = PrometheusQuery(expr='rate(cpu_usage[5m])', legend_format="{{host}}")
        cpu_panel = GrafanaPanel(
            id=1,
            title="CPU",
            panel_type=DashboardPanelType.TIMESERIES,
            targets=[cpu_query.to_target()]
        )
        dashboard.add_panel(cpu_panel)

        # Logs panel with Loki
        log_query = LokiQuery(expr='{app="test"} |= "error"', max_lines=100)
        log_panel = GrafanaPanel(
            id=2,
            title="Errors",
            panel_type=DashboardPanelType.LOGS,
            targets=[log_query.to_target()]
        )
        dashboard.add_panel(log_panel)

        json_data = dashboard.to_json()
        assert len(json_data["panels"]) == 2
        assert json_data["panels"][0]["targets"][0]["expr"] == 'rate(cpu_usage[5m])'

    def test_qdrant_semantic_search_workflow(self):
        """Test Qdrant semantic search workflow."""
        config = VectorConfig(size=768, distance=DistanceMetric.COSINE)
        collection = QdrantCollection(name="documents", vectors_config=config)
        model = EmbeddingModel(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            dimensions=768
        )
        search = SemanticSearch(
            collection=collection,
            embedding_model=model,
            top_k=5,
            score_threshold=0.8
        )

        # Build query
        query = search.build_query([0.1] * 768, {"category": {"$eq": "tech"}})
        assert query["limit"] == 5
        assert query["filter"] == {"category": {"$eq": "tech"}}

    def test_postgres_health_monitoring(self):
        """Test Postgres health monitoring workflow."""
        health = DatabaseHealth(
            status=HealthStatus.WARNING,
            connection_count=180,
            max_connections=200,
            cache_hit_ratio=0.92,
            active_queries=15,
            blocked_queries=2,
            warnings=["High connection usage"]
        )

        json_data = health.to_json()
        assert json_data["connectionUsagePercent"] == 90.0
        assert json_data["status"] == "warning"

    def test_branching_thought_process(self):
        """Test complete branching thought process."""
        session = BranchingThinkingSession(session_id="complex-thought")

        # Main line of thinking
        session.add_thought(EnhancedThought(
            id="t1",
            content="Problem statement: optimize database queries",
            thought_type=ThoughtType.OBSERVATION,
            thought_number=1,
            total_thoughts=10
        ))

        session.add_thought(EnhancedThought(
            id="t2",
            content="Hypothesis: indexing will help",
            thought_type=ThoughtType.HYPOTHESIS,
            thought_number=2,
            total_thoughts=10
        ))

        # Branch to explore alternative
        branch = session.create_branch("Query rewrite approach", "t2")
        session.switch_branch(branch.id)

        session.add_thought(EnhancedThought(
            id="t3",
            content="Alternative: rewrite queries instead",
            thought_type=ThoughtType.ANALYSIS,
            thought_number=3,
            total_thoughts=10
        ))

        # Revise based on findings
        session.revise_thought("t3", "Query rewrite shows 30% improvement")

        json_data = session.to_json()
        assert len(json_data["branches"]) == 1
        assert len(json_data["thoughts"]) == 4  # 2 main + 1 branch + 1 revision


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
