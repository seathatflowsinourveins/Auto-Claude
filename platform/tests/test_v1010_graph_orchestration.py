#!/usr/bin/env python3
"""
Tests for V10.10 Graph Orchestration and Programmatic LM Patterns.

Based on:
- LangGraph: https://github.com/langchain-ai/langgraph
- LangChain MCP Adapters: https://github.com/langchain-ai/langchain-mcp-adapters
- DSPy: https://github.com/stanfordnlp/dspy
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # LangGraph Graph Orchestration
    NodeType,
    EdgeType,
    GraphNode,
    GraphEdge,
    GraphCheckpoint,
    StateGraph,
    GraphExecutionStatus,
    GraphRuntime,
    CyclicalWorkflow,
    # DSPy Programmatic LM Patterns
    SignatureField,
    DSPySignature,
    DSPyExample,
    DSPyModule,
    DSPyPredict,
    OptimizationStrategy,
    OptimizationResult,
    DSPyOptimizer,
    DSPyTool,
    # MCP Multi-Server Patterns
    MCPTransportType,
    MCPServerConfig,
    MCPToolArtifact,
    MCPInterceptor,
    InterceptorChain,
    StatefulSession,
    SessionPool,
    MultiServerClient,
    ResourceLoader,
)


# =============================================================================
# LangGraph Graph Orchestration Tests
# =============================================================================

class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_type_values(self):
        """Test all node types are defined."""
        assert NodeType.START.value == "start"
        assert NodeType.END.value == "end"
        assert NodeType.TASK.value == "task"
        assert NodeType.CONDITIONAL.value == "conditional"
        assert NodeType.TOOL.value == "tool"
        assert NodeType.SUBGRAPH.value == "subgraph"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_edge_type_values(self):
        """Test all edge types are defined."""
        assert EdgeType.NORMAL.value == "normal"
        assert EdgeType.CONDITIONAL.value == "conditional"
        assert EdgeType.LOOP.value == "loop"


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_node_creation(self):
        """Test creating a graph node."""
        node = GraphNode(
            node_id="node_1",
            name="Process Data",
            node_type=NodeType.TASK,
            handler="process_data"
        )
        assert node.node_id == "node_1"
        assert node.name == "Process Data"
        assert node.node_type == NodeType.TASK
        assert node.handler == "process_data"

    def test_node_to_dict(self):
        """Test node serialization."""
        node = GraphNode(
            node_id="node_1",
            name="Start",
            node_type=NodeType.START
        )
        data = node.to_dict()
        assert data["node_id"] == "node_1"
        assert data["node_type"] == "start"

    def test_node_from_dict(self):
        """Test node deserialization."""
        data = {
            "node_id": "node_2",
            "name": "End",
            "node_type": "end"
        }
        node = GraphNode.from_dict(data)
        assert node.node_id == "node_2"
        assert node.node_type == NodeType.END


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_edge_creation(self):
        """Test creating a graph edge."""
        edge = GraphEdge(
            source_id="node_1",
            target_id="node_2",
            edge_type=EdgeType.NORMAL
        )
        assert edge.source_id == "node_1"
        assert edge.target_id == "node_2"

    def test_conditional_edge(self):
        """Test conditional edge."""
        edge = GraphEdge(
            source_id="decision",
            target_id="action_a",
            edge_type=EdgeType.CONDITIONAL,
            condition="should_proceed"
        )
        assert edge.edge_type == EdgeType.CONDITIONAL
        assert edge.condition == "should_proceed"

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = GraphEdge(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.LOOP,
            priority=5
        )
        data = edge.to_dict()
        assert data["edge_type"] == "loop"
        assert data["priority"] == 5


class TestGraphCheckpoint:
    """Tests for GraphCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        cp = GraphCheckpoint(
            checkpoint_id="cp_1",
            graph_id="graph_1",
            current_node="node_3",
            state={"key": "value"}
        )
        assert cp.checkpoint_id == "cp_1"
        assert cp.state["key"] == "value"

    def test_checkpoint_to_dict(self):
        """Test checkpoint serialization."""
        cp = GraphCheckpoint(
            checkpoint_id="cp_1",
            graph_id="graph_1",
            current_node="node_1",
            state={}
        )
        data = cp.to_dict()
        assert "created_at" in data
        assert data["graph_id"] == "graph_1"

    def test_checkpoint_from_dict(self):
        """Test checkpoint deserialization."""
        data = {
            "checkpoint_id": "cp_2",
            "graph_id": "graph_1",
            "current_node": "node_2",
            "state": {"x": 1},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        cp = GraphCheckpoint.from_dict(data)
        assert cp.checkpoint_id == "cp_2"
        assert cp.state["x"] == 1


class TestStateGraph:
    """Tests for StateGraph dataclass."""

    def test_graph_creation(self):
        """Test creating a state graph."""
        graph = StateGraph(
            graph_id="graph_1",
            name="My Workflow"
        )
        assert graph.graph_id == "graph_1"
        assert len(graph.nodes) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = StateGraph(graph_id="g1", name="Test")
        node = GraphNode(node_id="n1", name="Start", node_type=NodeType.START)
        graph.add_node(node)

        assert "n1" in graph.nodes
        assert graph.entry_point == "n1"

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="n1", name="A"))
        graph.add_node(GraphNode(node_id="n2", name="B"))
        graph.add_edge(GraphEdge(source_id="n1", target_id="n2"))

        assert len(graph.edges) == 1

    def test_add_conditional_edge(self):
        """Test adding conditional edges."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="decision", name="Decision"))
        graph.add_node(GraphNode(node_id="yes", name="Yes Path"))
        graph.add_node(GraphNode(node_id="no", name="No Path"))

        graph.add_conditional_edge(
            source_id="decision",
            targets={"should_proceed": "yes"},
            default="no"
        )

        assert len(graph.edges) == 2

    def test_get_successors(self):
        """Test getting successor nodes."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="a", name="A"))
        graph.add_node(GraphNode(node_id="b", name="B"))
        graph.add_node(GraphNode(node_id="c", name="C"))
        graph.add_edge(GraphEdge(source_id="a", target_id="b"))
        graph.add_edge(GraphEdge(source_id="a", target_id="c"))

        successors = graph.get_successors("a")
        assert len(successors) == 2

    def test_has_cycles(self):
        """Test cycle detection."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_edge(GraphEdge(
            source_id="a",
            target_id="a",
            edge_type=EdgeType.LOOP
        ))
        assert graph.has_cycles() is True

    def test_graph_to_dict(self):
        """Test graph serialization."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="n1", name="Node1"))
        data = graph.to_dict()

        assert data["graph_id"] == "g1"
        assert "n1" in data["nodes"]


class TestGraphExecutionStatus:
    """Tests for GraphExecutionStatus enum."""

    def test_status_values(self):
        """Test all statuses are defined."""
        assert GraphExecutionStatus.PENDING.value == "pending"
        assert GraphExecutionStatus.RUNNING.value == "running"
        assert GraphExecutionStatus.COMPLETED.value == "completed"
        assert GraphExecutionStatus.FAILED.value == "failed"


class TestGraphRuntime:
    """Tests for GraphRuntime dataclass."""

    def test_runtime_creation(self):
        """Test creating a runtime."""
        graph = StateGraph(graph_id="g1", name="Test")
        runtime = GraphRuntime(graph=graph)

        assert runtime.status == GraphExecutionStatus.PENDING
        assert runtime.current_node is None

    def test_runtime_start(self):
        """Test starting execution."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="start", name="Start", node_type=NodeType.START))
        runtime = GraphRuntime(graph=graph)

        runtime.start()
        assert runtime.status == GraphExecutionStatus.RUNNING
        assert runtime.current_node == "start"
        assert "start" in runtime.execution_history

    def test_runtime_step(self):
        """Test stepping through execution."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="start", name="Start", node_type=NodeType.START))
        graph.add_node(GraphNode(node_id="end", name="End", node_type=NodeType.END))
        graph.add_edge(GraphEdge(source_id="start", target_id="end"))

        runtime = GraphRuntime(graph=graph)
        runtime.start()
        next_node = runtime.step()

        assert next_node == "end"

    def test_runtime_checkpoint(self):
        """Test creating checkpoints."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="n1", name="Node1", node_type=NodeType.START))
        runtime = GraphRuntime(graph=graph, state={"x": 1})
        runtime.start()

        cp = runtime.checkpoint()
        assert cp.graph_id == "g1"
        assert cp.state["x"] == 1

    def test_runtime_restore(self):
        """Test restoring from checkpoint."""
        graph = StateGraph(graph_id="g1", name="Test")
        graph.add_node(GraphNode(node_id="n1", name="Node1"))
        runtime = GraphRuntime(graph=graph)

        cp = GraphCheckpoint(
            checkpoint_id="cp1",
            graph_id="g1",
            current_node="n1",
            state={"restored": True}
        )
        runtime.restore(cp)

        assert runtime.current_node == "n1"
        assert runtime.state["restored"] is True
        assert runtime.status == GraphExecutionStatus.RUNNING


class TestCyclicalWorkflow:
    """Tests for CyclicalWorkflow dataclass."""

    def test_workflow_creation(self):
        """Test creating a cyclical workflow."""
        graph = StateGraph(graph_id="g1", name="Test")
        workflow = CyclicalWorkflow(
            workflow_id="w1",
            graph=graph,
            loop_limit=5
        )
        assert workflow.loop_limit == 5
        assert workflow.current_iteration == 0

    def test_can_continue_loop(self):
        """Test loop continuation check."""
        graph = StateGraph(graph_id="g1", name="Test")
        workflow = CyclicalWorkflow(workflow_id="w1", graph=graph, loop_limit=3)

        assert workflow.can_continue_loop() is True
        workflow.current_iteration = 3
        assert workflow.can_continue_loop() is False

    def test_iterate(self):
        """Test loop iteration."""
        graph = StateGraph(graph_id="g1", name="Test")
        workflow = CyclicalWorkflow(workflow_id="w1", graph=graph, loop_limit=2)

        assert workflow.iterate() is True
        assert workflow.current_iteration == 1
        assert workflow.iterate() is True
        assert workflow.iterate() is False

    def test_reset_loop(self):
        """Test loop reset."""
        graph = StateGraph(graph_id="g1", name="Test")
        workflow = CyclicalWorkflow(workflow_id="w1", graph=graph)
        workflow.current_iteration = 5
        workflow.loop_state["key"] = "value"

        workflow.reset_loop()
        assert workflow.current_iteration == 0
        assert len(workflow.loop_state) == 0

    def test_exit_conditions(self):
        """Test exit condition checking."""
        graph = StateGraph(graph_id="g1", name="Test")
        workflow = CyclicalWorkflow(workflow_id="w1", graph=graph)

        workflow.set_loop_condition("done", True)
        assert workflow.check_exit_condition("done") is True
        assert workflow.check_exit_condition("missing") is False


# =============================================================================
# DSPy Programmatic LM Tests
# =============================================================================

class TestSignatureField:
    """Tests for SignatureField dataclass."""

    def test_field_creation(self):
        """Test creating a signature field."""
        field = SignatureField(
            name="question",
            description="The question to answer",
            is_input=True
        )
        assert field.name == "question"
        assert field.is_input is True


class TestDSPySignature:
    """Tests for DSPySignature dataclass."""

    def test_signature_creation(self):
        """Test creating a signature."""
        sig = DSPySignature(
            name="QA",
            inputs=[SignatureField(name="question")],
            outputs=[SignatureField(name="answer", is_input=False)]
        )
        assert sig.name == "QA"
        assert len(sig.inputs) == 1
        assert len(sig.outputs) == 1

    def test_signature_from_string(self):
        """Test parsing signature string."""
        sig = DSPySignature.from_string("question, context -> answer")

        assert len(sig.inputs) == 2
        assert sig.inputs[0].name == "question"
        assert sig.inputs[1].name == "context"
        assert len(sig.outputs) == 1
        assert sig.outputs[0].name == "answer"

    def test_signature_to_prompt(self):
        """Test prompt generation."""
        sig = DSPySignature(
            name="Test",
            inputs=[SignatureField(name="input")],
            outputs=[SignatureField(name="output", is_input=False)],
            instructions="Do something"
        )
        prompt = sig.to_prompt()

        assert "Instructions: Do something" in prompt
        assert "Input:" in prompt
        assert "Output:" in prompt

    def test_invalid_signature_format(self):
        """Test invalid signature string raises error."""
        with pytest.raises(ValueError):
            DSPySignature.from_string("no arrow here")


class TestDSPyExample:
    """Tests for DSPyExample dataclass."""

    def test_example_creation(self):
        """Test creating an example."""
        example = DSPyExample(
            inputs={"question": "What is 2+2?"},
            outputs={"answer": "4"}
        )
        assert example.inputs["question"] == "What is 2+2?"
        assert example.outputs["answer"] == "4"


class TestDSPyModule:
    """Tests for DSPyModule dataclass."""

    def test_module_creation(self):
        """Test creating a module."""
        sig = DSPySignature.from_string("question -> answer")
        module = DSPyModule(name="QA", signature=sig)

        assert module.name == "QA"
        assert module.compiled is False

    def test_module_forward(self):
        """Test module forward pass."""
        sig = DSPySignature.from_string("question -> answer")
        module = DSPyModule(name="QA", signature=sig)

        result = module.forward(question="What is AI?")
        assert "prompt" in result
        assert result["ready"] is True

    def test_module_forward_missing_input(self):
        """Test forward with missing required input."""
        sig = DSPySignature.from_string("question -> answer")
        module = DSPyModule(name="QA", signature=sig)

        with pytest.raises(ValueError, match="Missing required input"):
            module.forward()

    def test_add_demonstration(self):
        """Test adding demonstrations."""
        sig = DSPySignature.from_string("q -> a")
        module = DSPyModule(name="Test", signature=sig)

        example = DSPyExample(inputs={"q": "x"}, outputs={"a": "y"})
        module.add_demonstration(example)

        assert len(module.demonstrations) == 1

    def test_compile(self):
        """Test module compilation."""
        sig = DSPySignature.from_string("q -> a")
        module = DSPyModule(name="Test", signature=sig)

        module.compile()
        assert module.compiled is True


class TestDSPyPredict:
    """Tests for DSPyPredict dataclass."""

    def test_predict_creation(self):
        """Test creating a predictor."""
        sig = DSPySignature.from_string("question -> answer")
        predict = DSPyPredict(signature=sig)

        assert predict.max_tokens == 1000
        assert predict.temperature == 0.0

    def test_predict_call(self):
        """Test calling predictor."""
        sig = DSPySignature.from_string("question -> answer")
        predict = DSPyPredict(signature=sig, temperature=0.7)

        result = predict(question="Test?")
        assert result["temperature"] == 0.7


class TestOptimizationStrategy:
    """Tests for OptimizationStrategy enum."""

    def test_strategy_values(self):
        """Test all strategies are defined."""
        assert OptimizationStrategy.BOOTSTRAP.value == "bootstrap"
        assert OptimizationStrategy.MIPRO.value == "mipro"
        assert OptimizationStrategy.RANDOM.value == "random"
        assert OptimizationStrategy.BAYESIAN.value == "bayesian"


class TestDSPyOptimizer:
    """Tests for DSPyOptimizer dataclass."""

    def test_optimizer_creation(self):
        """Test creating an optimizer."""
        optimizer = DSPyOptimizer(
            strategy=OptimizationStrategy.BOOTSTRAP,
            num_candidates=5
        )
        assert optimizer.num_candidates == 5

    def test_optimize(self):
        """Test optimization."""
        sig = DSPySignature.from_string("q -> a")
        module = DSPyModule(name="Test", signature=sig)
        optimizer = DSPyOptimizer()

        examples = [
            DSPyExample(inputs={"q": "1"}, outputs={"a": "1"}),
            DSPyExample(inputs={"q": "2"}, outputs={"a": "2"}),
        ]
        result = optimizer.optimize(module, examples)

        assert result.strategy == OptimizationStrategy.BOOTSTRAP
        assert len(result.optimized_demos) <= optimizer.num_candidates


class TestDSPyTool:
    """Tests for DSPyTool dataclass."""

    def test_tool_creation(self):
        """Test creating a tool."""
        tool = DSPyTool(
            name="calculator",
            description="Performs calculations",
            parameters={"type": "object"}
        )
        assert tool.name == "calculator"
        assert tool.async_enabled is True

    def test_from_mcp_tool(self):
        """Test converting from MCP tool."""
        mcp_tool = {
            "name": "search",
            "description": "Search the web",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        }
        tool = DSPyTool.from_mcp_tool(mcp_tool)

        assert tool.name == "search"
        assert tool.source == "mcp"

    def test_tool_to_dict(self):
        """Test tool serialization."""
        tool = DSPyTool(
            name="test",
            description="Test tool",
            parameters={}
        )
        data = tool.to_dict()

        assert data["name"] == "test"
        assert data["async_enabled"] is True


# =============================================================================
# MCP Multi-Server Tests
# =============================================================================

class TestMCPTransportType:
    """Tests for MCPTransportType enum."""

    def test_transport_values(self):
        """Test all transports are defined."""
        assert MCPTransportType.STDIO.value == "stdio"
        assert MCPTransportType.HTTP.value == "http"
        assert MCPTransportType.STREAMABLE_HTTP.value == "streamable_http"
        assert MCPTransportType.SSE.value == "sse"
        assert MCPTransportType.WEBSOCKET.value == "websocket"


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_stdio_config(self):
        """Test stdio server configuration."""
        config = MCPServerConfig(
            name="local",
            transport=MCPTransportType.STDIO,
            command="python",
            args=["server.py"]
        )
        assert config.validate() is True

    def test_http_config(self):
        """Test HTTP server configuration."""
        config = MCPServerConfig(
            name="remote",
            transport=MCPTransportType.HTTP,
            url="http://localhost:8000/mcp"
        )
        assert config.validate() is True

    def test_invalid_stdio_config(self):
        """Test invalid stdio config (no command)."""
        config = MCPServerConfig(
            name="invalid",
            transport=MCPTransportType.STDIO
        )
        assert config.validate() is False

    def test_invalid_http_config(self):
        """Test invalid http config (no url)."""
        config = MCPServerConfig(
            name="invalid",
            transport=MCPTransportType.HTTP
        )
        assert config.validate() is False


class TestMCPToolArtifact:
    """Tests for MCPToolArtifact dataclass."""

    def test_artifact_creation(self):
        """Test creating an artifact."""
        artifact = MCPToolArtifact(
            content="Result text",
            artifact={"data": [1, 2, 3]},
            content_type="application/json"
        )
        assert artifact.content == "Result text"
        assert artifact.artifact["data"] == [1, 2, 3]


class TestMCPInterceptor:
    """Tests for MCPInterceptor dataclass."""

    def test_interceptor_creation(self):
        """Test creating an interceptor."""
        interceptor = MCPInterceptor(
            name="auth",
            priority=10
        )
        assert interceptor.priority == 10
        assert interceptor.enabled is True

    def test_before_call(self):
        """Test before_call hook."""
        interceptor = MCPInterceptor(name="test")
        args = {"key": "value"}
        result = interceptor.before_call("tool", args, {})
        assert result == args

    def test_after_call(self):
        """Test after_call hook."""
        interceptor = MCPInterceptor(name="test")
        artifact = MCPToolArtifact(content="test")
        result = interceptor.after_call("tool", artifact, {})
        assert result == artifact

    def test_on_error(self):
        """Test on_error hook."""
        interceptor = MCPInterceptor(name="test")
        result = interceptor.on_error("tool", Exception("test"), {})
        assert result is None


class TestInterceptorChain:
    """Tests for InterceptorChain dataclass."""

    def test_chain_creation(self):
        """Test creating an interceptor chain."""
        chain = InterceptorChain()
        assert len(chain.interceptors) == 0

    def test_add_interceptor(self):
        """Test adding interceptors."""
        chain = InterceptorChain()
        chain.add(MCPInterceptor(name="low", priority=1))
        chain.add(MCPInterceptor(name="high", priority=10))

        # Should be sorted by priority descending
        assert chain.interceptors[0].name == "high"
        assert chain.interceptors[1].name == "low"

    def test_before_chain(self):
        """Test executing before chain."""
        chain = InterceptorChain()
        chain.add(MCPInterceptor(name="test"))

        args = {"x": 1}
        result = chain.before("tool", args, {})
        assert result == args

    def test_after_chain(self):
        """Test executing after chain."""
        chain = InterceptorChain()
        chain.add(MCPInterceptor(name="test"))

        artifact = MCPToolArtifact(content="test")
        result = chain.after("tool", artifact, {})
        assert result == artifact


class TestStatefulSession:
    """Tests for StatefulSession dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        session = StatefulSession(
            session_id="sess_1",
            server_name="server1"
        )
        assert session.connected is False

    def test_touch(self):
        """Test updating last activity."""
        session = StatefulSession(
            session_id="sess_1",
            server_name="server1"
        )
        old_time = session.last_activity
        session.touch()
        assert session.last_activity >= old_time

    def test_is_stale(self):
        """Test stale session detection."""
        session = StatefulSession(
            session_id="sess_1",
            server_name="server1"
        )
        assert session.is_stale(max_idle_seconds=0.0) is True
        assert session.is_stale(max_idle_seconds=1000.0) is False


class TestSessionPool:
    """Tests for SessionPool dataclass."""

    def test_pool_creation(self):
        """Test creating a session pool."""
        pool = SessionPool(max_sessions=5)
        assert pool.max_sessions == 5
        assert len(pool.sessions) == 0

    def test_create_session(self):
        """Test creating a session."""
        pool = SessionPool()
        session = pool.create("server1")

        assert session.server_name == "server1"
        assert session.connected is True
        assert "server1" in pool.sessions

    def test_get_session(self):
        """Test getting an existing session."""
        pool = SessionPool()
        pool.create("server1")

        session = pool.get("server1")
        assert session is not None
        assert session.server_name == "server1"

    def test_get_missing_session(self):
        """Test getting a non-existent session."""
        pool = SessionPool()
        session = pool.get("nonexistent")
        assert session is None

    def test_release_session(self):
        """Test releasing a session."""
        pool = SessionPool()
        session = pool.create("server1")
        pool.release(session)

        assert session.connected is False

    def test_max_sessions_limit(self):
        """Test max sessions limit."""
        pool = SessionPool(max_sessions=2)
        pool.create("server1")
        pool.create("server2")
        pool.create("server3")

        assert len(pool.sessions) == 2


class TestMultiServerClient:
    """Tests for MultiServerClient dataclass."""

    def test_client_creation(self):
        """Test creating a client."""
        client = MultiServerClient()
        assert len(client.servers) == 0

    def test_add_server(self):
        """Test adding a server."""
        client = MultiServerClient()
        config = MCPServerConfig(
            name="test",
            transport=MCPTransportType.HTTP,
            url="http://localhost:8000"
        )
        client.add_server(config)

        assert "test" in client.servers

    def test_add_invalid_server(self):
        """Test adding invalid server config."""
        client = MultiServerClient()
        config = MCPServerConfig(
            name="invalid",
            transport=MCPTransportType.HTTP
            # Missing URL
        )
        client.add_server(config)

        assert "invalid" not in client.servers

    def test_register_tools(self):
        """Test registering tools."""
        client = MultiServerClient()
        tools = [
            {"name": "tool1", "description": "Tool 1"},
            {"name": "tool2", "description": "Tool 2"}
        ]
        client.register_tools("server1", tools)

        registered = client.get_tools("server1")
        assert len(registered) == 2

    def test_get_all_tools(self):
        """Test getting all tools from all servers."""
        client = MultiServerClient()
        client.register_tools("server1", [{"name": "t1"}])
        client.register_tools("server2", [{"name": "t2"}, {"name": "t3"}])

        all_tools = client.get_tools()
        assert len(all_tools) == 3

    def test_get_session(self):
        """Test getting a session."""
        client = MultiServerClient()
        session = client.get_session("server1")

        assert session.server_name == "server1"
        assert session.connected is True


class TestResourceLoader:
    """Tests for ResourceLoader dataclass."""

    def test_loader_creation(self):
        """Test creating a resource loader."""
        client = MultiServerClient()
        loader = ResourceLoader(client=client)
        assert loader.client == client

    def test_get_resources(self):
        """Test loading resources."""
        client = MultiServerClient()
        loader = ResourceLoader(client=client)
        resources = loader.get_resources("server1")
        assert resources == []

    def test_get_prompt(self):
        """Test loading a prompt."""
        client = MultiServerClient()
        loader = ResourceLoader(client=client)
        prompt = loader.get_prompt("server1", "summarize")
        assert prompt == []

    def test_list_prompts(self):
        """Test listing prompts."""
        client = MultiServerClient()
        loader = ResourceLoader(client=client)
        prompts = loader.list_prompts("server1")
        assert prompts == []


# =============================================================================
# Integration Tests
# =============================================================================

class TestV1010Integration:
    """Integration tests for V10.10 patterns."""

    def test_graph_workflow_execution(self):
        """Test complete graph workflow execution."""
        # Create graph
        graph = StateGraph(graph_id="workflow", name="Test Workflow")

        # Add nodes
        graph.add_node(GraphNode(node_id="start", name="Start", node_type=NodeType.START))
        graph.add_node(GraphNode(node_id="process", name="Process"))
        graph.add_node(GraphNode(node_id="end", name="End", node_type=NodeType.END))

        # Add edges
        graph.add_edge(GraphEdge(source_id="start", target_id="process"))
        graph.add_edge(GraphEdge(source_id="process", target_id="end"))

        # Execute
        runtime = GraphRuntime(graph=graph)
        runtime.start()

        assert runtime.current_node == "start"
        runtime.step()  # -> process
        assert runtime.current_node == "process"
        runtime.step()  # -> end
        assert runtime.current_node == "end"
        runtime.step()  # completes
        assert runtime.status == GraphExecutionStatus.COMPLETED

    def test_dspy_module_pipeline(self):
        """Test DSPy module pipeline."""
        # Create signature
        sig = DSPySignature.from_string("question, context -> answer, reasoning")
        sig.instructions = "Answer the question based on context"

        # Create module
        module = DSPyModule(name="QA", signature=sig)

        # Add demonstrations
        module.add_demonstration(DSPyExample(
            inputs={"question": "What is 2+2?", "context": "Math"},
            outputs={"answer": "4", "reasoning": "Basic addition"}
        ))

        # Run forward
        result = module.forward(question="What is 3+3?", context="Math")

        assert result["ready"] is True
        assert result["demonstrations"] == 1

    def test_multi_server_with_interceptors(self):
        """Test multi-server client with interceptors."""
        # Create client
        client = MultiServerClient()

        # Add interceptor
        auth_interceptor = MCPInterceptor(name="auth", priority=10)
        client.interceptor_chain.add(auth_interceptor)

        # Add server
        client.add_server(MCPServerConfig(
            name="api",
            transport=MCPTransportType.HTTP,
            url="http://localhost:8000"
        ))

        # Register tools
        client.register_tools("api", [
            {"name": "search", "description": "Search"}
        ])

        # Get tools
        tools = client.get_tools()
        assert len(tools) == 1

        # Test interceptor chain
        args = client.interceptor_chain.before("search", {"q": "test"}, {})
        assert args["q"] == "test"

    def test_cyclical_workflow_with_checkpoints(self):
        """Test cyclical workflow with checkpointing."""
        # Create graph with loop
        graph = StateGraph(graph_id="loop", name="Loop Workflow")
        graph.add_node(GraphNode(node_id="start", name="Start", node_type=NodeType.START))
        graph.add_node(GraphNode(node_id="process", name="Process"))
        graph.add_edge(GraphEdge(
            source_id="process",
            target_id="start",
            edge_type=EdgeType.LOOP
        ))

        # Create workflow
        workflow = CyclicalWorkflow(
            workflow_id="w1",
            graph=graph,
            loop_limit=3
        )

        # Create runtime
        runtime = GraphRuntime(graph=graph, state={"iteration": 0})
        runtime.start()

        # Checkpoint before loop
        cp = runtime.checkpoint()

        # Iterate
        iterations = 0
        while workflow.iterate():
            iterations += 1
            runtime.state["iteration"] = iterations

        assert iterations == 3
        assert not workflow.can_continue_loop()

        # Restore from checkpoint
        runtime.restore(cp)
        assert runtime.state["iteration"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
