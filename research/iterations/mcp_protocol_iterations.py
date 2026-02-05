"""
MCP PROTOCOL ITERATIONS - Model Context Protocol Deep Dive
===========================================================
MCP servers, tools, resources, prompts, transport

Migrated to base_executor.py (Gap05) - uses shared infrastructure for:
- Gap02: Quality filtering (30-char min, garbage patterns, word-boundary truncation)
- Gap04: Honest stats recomputation from actual saved data
- Gap06: Synthesis with 40+ claim indicators
- Gap07: 3-layer dedup (URL, content-hash, vector)
- Gap09: Fallback broadening on sparse results
- Gap11: Quality scoring and dashboard
"""

from base_executor import BaseResearchExecutor, run_research


MCP_TOPICS = [
    # MCP Fundamentals
    {"topic": "MCP protocol: JSON-RPC, capabilities, initialization", "area": "protocol"},
    {"topic": "MCP transport: stdio, SSE, HTTP, WebSocket", "area": "protocol"},
    {"topic": "MCP message types: request, response, notification", "area": "protocol"},
    {"topic": "MCP lifecycle: connect, initialize, list, execute", "area": "protocol"},

    # MCP Tools
    {"topic": "MCP tool definition: name, description, input schema", "area": "tools"},
    {"topic": "MCP tool execution: call, result, error handling", "area": "tools"},
    {"topic": "Dynamic tool registration: runtime tool addition", "area": "tools"},
    {"topic": "Tool composition: chaining, parallel execution", "area": "tools"},

    # MCP Resources
    {"topic": "MCP resources: URI schemes, MIME types, content", "area": "resources"},
    {"topic": "Resource templates: dynamic URIs, parameters", "area": "resources"},
    {"topic": "Resource subscriptions: change notifications", "area": "resources"},
    {"topic": "Binary resources: images, files, streaming", "area": "resources"},

    # MCP Servers
    {"topic": "Building MCP servers: TypeScript SDK, Python SDK", "area": "servers"},
    {"topic": "MCP server patterns: filesystem, database, API wrapper", "area": "servers"},
    {"topic": "MCP server discovery: registry, local config", "area": "servers"},
    {"topic": "MCP server security: authentication, authorization", "area": "servers"},

    # MCP Integration
    {"topic": "Claude Desktop MCP: configuration, debugging", "area": "integration"},
    {"topic": "MCP with LangChain: tool integration, agents", "area": "integration"},
    {"topic": "MCP marketplace: community servers, distribution", "area": "integration"},
    {"topic": "Custom MCP clients: building host applications", "area": "integration"},
]


class MCPProtocolExecutor(BaseResearchExecutor):
    """Custom executor with MCP-specific Perplexity prompting."""

    def perplexity_prompt(self, topic: str, area: str) -> str:
        return f"Model Context Protocol (MCP) Anthropic implementation: {topic}"


if __name__ == "__main__":
    run_research(
        "mcp",
        "MCP PROTOCOL ITERATIONS",
        MCP_TOPICS,
        executor_class=MCPProtocolExecutor,
    )
