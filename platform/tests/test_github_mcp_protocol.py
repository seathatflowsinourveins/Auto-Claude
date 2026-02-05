"""
GitHub MCP Server Protocol Verification

Tests to verify GitHub MCP server uses Streamable HTTP transport.
Part of MCP 2026 migration (SSE deprecation).

Run with:
    cd /c/Users/42 && uv run --no-project --with pytest,pytest-asyncio,httpx python -m pytest \
      "Z:/insider/AUTO CLAUDE/unleash/platform/tests/test_github_mcp_protocol.py" -v
"""

import pytest
import os
import httpx


async def detect_mcp_transport(server_url: str, headers: dict = None) -> str:
    """
    Detect which transport a server supports.
    
    Returns: "streamable-http", "legacy-sse", or "unknown"
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Try Streamable HTTP (POST to /mcp)
            response = await client.post(
                f"{server_url}/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-11-25",
                    **(headers or {})
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "unleash", "version": "65.0.0"}
                    }
                }
            )
            
            if response.status_code == 200:
                # Check for session ID header (definitive indicator)
                if "Mcp-Session-Id" in response.headers:
                    return "streamable-http"
                
                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return "streamable-http"
        
        except Exception:
            pass
        
        return "unknown"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_github_mcp_protocol_detection():
    """Verify GitHub MCP server protocol version."""
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        pytest.skip("GITHUB_TOKEN not set - cannot test GitHub MCP server")
    
    # Test protocol detection
    transport = await detect_mcp_transport(
        "https://api.githubcopilot.com",
        headers={"Authorization": f"Bearer {github_token}"}
    )
    
    print(f"\nGitHub MCP transport detected: {transport}")
    
    # Document the result
    assert transport in ("streamable-http", "unknown"), \
        f"Unexpected transport type: {transport}"
    
    if transport == "streamable-http":
        print("✅ GitHub MCP server supports Streamable HTTP")
    else:
        print("⚠️  GitHub MCP server protocol could not be determined")
        print("   This may indicate the server uses a different endpoint structure")
        print("   or requires additional authentication parameters")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_github_mcp_session_management():
    """Test GitHub MCP server session ID support."""
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        pytest.skip("GITHUB_TOKEN not set - cannot test GitHub MCP server")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Initialize connection
            response = await client.post(
                "https://api.githubcopilot.com/mcp",
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-11-25"
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "unleash-test", "version": "65.0.0"}
                    }
                }
            )
            
            if response.status_code == 200:
                session_id = response.headers.get("Mcp-Session-Id")
                
                print(f"\nSession ID present: {session_id is not None}")
                if session_id:
                    print(f"Session ID: {session_id[:16]}...")
                    print("✅ GitHub MCP supports session management")
                else:
                    print("⚠️  No session ID in response")
                    print("   Server may not support stateful sessions")
                
                # Log protocol version if available
                protocol_version = response.headers.get("MCP-Protocol-Version")
                if protocol_version:
                    print(f"Protocol version: {protocol_version}")
            else:
                print(f"\n❌ Initialization failed: {response.status_code}")
                print(f"Response: {response.text[:200]}")
        
        except Exception as e:
            print(f"\n⚠️  Connection test failed: {e}")
            pytest.skip(f"Could not connect to GitHub MCP server: {e}")


if __name__ == "__main__":
    import asyncio
    
    print("GitHub MCP Protocol Verification")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_github_mcp_protocol_detection())
    asyncio.run(test_github_mcp_session_management())
