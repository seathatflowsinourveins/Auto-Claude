# SDK Integration Reference Summary

## Quick Reference (2026-01-22)

### Key Findings from Deep Documentation Research

#### 1. Claude Code Architecture
- Terminal-native agentic coding tool
- Unix philosophy: composable, scriptable, pipeable
- Extension points: MCP, Skills, Plugins

#### 2. Tool Use Critical Patterns
- **Parallel tools**: Multiple tool_use blocks in single response
- **Sequential tools**: Chain dependent operations
- **Strict mode**: Add `strict: true` for guaranteed schema validation

#### 3. MCP Protocol (2025-06-18)
- JSON-RPC 2.0 based
- Two layers: Data (protocol) + Transport (stdio/HTTP)
- Capability negotiation on connect
- Convert `inputSchema` → `input_schema` for Claude

#### 4. Extended Thinking Requirements
- Min budget: 1,024 tokens
- **MUST preserve thinking blocks** during tool use
- Cannot toggle mid-turn
- Interleaved thinking: beta header `interleaved-thinking-2025-05-14`
- Above 32k: use batch processing

#### 5. Hooks System
- Lifecycle: SessionStart → UserPromptSubmit → PreToolUse → PostToolUse → Stop → SessionEnd
- Exit codes: 0 (success), 2 (block), other (non-blocking error)
- JSON control via stdout
- Environment: CLAUDE_PROJECT_DIR, CLAUDE_ENV_FILE

### Full Documentation
See: `Z:\insider\AUTO CLAUDE\unleash\docs\sdk-integration-reference.md`

### Official URLs
- Claude Code: https://code.claude.com/docs/en/overview
- Tool Use: https://platform.claude.com/docs/en/docs/build-with-claude/tool-use
- Extended Thinking: https://platform.claude.com/docs/en/docs/build-with-claude/extended-thinking
- MCP Spec: https://modelcontextprotocol.io/specification/latest
