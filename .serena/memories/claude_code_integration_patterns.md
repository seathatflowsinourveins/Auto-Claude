# Claude Code Integration Patterns
## Seamless SDK Access via Unleash Ecosystem

---

## 🔗 ACTIVE INTEGRATIONS

### 1. Serena (Code Intelligence)
**Status:** ✅ ACTIVE via MCP
**Config Location:** `C:\Users\42\.claude\mcp_servers_OPTIMAL.json`
**Executable:** `Z:/insider/AUTO CLAUDE/unleash/sdks/serena/.venv/Scripts/serena-mcp-server.exe`

**Tools Available:**
- `find_symbol` - Semantic code search
- `get_symbols_overview` - File symbol mapping
- `find_referencing_symbols` - Cross-file dependencies
- `replace_symbol_body` - Type-safe code editing
- `search_for_pattern` - Regex search
- `think_about_*` - Metacognition tools

### 2. Context7 (Library Documentation)
**Status:** ✅ ACTIVE via MCP
**Tools:**
- `resolve-library-id` - Find library IDs
- `query-docs` - Get up-to-date documentation

### 3. Exa (Deep Research)
**Status:** ✅ ACTIVE via MCP
**Tools:**
- `web_search_exa` - Real-time web search
- `deep_researcher_start` - Async research tasks
- `deep_researcher_check` - Check research status
- `get_code_context_exa` - Code context retrieval

### 4. TouchDesigner Creative
**Status:** ✅ ACTIVE via MCP
**Tools:**
- `create_node`, `delete_node`, `get_node_info`
- `set_parameter`, `set_parameters`
- `connect_nodes`, `disconnect_nodes`
- `execute_script` - Python execution in TD

---

## 📦 SDK INTEGRATION PATTERNS

### Pattern 1: MCP Server Consumption
```
SDK → MCP Server → Claude Code MCP Client
```
Used for: Serena, TouchDesigner, Context7, Exa

### Pattern 2: Direct Python Import
```
SDK → Python script → Bash tool execution
```
Used for: DSPy, Instructor, Temporal

### Pattern 3: Skill-Based Invocation
```
SDK patterns → CLAUDE.md skills → /skill-name
```
Used for: FastMCP, LangGraph patterns

---

## 🎯 WORKFLOW MAPPINGS

### AlphaForge Trading Workflow
```
1. Code Understanding → Serena (mcp__serena__*)
2. Strategy Research → Exa (mcp__exa__deep_researcher_*)
3. Library Docs → Context7 (mcp__plugin_context7__*)
4. Implementation → Standard Edit/Write tools
5. Testing → Bash (pytest)
```

### State of Witness Creative Workflow
```
1. Code Navigation → Serena (find_symbol, get_symbols_overview)
2. TouchDesigner Control → TD MCP (create_node, execute_script)
3. Research → Exa + Context7
4. Shader Generation → Standard Write + GLSL patterns
5. QD Exploration → Pyribs SDK (direct Python)
```

---

## 🔧 SESSION INITIALIZATION

### Recommended Startup Sequence
1. Read this memory file for integration status
2. Read `sdk_selection_guide.md` for SDK reference
3. Check Serena onboarding: `mcp__serena__check_onboarding_performed`
4. List available memories: `mcp__serena__list_memories`

### Available Serena Projects
1. **Touchdesigner-createANDBE** (State of Witness)
2. **unleash** (SDK Ecosystem)

Use `mcp__serena__activate_project` to switch context.

---

## 📊 TOKEN EFFICIENCY TIPS

### Serena Symbol-Based Reading
Instead of: `Read entire_file.py` (~10,000 tokens)
Use: `mcp__serena__find_symbol name_path="ClassName/method"` (~500 tokens)
**Savings: 95%**

### Context7 Targeted Docs
Instead of: Full library documentation
Use: `mcp__plugin_context7_context7__query-docs query="specific API question"`
**Savings: 80%**

### Exa Research Batching
Instead of: Multiple web searches
Use: `mcp__exa__deep_researcher_start` for comprehensive async research
**Savings: 60%**

---

*Last Updated: 2026-01-20*
