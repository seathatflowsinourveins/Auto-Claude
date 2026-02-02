# V34 Architecture Integration - Cross-Session Reference

**Created**: 2026-01-23
**Purpose**: Quick reference for V34 patterns discovered through deep SDK research

## Key SDK Patterns

### Crawl4AI
- `AsyncWebCrawler` - Context manager pattern with Playwright
- `BFSDeepCrawlStrategy` - Resumable state with `on_state_change` callback
- `CosineStrategy` - Semantic extraction with MiniLM embeddings
- MCP Bridge: `@mcp_tool`, `@mcp_resource`, `@mcp_template` decorators

### LightRAG
- Knowledge graph with `entities_vdb`, `relationships_vdb`, `chunks_vdb`
- Document pipeline: enqueue → process → extract entities → merge
- Query modes: `naive` (vector only), `local` (KG), `hybrid` (both)
- `get_knowledge_graph()` for subgraph exploration

### CrewAI Flows
- `@start()` - Entry point methods
- `@listen("method_name")` - Event triggers
- `@router("method_name")` - Conditional routing
- `and_()`, `or_()` - Condition combinators
- `@persist(SQLiteFlowPersistence())` - State persistence
- `Flow.from_pending()` - Resume paused flows

### Mem0 Multi-Backend
- `Memory.from_config(config_dict=config)` - Initialize with config
- Supported: Qdrant, pgvector, Chroma, Weaviate, Redis, Milvus, FAISS
- Auto-detection based on environment variables
- `custom_fact_extraction_prompt` for custom memory handling

### Opik EvolutionaryOptimizer
- `optimize_prompt()` with quality metrics
- `CrewAIAgent` wrapper for agent optimization
- `LevenshteinRatio`, `AnswerRelevance` metrics

### Claude SDK 2026 Optimizations
- Prompt caching with `cache_control={"type": "ephemeral"}`
- Tool whitelisting: `--allowed-tools "Read,Write,Bash"`
- Batch processing via Batches API
- Memory segmentation by category

## File Locations
- V34 Document: `Z:/insider/AUTO CLAUDE/unleash/UNIFIED_ARCHITECTURE_V34_INTEGRATION.md`
- V33 Base: `Z:/insider/AUTO CLAUDE/unleash/UNIFIED_ARCHITECTURE_V33_PLUS.md`
- Crawl4AI SDK: `Z:/insider/AUTO CLAUDE/unleash/sdks/crawl4ai/`
- LightRAG SDK: `Z:/insider/AUTO CLAUDE/unleash/sdks/lightrag/`
- CrewAI SDK: `Z:/insider/AUTO CLAUDE/unleash/sdks/crewai/`
- Mem0 SDK: `Z:/insider/AUTO CLAUDE/unleash/sdks/mem0/`

## Quick Commands

```python
# Crawl4AI
async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(url, config=CrawlerRunConfig(...))

# LightRAG
rag = LightRAG(working_dir="./storage")
await rag.ainsert(documents)
result = await rag.aquery(query, param=QueryParam(mode="hybrid"))

# CrewAI Flow
class MyFlow(Flow):
    @start()
    def begin(self): ...
    @listen("begin")
    def next_step(self): ...

# Mem0
memory = Memory.from_config(config_dict=config)
memory.add("fact", user_id="user1")
```
