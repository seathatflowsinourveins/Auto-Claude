# TASK-002: Chonkie Response Chunking

**Task ID**: TASK-002
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P0
**Complexity**: Low
**Estimated Hours**: 2
**Depends On**: None
**Blocks**: TASK-003

---

## Objective

Implement large MCP response handling using Chonkie for intelligent chunking, enabling map-reduce synthesis for responses exceeding token limits.

---

## Background

MCP tool responses can exceed 100k tokens. Current implementation truncates large responses, losing valuable information. Chonkie provides 33x faster chunking with semantic awareness.

### Existing Integration
- `chonkie_adapter_bridge.py` exists with RealChonkieAdapter
- Already integrated with ultimate_orchestrator
- Supports: chunk_code, chunk_semantic, chunk_token, chunk_sentence

---

## Implementation Plan

### 1. Create ResponseChunker Class

```python
# ~/.claude/integrations/response_chunker.py

from chonkie_adapter_bridge import RealChonkieAdapter

class ResponseChunker:
    def __init__(self, max_chunk_tokens: int = 4096):
        self.adapter = RealChonkieAdapter()
        self.max_chunk_tokens = max_chunk_tokens
        
    async def chunk_response(
        self, 
        content: str,
        strategy: str = "semantic"
    ) -> list[ChunkResult]:
        """
        Chunk large response for map-reduce processing.
        
        Strategies:
        - semantic: Preserve meaning boundaries
        - token: Fixed size chunks
        - sentence: Sentence boundaries
        """
        if strategy == "semantic":
            result = await self.adapter.execute(
                operation="chunk_semantic",
                text=content,
                threshold=0.5
            )
        elif strategy == "token":
            result = await self.adapter.execute(
                operation="chunk_token",
                text=content,
                chunk_size=self.max_chunk_tokens,
                overlap=100
            )
        else:
            result = await self.adapter.execute(
                operation="chunk_sentence",
                text=content
            )
            
        return [
            ChunkResult(content=c["content"], tokens=c["tokens"])
            for c in result["data"]["chunks"]
        ]
```

### 2. Map-Reduce Synthesis

```python
async def map_reduce_synthesis(
    self,
    chunks: list[ChunkResult],
    query: str,
    synthesizer: Callable
) -> str:
    """
    Map: Extract key info from each chunk
    Reduce: Combine into coherent answer
    """
    # Map phase - parallel extraction
    extractions = await asyncio.gather(*[
        synthesizer(query, chunk.content)
        for chunk in chunks
    ])
    
    # Reduce phase - combine
    combined = "\n---\n".join(extractions)
    return await synthesizer(
        f"Synthesize these extractions for: {query}",
        combined
    )
```

### 3. Integration with Research Orchestrator

- Add `ResponseChunker` to `ComprehensiveResearchOrchestrator`
- Call chunking when response > threshold
- Use map-reduce for synthesis of chunked responses

---

## Acceptance Criteria

### Functional
- [ ] ResponseChunker class with 3 strategies
- [ ] Map-reduce synthesis pipeline
- [ ] Automatic chunking for large responses (>8k tokens)
- [ ] Chunk metadata preservation

### Non-Functional
- [ ] Chunking adds < 100ms latency (Chonkie is 33x faster)
- [ ] Handle responses up to 100k tokens
- [ ] Memory-efficient streaming support

### Testing
- [ ] Unit tests for each chunking strategy
- [ ] Test with 50k token response
- [ ] Test map-reduce synthesis quality
- [ ] Integration test with MCP response

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `response_chunker.py` | Create | Chunking module |
| `research_orchestrator.py` | Modify | Add chunking integration |
| `tests/test_response_chunker.py` | Create | Unit tests |

---

## Technical Notes

### Chonkie SDK (verified from chonkie==1.5.4)
```python
# Uses chonkie_adapter_bridge.RealChonkieAdapter
# Operations: chunk_code, chunk_semantic, chunk_token, chunk_sentence
# Returns: {"data": {"chunks": [{"content": str, "tokens": int}]}}
```

### Token Counting
- Use tiktoken for accurate token counts
- Default chunk size: 4096 tokens
- Overlap: 100 tokens for context continuity

---

**Assignee**: TBD
**Started**: -
**Completed**: -
