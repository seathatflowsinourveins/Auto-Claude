# TASK-005: HyDE Query Expansion

**Task ID**: TASK-005
**Epic**: EPIC-001 (Advanced Research Architecture v3.0)
**Status**: Ready
**Priority**: P1
**Complexity**: Medium
**Estimated Hours**: 3
**Depends On**: None
**Blocks**: TASK-006

---

## Objective

Implement HyDE (Hypothetical Document Embeddings) for query expansion, bridging the semantic gap between short queries and longer documents.

---

## Background

HyDE transforms a query into a hypothetical document containing the answer, then searches for documents similar to that hypothetical document. This addresses the query-document distribution mismatch.

### Key Benefits
- 42% improvement in retrieval precision (reported)
- 14% higher accuracy than query-only
- Zero-shot capability (no labeled data needed)
- Works well for complex/nuanced queries

### Implementation Reference
- [Haystack HyDE Docs](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [Zilliz HyDE Guide](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)

---

## Implementation Plan

### 1. Create HyDE Expander Class

```python
# ~/.claude/integrations/hyde_expander.py

from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class HyDEResult:
    original_query: str
    hypothetical_documents: list[str]
    combined_embedding: list[float] | None
    expansion_strategy: str

class HyDEExpander:
    def __init__(
        self,
        llm_generator: Callable[[str], Awaitable[str]],
        num_hypotheticals: int = 5,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.llm_generator = llm_generator
        self.num_hypotheticals = num_hypotheticals
        self.embedding_model = embedding_model
        
    async def expand_query(
        self,
        query: str,
        domain: str = "general"
    ) -> HyDEResult:
        """
        Generate hypothetical documents for query expansion.
        
        1. Generate N hypothetical answers to the query
        2. Embed each hypothetical document
        3. Average embeddings to get combined embedding
        4. Use combined embedding for similarity search
        """
        # Generate hypothetical documents
        hypotheticals = await self._generate_hypotheticals(query, domain)
        
        # Optionally compute combined embedding
        combined_embedding = None
        if self._embedding_available():
            embeddings = await self._embed_documents(hypotheticals)
            combined_embedding = self._average_embeddings(embeddings)
            
        return HyDEResult(
            original_query=query,
            hypothetical_documents=hypotheticals,
            combined_embedding=combined_embedding,
            expansion_strategy="hyde_multi"
        )
```

### 2. Hypothetical Document Generation

```python
HYDE_PROMPT_TEMPLATE = """
Given the question: "{query}"

Write a detailed, factual passage that would answer this question.
The passage should be written as if it comes from a technical documentation
or authoritative source. Include specific details, code examples if relevant,
and be comprehensive.

Domain: {domain}

Hypothetical Answer:
"""

async def _generate_hypotheticals(
    self,
    query: str,
    domain: str
) -> list[str]:
    """Generate N hypothetical documents."""
    hypotheticals = []
    
    for i in range(self.num_hypotheticals):
        prompt = HYDE_PROMPT_TEMPLATE.format(
            query=query,
            domain=domain
        )
        
        # Add variation prompts for diversity
        if i > 0:
            prompt += f"\n\nProvide a different perspective (variation {i}):"
            
        hypothetical = await self.llm_generator(prompt)
        hypotheticals.append(hypothetical)
        
    return hypotheticals
```

### 3. Embedding and Search

```python
async def search_with_hyde(
    self,
    query: str,
    vector_search: Callable,
    domain: str = "general"
) -> list[dict]:
    """
    Perform search using HyDE expansion.
    
    Instead of searching with query embedding,
    search with averaged hypothetical document embeddings.
    """
    hyde_result = await self.expand_query(query, domain)
    
    if hyde_result.combined_embedding:
        # Use combined embedding for vector search
        results = await vector_search(
            embedding=hyde_result.combined_embedding,
            top_k=10
        )
    else:
        # Fallback: use hypothetical text for keyword search
        combined_text = " ".join(hyde_result.hypothetical_documents)
        results = await vector_search(
            query=combined_text,
            top_k=10
        )
        
    return results
```

### 4. Integration with Research Orchestrator

```python
# In research() method

# Detect if HyDE would benefit this query
if self._should_use_hyde(query, intent):
    hyde_result = await self._hyde_expander.expand_query(query)
    
    # Use hypothetical documents to augment search
    augmented_query = f"{query}\n\nContext: {hyde_result.hypothetical_documents[0]}"
    
    # Execute research with augmented query
    results = await self._execute_parallel_research(augmented_query, tools)
```

### 5. When to Use HyDE

```python
def _should_use_hyde(self, query: str, intent: ResearchIntent) -> bool:
    """
    HyDE works best for:
    - Complex queries
    - Conceptual questions
    - Queries with domain-specific terminology
    
    HyDE may hurt:
    - Simple factual queries
    - Queries with specific named entities
    - Time-sensitive queries
    """
    # Complex intents benefit from HyDE
    hyde_beneficial_intents = {
        ResearchIntent.ARCHITECTURE,
        ResearchIntent.DEEP_SEMANTIC,
        ResearchIntent.COMPARISON,
        ResearchIntent.ACADEMIC
    }
    
    # Short queries (< 5 words) benefit from expansion
    is_short_query = len(query.split()) < 5
    
    return intent in hyde_beneficial_intents or is_short_query
```

---

## Acceptance Criteria

### Functional
- [ ] HyDEExpander class with configurable parameters
- [ ] Generate N hypothetical documents per query
- [ ] Optional embedding computation
- [ ] Query augmentation for keyword search
- [ ] Intent-based activation logic

### Non-Functional
- [ ] Hypothetical generation < 1s (parallel if needed)
- [ ] Improves retrieval precision by >10%
- [ ] Graceful degradation without embeddings

### Testing
- [ ] Unit tests for HyDEExpander
- [ ] Test with various query types
- [ ] A/B test: with HyDE vs without
- [ ] Measure precision/recall improvement

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `hyde_expander.py` | Create | HyDE implementation |
| `research_orchestrator.py` | Modify | Add HyDE integration |
| `tests/test_hyde_expander.py` | Create | Unit tests |

---

## Technical Notes

### LLM Integration
Uses existing Claude/OpenAI integration from platform. Pass as `llm_generator` callable.

### Embedding Options
- OpenAI: text-embedding-3-small (recommended)
- Local: sentence-transformers
- None: fallback to text augmentation

### Diversity in Hypotheticals
Generate diverse hypotheticals by:
1. Varying prompts
2. Different temperature settings
3. Asking for different perspectives

---

## Risks

| Risk | Mitigation |
|------|------------|
| LLM hallucination in hypotheticals | Use low temperature, verify factuality |
| Added latency (LLM calls) | Cache common queries, parallel generation |
| Over-expansion for simple queries | Intent-based activation |

---

**Assignee**: TBD
**Started**: -
**Completed**: -
