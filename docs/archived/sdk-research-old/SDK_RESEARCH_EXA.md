# Exa AI Search API & Python SDK - Comprehensive Research Report

*Deep Research Report - 2026-01-19*
*Research Time: 150 seconds*

---

## Overview

Exa AI is a leading semantic and neural network search engine platform offering a powerful web search API and Python SDK (exa-py). Designed for fast, accurate, agentic search capabilities supporting RAG and deep content discovery.

---

## 1. Search API Capabilities (Exa 2.1)

### Search Modes
| Mode | Description | Latency |
|------|-------------|---------|
| `auto` | Intelligently selects best method | Varies |
| `neural` | Embeddings-based semantic search | Medium |
| `fast` | Optimized for low-latency | <500ms |
| `deep` | Agentic search with query expansion | Higher |

### Response Fields
- Titles, URLs, publication dates
- Authors, text snippets
- Highlights and summaries
- Ranked relevance scores

---

## 2. SDK Methods

### Search
```python
from exa_py import Exa
import os

exa = Exa(os.getenv('EXA_API_KEY'))

# Basic search
results = exa.search("hottest AI startups", num_results=2)

# Deep search with query variations
deep_results = exa.search(
    "blog post about AI",
    type="deep",
    additional_queries=["AI blogpost", "machine learning blogs"],
    num_results=5
)
```

### Find Similar (Content Discovery)
```python
similar_results = exa.find_similar(
    "https://example.com/article",
    exclude_source_domain=True
)
```

### Get Contents (Full Text Retrieval)
```python
contents = exa.get_contents(
    ["https://example.com/article"],
    text={"include_html_tags": True, "max_characters": 1000},
    highlights={
        "highlights_per_url": 2,
        "num_sentences": 1,
        "query": "AI advancements"
    }
)
```

### Answer Endpoint (RAG)
```python
# Direct question answering
answer = exa.answer("What are the latest advancements in AI?", text=True)
print(answer)

# Streaming answer
for chunk in exa.stream_answer("Explain quantum computing:"):
    if chunk.content:
        print(chunk.content, end='', flush=True)
    if chunk.citations:
        print("\nCitations:", chunk.citations)
```

---

## 3. Filtering Capabilities

### Domain Filtering
```python
filtered_results = exa.search(
    "latest AI research",
    include_domains=["arxiv.org", "openai.com"],
    exclude_domains=["reddit.com"]
)
```

### Date Filtering
```python
dated_results = exa.search(
    "AI breakthroughs",
    start_published_date="2025-01-01",
    end_published_date="2025-12-31",
    start_crawl_date="2025-06-01"
)
```

---

## 4. Integration Best Practices

### Secure API Key Management
```python
import os
# Store in environment variables
exa = Exa(os.getenv('EXA_API_KEY'))
```

### Choose Appropriate Search Types
- **Exa Fast**: Low-latency applications, quick lookups
- **Exa Auto**: General purpose, let API decide
- **Exa Deep**: Comprehensive research, high-quality results

### Efficient Pagination
```python
# First page
results = exa.search("query", num_results=10)

# Use offset for pagination
next_page = exa.search("query", num_results=10, offset=10)
```

### Combine with RAG Workflows
```python
# Search → Get Contents → Feed to LLM
search_results = exa.search("topic", num_results=5)
urls = [r.url for r in search_results.results]
contents = exa.get_contents(urls, text=True)
# Pass contents to Claude for synthesis
```

---

## 5. Rate Limits and Performance

| Plan | Requests/min | Results/request |
|------|-------------|-----------------|
| Free | 10 | 10 |
| Pro | 100 | 100 |
| Enterprise | Custom | Custom |

### Performance Tips
- Use `find_similar` for content discovery (fastest)
- Batch URLs with `get_contents` instead of individual calls
- Use streaming for real-time applications
- Cache results when appropriate

---

## 6. Recent Updates (2026)

- **Exa 2.1**: Three search modes (Fast, Auto, Deep)
- **Sub-500ms latency** for Fast mode
- **Answer endpoint**: Direct RAG integration
- **Streaming responses**: Real-time chunk delivery
- **Enhanced highlights**: Query-focused extraction
- **Improved semantic search**: Better embeddings

---

## Integration with Unleash Platform

```python
# In research_engine.py
from exa_py import Exa

class ResearchEngine:
    def __init__(self):
        self.exa = Exa(api_key=os.getenv("EXA_API_KEY"))

    def exa_search(self, query: str, num_results: int = 10) -> dict:
        result = self.exa.search(query, num_results=num_results)
        return {
            "success": True,
            "results": [self._format_result(r) for r in result.results],
            "data": {"results": result.results}
        }

    def exa_find_similar(self, url: str, num_results: int = 5) -> dict:
        result = self.exa.find_similar(
            url,
            num_results=num_results,
            exclude_source_domain=True
        )
        return {"success": True, "results": result.results}

    def exa_get_contents(self, urls: list) -> dict:
        result = self.exa.get_contents(urls, text=True, highlights=True)
        return {"success": True, "contents": result.results}
```

---

*Sources: [exa.ai](https://exa.ai), [docs.exa.ai](https://docs.exa.ai), [github.com/exa-labs/exa-py](https://github.com/exa-labs/exa-py)*
