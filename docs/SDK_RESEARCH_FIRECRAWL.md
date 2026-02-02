# Firecrawl SDK - Comprehensive Research Report

*Deep Research Report - 2026-01-19*
*Research Time: 221 seconds*

---

## Overview

Firecrawl SDK is a modern web data API optimized for AI/LLM applications. It handles proxies, caching, rate limits, and JavaScript-rendered content automatically.

---

## 1. SDK Methods

### Scrape
Fetch content from a single URL in multiple formats.

```python
from firecrawl import Firecrawl

firecrawl = Firecrawl(api_key="fc-YOUR-API-KEY")
doc = firecrawl.scrape("https://firecrawl.dev", formats=["markdown", "html"])
print(doc)
```

### Crawl
Recursively crawl a URL and all accessible subpages.

```python
docs = firecrawl.crawl(
    url="https://docs.firecrawl.dev",
    limit=10
)
```

### Map
Quickly retrieve all URLs of a website (optimized for speed).

```python
urls = firecrawl.map("https://example.com", limit=100)
```

### Extract
AI-powered structured data extraction with custom schemas.

```python
result = firecrawl.extract(
    ["https://example.com/pricing"],
    prompt="Extract pricing tiers and features"
)
```

### Search
Web search with customizable parameters.

```python
results = firecrawl.search("python tutorials", limit=10)
```

### Batch Scrape
Scrape multiple URLs simultaneously.

```python
job = firecrawl.batch_scrape([
    'https://firecrawl.dev',
    'https://docs.firecrawl.dev'
], formats=["markdown"])
```

---

## 2. LLM Extraction with Custom Schemas

### Using Pydantic
```python
from firecrawl import Firecrawl
from pydantic import BaseModel

class CompanyInfo(BaseModel):
    company_mission: str
    supports_sso: bool
    is_open_source: bool
    is_in_yc: bool

app = Firecrawl(api_key="fc-YOUR-API-KEY")
result = app.scrape(
    'https://firecrawl.dev',
    formats=[{
        "type": "json",
        "schema": CompanyInfo.model_json_schema()
    }],
    only_main_content=False,
    timeout=120000
)
```

### Prompt-Based Extraction
```python
result = app.scrape(
    'https://firecrawl.dev',
    formats=[{
        "type": "json",
        "prompt": "Extract the company mission from the page."
    }]
)
```

---

## 3. Crawling Strategies and Depth Control

| Parameter | Description |
|-----------|-------------|
| `limit` | Maximum pages to crawl |
| `crawlEntireDomain` | Crawl entire domain |
| `allowSubdomains` | Include subdomains |
| `max_discovery_depth` | Link levels to traverse |

```python
docs = firecrawl.crawl(
    url="https://example.com",
    limit=100,
    scrape_options={
        "formats": ["markdown", {"type": "json", "schema": {...}}],
        "proxy": "auto",
        "maxAge": 600000,
        "onlyMainContent": True,
    },
    max_discovery_depth=3
)
```

---

## 4. Output Formats

| Format | Description |
|--------|-------------|
| `markdown` | Clean markdown text |
| `summary` | AI-generated summary |
| `html` | Cleaned HTML |
| `rawHtml` | Original HTML |
| `screenshot` | Page screenshot |
| `links` | Extracted links |
| `json` | Structured JSON |
| `images` | Image URLs |
| `branding` | Site branding info |

---

## 5. Rate Limits and Async Patterns

### Rate Limits by Plan
| Plan | Concurrent Browsers | Scrape/min |
|------|---------------------|------------|
| Free | 2 | 10 |
| Starter | 5 | 50 |
| Pro | 10 | 100 |

### Async Pattern
```python
# Start async job
job = firecrawl.start_batch_scrape([
    "https://firecrawl.dev",
    "https://docs.firecrawl.dev"
], formats=["markdown"])

print(job.id)

# Poll status
status = firecrawl.get_batch_scrape_status(job.id)
print(status)
```

---

## 6. Recent Updates (2026)

- **Adaptive Web Crawling**: Crawler learns reliable selectors
- **Webhook Infrastructure**: Real-time job notifications
- **Spark Models**: New Spark 1 Pro and Mini in /agent endpoint
- **Enhanced Branding**: Better brand extraction
- **Self-Hosted Improvements**: Easier deployment
- **Unified Billing**: Merged credits and tokens
- **Semantic Crawling**: Improved content understanding
- **PDF Search**: New category support

---

## Integration with Unleash Platform

```python
# In research_engine.py
from firecrawl import FirecrawlApp

class ResearchEngine:
    def __init__(self):
        self.firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    def scrape(self, url: str, **kwargs) -> dict:
        result = self.firecrawl.scrape(url, **kwargs)
        return {"success": True, "data": self._to_dict(result)}

    def map_site(self, url: str, limit: int = 100) -> dict:
        result = self.firecrawl.map(url, limit=limit)
        data = self._to_dict(result)
        # Normalize: add 'urls' key for consistency
        if 'links' in data:
            data['urls'] = data['links']
        return {"success": True, "data": data}
```

---

*Sources: [firecrawl.dev](https://firecrawl.dev), [docs.firecrawl.dev](https://docs.firecrawl.dev), [github.com/firecrawl/firecrawl](https://github.com/firecrawl/firecrawl)*
