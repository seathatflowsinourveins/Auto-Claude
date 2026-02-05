# Anthropic Prompt Caching Guide - V68

## Overview

Prompt caching enables **90% cost reduction** on repeated API calls by caching static content (system prompts, tool definitions, large context). The UNLEASH platform provides a comprehensive wrapper around Anthropic's prompt caching API with:

- **TTL Support**: 5-minute (default) or 1-hour cache duration
- **Model-Specific Thresholds**: Automatic minimum token detection per model
- **Cost Tracking**: Detailed savings reports from API responses
- **Model Routing Integration**: Automatic cache + optimal model selection

## Quick Start

```python
from core.orchestration.prompt_cache import (
    create_cached_messages_params,
    create_routed_cached_message_async,
    CacheTTL,
    get_cache_savings_report,
)
import anthropic

# Basic usage - automatic caching
client = anthropic.Anthropic()
params = create_cached_messages_params(
    messages=[{"role": "user", "content": "Analyze this document..."}],
    system_prompt=LARGE_SYSTEM_PROMPT,  # Will be cached if >= 1024 tokens
)
response = client.messages.create(**params)

# With model routing - optimal model + caching
response = await create_routed_cached_message_async(
    task="Design a secure authentication system",
    messages=[{"role": "user", "content": "Implement OAuth2 + PKCE"}],
    system_prompt=AUTH_CONTEXT,
    system_ttl=CacheTTL.ONE_HOUR,  # Use 1h cache for static content
)

# Check savings
report = get_cache_savings_report()
print(f"Savings: ${report['cost_breakdown']['actual_savings_usd']:.4f}")
```

## Cache TTL Options

### 5-Minute Cache (Default)
- **Cost**: 1.25x base input price for writes, 0.1x for reads
- **Use when**: Content is accessed frequently (> 1x per 5 minutes)
- **TTL refreshes**: On each cache hit (no additional cost)

```python
# Default 5-minute TTL
params = create_cached_messages_params(
    messages=messages,
    system_prompt=prompt,
)
```

### 1-Hour Cache
- **Cost**: 2x base input price for writes, 0.1x for reads
- **Use when**:
  - Prompts accessed less frequently than every 5 minutes
  - Agentic workflows taking > 5 minutes per iteration
  - Long-running conversations with slow user responses
  - Latency reduction is critical

```python
from core.orchestration.prompt_cache import CacheTTL

params = create_cached_messages_params(
    messages=messages,
    system_prompt=large_rag_context,
    system_ttl=CacheTTL.ONE_HOUR,  # 1h cache for infrequent access
    tools_ttl=CacheTTL.ONE_HOUR,   # Tools rarely change
)
```

## Model-Specific Requirements

Different Claude models have different minimum token requirements:

| Model | Min Tokens | Notes |
|-------|------------|-------|
| Claude Opus 4.5 | 4096 | Highest quality, highest cost |
| Claude Opus 4/4.1 | 1024 | |
| Claude Sonnet 4/4.5 | 1024 | Best balance |
| Claude Haiku 4.5 | 4096 | Fast, requires larger context |
| Claude Haiku 3/3.5 | 2048 | Fastest, lowest cost |

The platform automatically detects and applies these thresholds:

```python
from core.orchestration.prompt_cache import get_min_tokens_for_model

# Get threshold for a model
threshold = get_min_tokens_for_model("claude-opus-4-5-20251101")  # Returns 4096
```

## Pricing Reference

### Cost per Million Tokens (MTok)

| Model | Base Input | 5m Write | 1h Write | Cache Read |
|-------|------------|----------|----------|------------|
| Opus 4.5 | $5.00 | $6.25 | $10.00 | $0.50 |
| Opus 4/4.1 | $15.00 | $18.75 | $30.00 | $1.50 |
| Sonnet 4/4.5 | $3.00 | $3.75 | $6.00 | $0.30 |
| Haiku 4.5 | $1.00 | $1.25 | $2.00 | $0.10 |
| Haiku 3.5 | $0.80 | $1.00 | $1.60 | $0.08 |
| Haiku 3 | $0.25 | $0.30 | $0.50 | $0.03 |

### Break-Even Analysis

- **5-minute cache**: Break-even at 2 calls
- **1-hour cache**: Break-even at ~3 calls
- **At 10+ calls**: ~90% savings regardless of TTL

## Integration with Model Routing

The prompt cache integrates with the 3-tier model routing system:

```python
from core.orchestration.prompt_cache import create_routed_cached_message

# Automatically routes to optimal model AND applies caching
params = create_routed_cached_message(
    task="Fix this typo in config.py",  # Routes to Haiku (Tier 1)
    messages=[{"role": "user", "content": "Fix the typo"}],
    system_prompt=CODE_CONTEXT,
)

params = create_routed_cached_message(
    task="Design a distributed consensus algorithm",  # Routes to Opus (Tier 3)
    messages=[{"role": "user", "content": "Implement Raft"}],
    system_prompt=ARCHITECTURE_DOCS,
    system_ttl=CacheTTL.ONE_HOUR,
)
```

## Tracking Cache Performance

### From API Responses

```python
from core.orchestration.prompt_cache import get_prompt_cache_manager

# After each API call, update stats from response
manager = get_prompt_cache_manager()
manager.update_from_api_response(response.usage.model_dump())

# Get detailed report
from core.orchestration.prompt_cache import get_cache_savings_report
report = get_cache_savings_report(model="claude-sonnet-4-20250514")

print(report)
# {
#     "statistics": {
#         "total_requests": 100,
#         "cache_hits": 85,
#         "cache_misses": 15,
#         "hit_rate": 0.85,
#         "api_metrics": {
#             "cache_read_tokens": 1000000,
#             "cache_creation_tokens": 50000,
#             ...
#         }
#     },
#     "cost_breakdown": {
#         "uncached_cost_usd": 3.15,
#         "cached_cost_usd": 0.475,
#         "actual_savings_usd": 2.675,
#         "savings_percentage": 84.92
#     },
#     "recommendation": "Excellent cache hit rate (85.0%). Actual savings: $2.6750"
# }
```

## Best Practices

### 1. Structure Your Prompts for Caching

Place static content at the beginning:
```
1. Tools (rarely change) -> cache with 1h TTL
2. System prompt (static instructions) -> cache with 1h TTL
3. RAG context (changes daily) -> cache with 5m TTL
4. Conversation history -> cache at each turn
```

### 2. Use Static Markers for Important Content

```python
manager = get_prompt_cache_manager()
manager.mark_static(system_prompt, ttl=CacheTTL.ONE_HOUR)
```

### 3. Batch Similar Requests

Cache hits are more likely when requests use identical prefixes:
```python
# Good: Same system prompt for all requests
for query in queries:
    params = create_cached_messages_params(
        messages=[{"role": "user", "content": query}],
        system_prompt=SHARED_SYSTEM_PROMPT,  # Cached
    )
```

### 4. Monitor Hit Rates

```python
report = get_cache_savings_report()
if report["statistics"]["hit_rate"] < 0.5:
    print("Low hit rate - consider restructuring prompts")
```

## What Invalidates Cache

Changes to these will break the cache:
- Tool definitions (any change)
- Web search toggle
- Citations toggle
- tool_choice parameter
- Images (adding/removing)
- Extended thinking parameters

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `create_cached_messages_params()` | Create API params with caching |
| `create_cached_message_async()` | Create + execute async with caching |
| `create_cached_message_sync()` | Create + execute sync with caching |
| `create_routed_cached_message()` | Routing + caching combined |
| `get_cache_savings_report()` | Detailed savings analysis |

### Classes

| Class | Description |
|-------|-------------|
| `PromptCacheManager` | Main cache management |
| `PromptCacheConfig` | Configuration with model-specific settings |
| `CacheTTL` | Enum: FIVE_MINUTES, ONE_HOUR |
| `CacheStats` | Aggregate statistics |

### Helper Functions

| Function | Description |
|----------|-------------|
| `get_min_tokens_for_model(model)` | Get minimum cacheable tokens |
| `get_pricing_for_model(model)` | Get pricing configuration |
| `get_prompt_cache_manager()` | Get global singleton |

## Sources

- [Anthropic Prompt Caching Documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Anthropic Cookbook - Prompt Caching](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb)
- Platform implementation: `platform/core/orchestration/prompt_cache.py`
- Tests: `platform/tests/test_prompt_cache.py` (102 tests)
