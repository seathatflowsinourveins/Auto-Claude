# Anthropic Message Batches API Research 2026

**Research Date:** 2026-02-05  
**Status:** Complete

## Executive Summary

50% cost savings, up to 100K requests per batch, 24h processing, supports all Messages API features.

## Official Documentation

- API: https://platform.claude.com/docs/en/api/creating-message-batches
- Guide: https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing
- Cookbook: https://platform.claude.com/cookbook/misc-batch-processing
- SDK: https://github.com/anthropics/anthropic-sdk-python

## Cost Savings (50% Discount)

| Model | Batch Input | Batch Output |
|-------|-------------|--------------|
| Opus 4.5 | $2.50/MTok | $12.50/MTok |
| Sonnet 4.5 | $1.50/MTok | $7.50/MTok |
| Haiku 4.5 | $0.50/MTok | $2.50/MTok |

Stackable with prompt caching (30-98% hit rates).

## Python SDK

```python
import anthropic
client = anthropic.Anthropic()

# Create batch
batch = client.messages.batches.create(requests=[...])

# Monitor
batch = client.messages.batches.retrieve(batch_id)

# Stream results
for result in client.messages.batches.results(batch_id):
    print(result)
```

## Request Structure

```json
{
  "custom_id": "unique-id",
  "params": {
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "..."}]
  }
}
```

## Result Types

1. **succeeded** - Contains message response
2. **errored** - Invalid request or server error
3. **canceled** - User canceled (no charge)
4. **expired** - 24h timeout (no charge)

## Best Practices

- Use `custom_id` for result matching (order not guaranteed)
- Poll every 60s with `retrieve()`
- Stream results with `.results()` for large batches
- Retry transient errors: exponential backoff (1s, 2s, 4s)
- No native webhooks - implement polling
- Test single request with Messages API first
- Prompt caching: use `ttl="1h"` for batches

## Limits

- Max: 100,000 requests OR 256 MB
- Processing: 24h max (most <1h)
- Results: 29 days availability
- Rate limits: Separate from Messages API
- Workspace-scoped isolation

## Error Handling

```python
def retry_with_backoff(func, max_retries=3):
    delays = [1, 2, 4]
    for attempt, delay in enumerate(delays):
        try:
            return func()
        except APIError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                time.sleep(delay)
            elif e.status_code >= 400 and e.status_code < 500:
                raise  # Don't retry client errors
```

## Monitoring Pattern

```python
def monitor_batch(batch_id, polling_interval=60):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        time.sleep(polling_interval)
```

## Webhook Integration

**No native webhook support** as of 2026-02-05.

Workarounds:
- Poll every 60s with exponential backoff
- Use integration platforms (n8n, Temporal)
- Implement custom notification service

## Prompt Caching

```python
{
  "system": [
    {"type": "text", "text": "Instructions"},
    {
      "type": "text",
      "text": "<long context>",
      "cache_control": {"type": "ephemeral", "ttl": "1h"}
    }
  ]
}
```

## Supported Features

- Vision, tool use, system prompts
- Multi-turn conversations
- Extended thinking
- Structured outputs
- All beta features

## Use Cases

- Large-scale evaluations
- Content moderation
- Data analysis
- Bulk content generation
- Offline batch processing

## Key Notes

- Results NOT in request order (use `custom_id`)
- Concurrent async processing
- Delete only after processing ends
- Cancellation may lag (state: `canceling`)
- Results streaming recommended

## References

- [Creating Message Batches](https://platform.claude.com/docs/en/api/creating-message-batches)
- [Batch Processing Guide](https://platform.claude.com/docs/en/docs/build-with-claude/batch-processing)
- [Cookbook Examples](https://platform.claude.com/cookbook/misc-batch-processing)
- [SDK Implementation](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/resources/beta/messages/batches.py)
- [API Announcement](https://www.anthropic.com/news/message-batches-api)
- [Temporal Integration](https://stevekinney.com/writing/anthropic-batch-api-with-temporal)
- [n8n Workflow](https://n8n.io/workflows/3409-batch-process-prompts-with-anthropic-claude-api/)
