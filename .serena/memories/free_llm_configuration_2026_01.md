# Free LLM Configuration for Unleash Platform

## Verified Working (Jan 2026)

### Ollama (LOCAL - FREE)
- **Model**: `qwen3:8b` (8.2B parameters, Q4_K_M quantization)
- **Endpoint**: `http://localhost:11434`
- **DSPy Format**: `ollama_chat/qwen3:8b`
- **Status**: WORKING

### Voyage AI (PAID - cheap embeddings)
- **Model**: `voyage-3-lite` (512 dimensions)
- **Cost**: ~$0.00005/1K tokens
- **API Key**: Set in `.config/.env`
- **Status**: WORKING

## DSPy Configuration

```python
import dspy

# FREE local LLM
lm = dspy.LM(model='ollama_chat/qwen3:8b', api_base='http://localhost:11434')
dspy.configure(lm=lm)
```

## Available Ollama Models
- qwen3:8b (8.2B) - General purpose
- qwen3-embedding:8b (7.6B) - Embeddings
- mxbai-embed-large (334M) - Fast embeddings
- embeddinggemma (307M) - Lightweight embeddings

## Alternative Free Providers

### Groq (Cloud - FREE tier)
- 14,400 requests/day free
- Models: llama-3.3-70b-versatile, mixtral-8x7b-32768
- Requires: GROQ_API_KEY

### Together AI (Cloud - Free credits)
- Requires: TOGETHER_API_KEY

## Cost Summary
| Component | Monthly Cost |
|-----------|-------------|
| Ollama LLM | $0 |
| Voyage embeddings | ~$5 (estimated) |
| Opik (self-host) | $0 |
| **Total** | ~$5/month |

No Anthropic API key required for automated pipelines!
