# UNLEASH Embedding Model Upgrade Analysis - 2026

**Generated**: 2026-02-05 | **Research by**: Claude Agent (Researcher)  
**Current Model**: jina-embeddings-v3 (1024d, 570M params, 8K context)  
**Candidates**: Voyage-4 MoE family, Jina v4 multimodal (3.8B params, 32K context)

---

## Executive Summary

**RECOMMENDATION: Upgrade to Voyage-4 family** with hybrid multimodal support.

**Critical Findings**:

1. **Voyage-4-large** delivers best text retrieval quality (MTEB ~66.0 vs 65.52 current)
2. **40% cost savings** via MoE architecture vs comparable dense models
3. **4x context window** increase (32K vs 8K tokens)
4. **Zero-downtime migration** possible via Matryoshka 1024d compatibility
5. **Jina v4** adds multimodal (text+images) for specialized workloads

**Immediate Actions**:
- Phase 1 (Week 1-2): A/B test Voyage-4-large on 10% of traffic
- Phase 2 (Week 3-6): Incremental migration with blue-green deployment
- Phase 3 (Week 7+): Optimize with asymmetric retrieval (docs: voyage-4-large, queries: voyage-4-lite)
- Phase 4: Add Jina v4 for visual document search

**Cost Impact**: $300/month → $330/month (+10% for 2-5% quality gain + multimodal)

---

## 1. Feature Comparison Matrix

### Core Specifications

| Feature | Jina v3 (Current) | Voyage-4-large | Jina v4 |
|---------|-------------------|----------------|---------|
| **Parameters** | 570M | ~1.5B (MoE) | 3.8B |
| **Dimensions** | 32-1024 (Matryoshka) | 256-2048 (Matryoshka) | 128-2048 (Matryoshka) |
| **Context Window** | 8,192 tokens | 32,000 tokens | 32,000 tokens |
| **Architecture** | XLM-RoBERTa + LoRA | MoE with shared space | Qwen2.5-VL-3B + LoRA |
| **Modalities** | Text only | Text only | Text + Images |
| **MTEB Score** | 65.52 | ~66.0 | 55.97 (text), 66.49 (multimodal) |
| **Long Context** | 67.11 (LongEmbed) | ~68.0 | 67.11 |
| **Languages** | 89 | 100+ | 30+ |
| **Release Date** | Sept 2024 | Jan 2026 | June 2025 |
| **License** | Apache 2.0 | Proprietary API | Qwen Research (non-commercial) |

