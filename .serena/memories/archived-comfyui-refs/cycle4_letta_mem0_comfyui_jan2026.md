# Cycle 4 Research Findings - January 2026 Week 4

## Letta Framework Updates

### Letta Code: Memory-First Coding Agent
**Key Achievement**: #1 model-agnostic OSS harness on TerminalBench
**Core Concept**: Long-lived agents that persist across sessions and improve with use

```python
# Letta Code Architecture
class LettaCodeAgent:
    """Memory-first coding agent pattern."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.core_memory = CoreMemory()      # In-context, system prompt
        self.archival_memory = ArchivalMemory()  # Queryable via tools
        self.skills = SkillLibrary()         # Dynamically loaded
    
    async def learn_from_trajectory(self, trajectory: list[Step]):
        """Continual learning from experiences."""
        # Extract patterns from trajectory
        patterns = self.extract_patterns(trajectory)
        
        # Store successful patterns in memory
        for pattern in patterns:
            if pattern.success_rate > 0.8:
                await self.archival_memory.add(pattern)
        
        # Update skills based on learnings
        await self.skills.update_from_patterns(patterns)
```

### Skill Learning: Continual Learning for CLI Agents
**Problem**: RecoveryBench shows performance DEGRADES with errors in-context
**Solution**: Learn from trajectories, not just accumulate history

**Key Insight**: Skills (from Anthropic) = dynamically loaded reference files
- Handbooks, cheatsheets, task-specific context
- Agents load relevant skills on-demand

### Context Hierarchy (Letta Docs)
```
Priority Order (highest to lowest):
1. Memory Blocks (core memory) - persists in-context
2. Files - agent can read segments and search
3. Archival Memory - queryable via built-in tools
4. External DB (Vector/RAG) - accessed via MCP/tool calling
```

### Benchmarking Insight
**Surprising Result**: 74% accuracy on LoCoMo with just filesystem storage
**Implication**: Memory is about context management, not retrieval mechanism

## Mem0 Framework Updates

### Funding & Growth
- **$24M Funding** (Seed + Series A) led by Basis Set Ventures
- **45.8k GitHub Stars** - Universal memory layer
- Y Combinator backed

### Graph Memory for AI Agents (Jan 2026)
Top 5 graph-based memory solutions compared:
1. Mem0 (native graph support)
2. Graphiti (temporal knowledge graphs)
3. Neo4j + LangChain
4. Qdrant + Graph overlay
5. Custom implementations

### Core Mem0 Pattern
```python
from mem0 import Memory

# Initialize with user-scoped memory
memory = Memory()

# Add memories with automatic extraction
memory.add(
    "I prefer aisle seats on short flights, window on long flights",
    user_id="user_123"
)

# Self-improving retrieval
relevant = memory.search(
    "Book me a flight to NYC",
    user_id="user_123"
)
# Returns: preference for seat type based on flight duration

# Automatic conflict resolution
memory.add(
    "Actually, I always want window seats now",
    user_id="user_123"
)
# Mem0 updates the conflicting preference
```

### Memory Engineering 2026 Key Insight
**Quote**: "LLMs are NOT continuously learning. In the cleanest abstraction, an LLM is simply: tokens in → tokens out"

**Implications**:
- Memory must be EXTERNAL to the model
- Agents create ILLUSION of memory through retrieval
- Architecture matters more than model capability

## ComfyUI Updates (January 2026)

### New Models & Features
| Date | Feature | Impact |
|------|---------|--------|
| Jan 17 | WAN 2.6 Reference-to-Video | Video generation from reference images |
| Jan 15 | FLUX.2 [klein] 4B & 9B | Fast local image editing |
| Jan 15 | Preprocessor & Frame Interpolation | Smooth video workflows |
| Jan 19 | Kling 2.6 Motion Control | Advanced motion in video |
| Jan 8 | New Revamped UI | Better workflow management |

### FLUX.2 Klein for WITNESS
```json
{
  "workflow": "flux2_klein_fast_edit",
  "nodes": [
    {"type": "LoadCheckpoint", "model": "flux2_klein_9b"},
    {"type": "CLIPTextEncode", "prompt": "archetype visualization"},
    {"type": "KSampler", "steps": 4, "cfg": 1.0},
    {"type": "VAEDecode"},
    {"type": "SaveImage"}
  ],
  "performance": {
    "generation_time": "~2 seconds",
    "quality": "production-ready",
    "vram": "12GB"
  }
}
```

### WAN 2.6 Reference-to-Video for Creative Loops
```python
# Generate video from archetype reference image
async def archetype_to_video(archetype_image: str, motion_prompt: str):
    workflow = load_workflow("wan26_ref_to_video")
    workflow.set_input("reference_image", archetype_image)
    workflow.set_input("motion_prompt", motion_prompt)
    
    result = await comfyui.queue_prompt(workflow)
    return result.video_path
```

### NVIDIA RTX AI Garage
- Local video generation with LTX-2
- Sage Attention for speed on newer GPUs
- No cloud costs, full creative control

## Project Integration Recommendations

### WITNESS (Creative AI)
1. **Integrate FLUX.2 Klein** for fast iteration in MAP-Elites
2. **Use WAN 2.6** for archetype-to-video generation
3. **Deploy Letta Context Hierarchy** for creative memory
4. **Enable Kling 2.6** for motion-controlled particles

### TRADING (AlphaForge)
1. **Mem0 for trader preferences** and strategy memory
2. **Letta archival memory** for decision logging
3. **Graph memory** for market relationship tracking
4. **Skill Learning** for strategy refinement

### UNLEASH (Meta-Project)
1. **Full Mem0 integration** for cross-session context
2. **Letta Code patterns** for self-improving agents
3. **ComfyUI workflows** for documentation/visualization
4. **Memory Engineering** principles throughout

## Key Quotes

> "Memory is more about how agents manage context than the exact retrieval mechanism used." - Letta Team

> "AI agents forget. Mem0 remembers." - Mem0 Tagline

> "Instead of hoping for luck, you assemble a workflow: models, prompts, controls, refiners, upscalers" - ComfyUI Tutorial

---
Last Updated: 2026-01-25
Cycle: Enhancement Loop Cycle 4
Sources: Letta docs, Mem0 blog, ComfyUI.org, Medium, NVIDIA blog
