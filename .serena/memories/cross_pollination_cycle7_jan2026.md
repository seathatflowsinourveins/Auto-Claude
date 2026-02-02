# Cross-Pollination: Cycle 7 → Project Workflows

## Pattern Distribution Matrix

| Pattern | WITNESS | TRADING | UNLEASH |
|---------|---------|---------|---------|
| Agent Skills API | Archetype skills | Strategy skills | Meta-skills |
| Tool Search | Dynamic MCP | Sandboxed search | Full catalog |
| Memory Tool | Creative memory | Decision audit | Cross-session |
| POPs | 2M particles | N/A | Research viz |
| ComfyTD | Real-time gen | N/A | Prototype |
| CIMD/XAA | MCP auth | Strict auth | Full OAuth |
| MCP Security | Sandbox | CRITICAL | Moderate |

---

## WITNESS Integration Priority

### 1. POPs for Particle System (HIGH)
Replace current GPU compute with native POPs:
```python
# Current: Custom GLSL compute shaders
# New: Native POPs operators

class WitnessPOPsMigration:
    def migrate_particle_system(self):
        # ParticlesGPU → ParticlesPOP
        self.pops.create("particlesPOP", params={
            "numParticles": 2_000_000,
            "forces": ["gravity", "attractor", "turbulence"],
            "archetype_color_mapping": True
        })
        
        # Trail SOP → Trail POP
        self.pops.create("trailPOP", params={
            "length": 30,  # frames
            "fade": "exponential"
        })
```

### 2. ComfyTD for Archetype Visualization (HIGH)
```python
class ArchetypeComfyTD:
    ARCHETYPE_PROMPTS = {
        "WARRIOR": "aggressive red energy, sharp angles, intense",
        "SAGE": "calm cyan wisdom, vertical alignment, deliberate",
        "JESTER": "chaotic yellow movement, erratic, playful"
    }
    
    async def generate_archetype_texture(self, archetype: str):
        return await self.comfy.generate({
            "prompt": self.ARCHETYPE_PROMPTS[archetype],
            "model": "flux2_klein_9b",
            "steps": 4,  # Fast iteration
            "size": (1024, 1024)
        })
```

### 3. Memory Tool for Session Persistence (MEDIUM)
```python
# Persist creative discoveries across sessions
await memory_tool.store({
    "type": "archetype_discovery",
    "archetype": "WARRIOR",
    "best_params": {"intensity": 0.8, "complexity": 0.6},
    "fitness_score": 0.92,
    "session_id": session.id
})
```

---

## TRADING Integration Priority

### 1. MCP Security Hardening (CRITICAL)
```python
class AlphaForgeMCPConfig:
    # CRITICAL: January 2026 vulnerability mitigation
    
    BLOCKED_SERVERS = ["git", "filesystem", "bash"]
    ALLOWED_SERVERS = {
        "questdb": {"read_only": True},
        "redis": {"operations": ["get", "set", "publish"]},
        "grafana": {"read_only": True}
    }
    
    def __init__(self):
        self.cimd_config = {
            "verify_client_id": True,
            "document_url": "https://alphaforge.internal/.well-known/cimd.json"
        }
        self.xaa_config = {
            "enabled": True,
            "trusted_apps": ["monitoring", "alerting"]
        }
```

### 2. Agent Skills for Trading (MEDIUM)
```python
# Define reusable trading skills
TRADING_SKILLS = {
    "risk_assessment": {
        "tools": ["calculator", "questdb_read"],
        "max_tokens": 2000,
        "timeout": 5000
    },
    "position_sizing": {
        "tools": ["calculator"],
        "constraints": ["max_position_pct: 0.02"]
    }
}
```

### 3. Audit-First Memory (HIGH)
```python
class TradingAuditMemory:
    async def record_decision(self, decision: TradingDecision):
        await self.memory_tool.store({
            "type": "trading_decision",
            "action": decision.action,
            "reasoning": decision.reasoning,
            "risk_score": decision.risk_score,
            "timestamp": datetime.utcnow().isoformat(),
            "audit_trail": decision.audit_trail
        })
```

---

## UNLEASH Integration Priority

### 1. Full Agent Skills API (HIGH)
```python
class UnleashAgentSkills:
    def __init__(self):
        self.skills_catalog = SkillsCatalog()
        self.tool_search = ToolSearchClient()
    
    async def discover_skills(self, task: str):
        # Use Tool Search to find relevant skills
        relevant_tools = await self.tool_search.search(
            query=task,
            catalog="unleash_mcp_servers"
        )
        
        # Load skills dynamically
        for tool in relevant_tools:
            await self.skills_catalog.load(tool.skill_id)
```

### 2. Cross-Session Memory Architecture (HIGH)
```python
class UnleashMemoryArchitecture:
    layers = {
        "episodic": EpisodicMemory(),      # Past conversations
        "semantic": SemanticMemory(),       # Facts and knowledge
        "procedural": ProceduralMemory(),   # Learned skills
        "working": WorkingMemory()          # Current context
    }
    
    async def unified_recall(self, query: str):
        results = await asyncio.gather(
            self.layers["episodic"].search(query),
            self.layers["semantic"].search(query),
            self.layers["procedural"].search(query)
        )
        return self.merge_and_rank(results)
```

### 3. Meta-Enhancement Loop (CRITICAL)
```python
class MetaEnhancementLoop:
    async def cycle(self, cycle_number: int):
        # Research
        findings = await self.parallel_research([
            "anthropic api updates",
            "mcp releases",
            "touchdesigner updates"
        ])
        
        # Analyze
        patterns = self.extract_patterns(findings)
        
        # Persist
        await self.write_memory(f"cycle{cycle_number}_findings", patterns)
        
        # Cross-pollinate
        await self.distribute_to_projects(patterns)
        
        # Schedule next
        return cycle_number + 1
```

---

## Immediate Actions

### WITNESS
1. [ ] Create POPs migration plan
2. [ ] Set up ComfyTD component
3. [ ] Integrate Memory Tool with archive

### TRADING
1. [x] Update MCP config with CIMD/XAA
2. [ ] Block vulnerable MCP servers
3. [ ] Implement audit memory trail

### UNLEASH
1. [ ] Implement Tool Search integration
2. [ ] Build unified memory architecture
3. [ ] Automate enhancement loop scheduling

---

Last Updated: 2026-01-25
Cycle: 7 Cross-Pollination
Next Cycle: 8 (Agent Skills Deep Dive)
