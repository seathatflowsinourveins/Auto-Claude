"""
SWARM COORDINATION ITERATIONS - Multi-Agent Research Patterns
==============================================================
Parallel agent execution with coordination and synthesis
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# SWARM RESEARCH TOPICS
# ============================================================================

SWARM_TOPICS = [
    # Swarm Architecture
    {"topic": "Multi-agent orchestration: hierarchical vs flat vs mesh topologies", "agents": 3},
    {"topic": "Agent communication: message passing vs shared state vs blackboard", "agents": 3},
    {"topic": "Consensus mechanisms: voting, leader election, Byzantine fault tolerance", "agents": 4},
    {"topic": "Load balancing: round-robin vs least-connections vs adaptive routing", "agents": 3},

    # Coordination Patterns
    {"topic": "MapReduce for LLM: parallel processing with aggregation", "agents": 4},
    {"topic": "Pipeline parallelism: stage-based concurrent processing", "agents": 3},
    {"topic": "Fork-join patterns: parallel subtasks with synchronization", "agents": 4},
    {"topic": "Actor model: message-driven concurrent agents", "agents": 3},

    # Swarm Intelligence
    {"topic": "Ant colony optimization: pheromone-based path finding", "agents": 5},
    {"topic": "Particle swarm optimization: collective search strategies", "agents": 5},
    {"topic": "Genetic algorithms: evolutionary agent populations", "agents": 4},
    {"topic": "Stigmergy: indirect coordination through environment", "agents": 3},

    # Production Swarms
    {"topic": "Kubernetes operators for agent orchestration", "agents": 3},
    {"topic": "Celery distributed task queues for LLM workloads", "agents": 3},
    {"topic": "Ray distributed computing for parallel inference", "agents": 4},
    {"topic": "Apache Kafka for agent event streaming", "agents": 3},

    # Advanced Patterns
    {"topic": "Self-organizing swarms: emergent behavior from simple rules", "agents": 5},
    {"topic": "Adaptive swarms: dynamic reconfiguration based on load", "agents": 4},
    {"topic": "Resilient swarms: fault tolerance and self-healing", "agents": 4},
    {"topic": "Heterogeneous swarms: specialized agents with different capabilities", "agents": 4},
]


@dataclass
class AgentResult:
    agent_id: str
    sdk: str
    sources: list
    findings: list
    latency: float


@dataclass
class SwarmResult:
    topic: str
    agents_spawned: int
    agent_results: List[AgentResult]
    synthesis: dict
    total_sources: int
    total_findings: int
    total_latency: float


class SwarmCoordinator:
    """Coordinate multi-agent research swarms."""

    def __init__(self):
        self.exa = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.stats = {
            "swarms_executed": 0,
            "agents_spawned": 0,
            "total_sources": 0,
            "total_findings": 0,
            "consensus_reached": 0
        }

    async def initialize(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] Swarm Coordinator initialized")

    async def execute_swarm(self, topic: str, num_agents: int) -> SwarmResult:
        """Execute a research swarm with multiple agents."""
        start = time.time()

        # Assign SDKs to agents
        sdk_rotation = ["exa", "tavily", "perplexity", "exa", "tavily"]
        agent_tasks = []

        for i in range(num_agents):
            sdk = sdk_rotation[i % len(sdk_rotation)]
            agent_tasks.append(self._spawn_agent(f"agent_{i}", sdk, topic))

        # Execute agents in parallel
        agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Filter valid results
        valid_results = [r for r in agent_results if isinstance(r, AgentResult)]

        # Synthesize findings
        synthesis = await self._synthesize(valid_results, topic)

        total_sources = sum(len(r.sources) for r in valid_results)
        total_findings = sum(len(r.findings) for r in valid_results)

        self.stats["swarms_executed"] += 1
        self.stats["agents_spawned"] += num_agents
        self.stats["total_sources"] += total_sources
        self.stats["total_findings"] += total_findings

        return SwarmResult(
            topic=topic,
            agents_spawned=num_agents,
            agent_results=valid_results,
            synthesis=synthesis,
            total_sources=total_sources,
            total_findings=total_findings,
            total_latency=time.time() - start
        )

    async def _spawn_agent(self, agent_id: str, sdk: str, topic: str) -> AgentResult:
        """Spawn a single research agent."""
        start = time.time()
        sources = []
        findings = []

        async with httpx.AsyncClient(timeout=60) as client:
            if sdk == "exa":
                try:
                    search = self.exa.search_and_contents(topic, type="auto", num_results=4, text=True, highlights=True)
                    sources = [{"title": r.title, "url": r.url, "text": r.text[:300] if r.text else ""} for r in search.results]
                    findings = [f"[{agent_id}:exa] {r.title}" for r in search.results[:2]]
                    if search.results and hasattr(search.results[0], 'highlights') and search.results[0].highlights:
                        findings.append(f"[{agent_id}:highlight] {search.results[0].highlights[0][:100]}")
                except:
                    pass

            elif sdk == "tavily":
                try:
                    r = await client.post('https://api.tavily.com/search', json={
                        'api_key': self.keys["tavily"],
                        'query': topic,
                        'search_depth': 'advanced',
                        'include_answer': True,
                        'max_results': 4
                    })
                    data = r.json()
                    sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:300]} for s in data.get("results", [])]
                    if data.get("answer"):
                        findings.append(f"[{agent_id}:tavily] {data['answer'][:150]}")
                except:
                    pass

            elif sdk == "perplexity":
                try:
                    r = await client.post(
                        'https://api.perplexity.ai/chat/completions',
                        headers={'Authorization': f'Bearer {self.keys["perplexity"]}'},
                        json={
                            'model': 'sonar-pro',
                            'messages': [{'role': 'user', 'content': f"Research: {topic}. Provide key insights and patterns."}],
                            'return_citations': True
                        }
                    )
                    data = r.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    citations = data.get('citations', [])
                    sources = [{"title": f"Citation {i+1}", "url": c, "text": ""} for i, c in enumerate(citations[:3])]
                    if content:
                        findings.append(f"[{agent_id}:perplexity] {content[:180]}")
                except:
                    pass

        return AgentResult(
            agent_id=agent_id,
            sdk=sdk,
            sources=sources,
            findings=findings,
            latency=time.time() - start
        )

    async def _synthesize(self, agent_results: List[AgentResult], topic: str) -> dict:
        """Synthesize findings from all agents."""
        all_findings = []
        for r in agent_results:
            all_findings.extend(r.findings)

        # Deduplicate
        seen = set()
        unique = []
        for f in all_findings:
            key = f[:40].lower()
            if key not in seen:
                seen.add(key)
                unique.append(f)

        # Identify consensus (findings mentioned by multiple agents)
        finding_counts = {}
        for f in all_findings:
            key = f[f.find(']')+1:f.find(']')+30].strip().lower() if ']' in f else f[:30].lower()
            finding_counts[key] = finding_counts.get(key, 0) + 1

        consensus_items = [k for k, v in finding_counts.items() if v > 1]
        if consensus_items:
            self.stats["consensus_reached"] += 1

        return {
            "unique_findings": len(unique),
            "total_findings": len(all_findings),
            "consensus_items": len(consensus_items),
            "top_findings": unique[:5],
            "agents_contributed": len(agent_results)
        }


async def main():
    print("="*70)
    print("SWARM COORDINATION ITERATIONS")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Swarm Topics: {len(SWARM_TOPICS)}")
    print(f"Total Agents to Spawn: {sum(t['agents'] for t in SWARM_TOPICS)}")
    print("="*70)

    coordinator = SwarmCoordinator()
    await coordinator.initialize()

    all_results = []

    for i, topic_data in enumerate(SWARM_TOPICS, 1):
        topic = topic_data["topic"]
        num_agents = topic_data["agents"]

        print(f"\n[{i:02d}] Spawning {num_agents} agents: {topic[:45]}...")

        result = await coordinator.execute_swarm(topic, num_agents)
        all_results.append(result)

        print(f"    Agents: {len(result.agent_results)}/{num_agents}")
        print(f"    Sources: {result.total_sources}, Findings: {result.total_findings}")
        print(f"    Synthesis: {result.synthesis['unique_findings']} unique, {result.synthesis['consensus_items']} consensus")
        print(f"    Latency: {result.total_latency:.1f}s")

        if result.synthesis['top_findings']:
            f = result.synthesis['top_findings'][0]
            clean = f.encode('ascii', 'replace').decode('ascii')
            print(f"    -> {clean[:55]}...")

        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*70)
    print("SWARM COORDINATION COMPLETE")
    print("="*70)

    print(f"\n  Swarms Executed: {coordinator.stats['swarms_executed']}")
    print(f"  Total Agents Spawned: {coordinator.stats['agents_spawned']}")
    print(f"  Total Sources: {coordinator.stats['total_sources']}")
    print(f"  Total Findings: {coordinator.stats['total_findings']}")
    print(f"  Consensus Reached: {coordinator.stats['consensus_reached']}")

    avg_latency = sum(r.total_latency for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Swarm Latency: {avg_latency:.1f}s")

    # Agent distribution
    print("\n  AGENT DISTRIBUTION:")
    sdk_counts = {"exa": 0, "tavily": 0, "perplexity": 0}
    for r in all_results:
        for ar in r.agent_results:
            sdk_counts[ar.sdk] = sdk_counts.get(ar.sdk, 0) + 1
    for sdk, count in sorted(sdk_counts.items(), key=lambda x: -x[1]):
        print(f"    {sdk}: {count} agents")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": coordinator.stats,
        "results": [
            {
                "topic": r.topic,
                "agents_spawned": r.agents_spawned,
                "total_sources": r.total_sources,
                "total_findings": r.total_findings,
                "synthesis": r.synthesis,
                "latency": r.total_latency
            }
            for r in all_results
        ]
    }

    with open("swarm_coordination_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to swarm_coordination_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
