"""
ITERATION LOOP - Continuous Deep Dive Research
===============================================
Runs continuous research iterations using all 5 APIs with the Ultimate Research Swarm.
"""

import asyncio
import os
import sys
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv('.config/.env')

# Add platform to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'platform'))

from core.ultimate_research_swarm import (
    UltimateResearchSwarm,
    ResearchDepth,
    get_ultimate_swarm,
)


# Research topics for deep dive iteration - ROUND 3: EMERGING TECH
RESEARCH_TOPICS = [
    # Round 3: Next-Gen Architectures
    "Mixture of Experts MoE vs dense transformer architecture tradeoffs 2026",
    "Sparse attention vs linear attention vs state space models comparison",

    # Round 3: Agentic Workflows
    "Tool use patterns: function calling vs structured outputs vs JSON mode",
    "Multi-modal agents: vision-language-action models production deployment",

    # Round 3: Fine-tuning & Adaptation
    "LoRA vs QLoRA vs DoRA vs ReLoRA fine-tuning comparison 2026",
    "Continual learning vs catastrophic forgetting mitigation strategies",

    # Round 3: Evaluation & Testing
    "LLM evaluation frameworks: RAGAS vs DeepEval vs OpenAI Evals comparison",
    "Red teaming and adversarial testing for AI agents best practices",

    # Round 3: Enterprise Patterns
    "AI gateway patterns: LiteLLM vs Portkey vs AI Router comparison",
    "Observability stack for LLM applications: LangSmith vs Langfuse vs Helicone",
]


async def run_iteration(swarm: UltimateResearchSwarm, query: str, iteration: int) -> dict:
    """Run a single research iteration."""
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: {query[:60]}...")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Deep dive with all features
        result = await swarm.deep_dive(
            query,
            include_reasoning=True,
            include_deepsearch=True,
            memory_key=f"iteration_{iteration}",
        )

        latency = time.time() - start_time

        # Extract key metrics
        metrics = {
            "iteration": iteration,
            "query": query,
            "agents": result.agents_spawned,
            "sources": len(result.sources),
            "confidence": result.confidence,
            "tools": result.tools_used,
            "latency_s": latency,
            "key_findings": result.key_findings[:3] if result.key_findings else [],
            "timestamp": datetime.now().isoformat(),
        }

        print(f"\n  ✓ Agents: {metrics['agents']}")
        print(f"  ✓ Sources: {metrics['sources']}")
        print(f"  ✓ Confidence: {metrics['confidence']:.0%}")
        print(f"  ✓ Latency: {latency:.1f}s")
        print(f"  ✓ Tools: {', '.join(metrics['tools'][:5])}")

        if result.key_findings:
            print(f"\n  Key Findings:")
            for i, finding in enumerate(result.key_findings[:3], 1):
                print(f"    {i}. {finding[:100]}...")

        return metrics

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            "iteration": iteration,
            "query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def main():
    """Main iteration loop."""
    print("="*70)
    print("ITERATION LOOP - CONTINUOUS DEEP DIVE RESEARCH")
    print("="*70)
    print(f"Topics: {len(RESEARCH_TOPICS)}")
    print(f"Start Time: {datetime.now().isoformat()}")
    print("="*70)

    # Initialize swarm
    swarm = get_ultimate_swarm()
    await swarm.initialize()
    print("\n✓ Ultimate Research Swarm initialized")

    # Run iterations
    all_results = []
    total_sources = 0
    total_agents = 0

    for i, topic in enumerate(RESEARCH_TOPICS, 1):
        result = await run_iteration(swarm, topic, i)
        all_results.append(result)

        if "sources" in result:
            total_sources += result["sources"]
            total_agents += result["agents"]

        # Brief pause between iterations
        if i < len(RESEARCH_TOPICS):
            await asyncio.sleep(1)

    # Summary
    print("\n" + "="*70)
    print("ITERATION LOOP COMPLETE")
    print("="*70)

    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]

    print(f"\n  Total Iterations: {len(all_results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total Sources: {total_sources}")
    print(f"  Total Agent Calls: {total_agents}")

    if successful:
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_latency = sum(r["latency_s"] for r in successful) / len(successful)
        print(f"  Avg Confidence: {avg_confidence:.0%}")
        print(f"  Avg Latency: {avg_latency:.1f}s")

    # Save results
    output_file = "iteration_loop_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total_iterations": len(all_results),
                "successful": len(successful),
                "failed": len(failed),
                "total_sources": total_sources,
                "total_agents": total_agents,
                "avg_confidence": avg_confidence if successful else 0,
                "avg_latency_s": avg_latency if successful else 0,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
