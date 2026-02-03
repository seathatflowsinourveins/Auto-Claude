"""
PRODUCTION DEPLOYMENT ITERATIONS - Packaging & Deployment
===========================================================
Production-ready deployment patterns and configurations
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import httpx

load_dotenv('.config/.env')

# ============================================================================
# DEPLOYMENT TOPICS
# ============================================================================

DEPLOYMENT_TOPICS = [
    # Containerization
    {"topic": "Docker for LLM applications: multi-stage builds, GPU support, optimization", "category": "container"},
    {"topic": "Docker Compose for agent orchestration: services, networks, volumes", "category": "container"},
    {"topic": "Kubernetes deployment: pods, services, ingress for LLM apps", "category": "container"},
    {"topic": "Helm charts for LLM: templating, values, dependencies", "category": "container"},

    # CI/CD
    {"topic": "GitHub Actions for LLM: testing, building, deploying agents", "category": "cicd"},
    {"topic": "GitLab CI for ML: pipeline stages, artifacts, environments", "category": "cicd"},
    {"topic": "ArgoCD for GitOps: declarative agent deployments", "category": "cicd"},
    {"topic": "Tekton pipelines for ML: cloud-native CI/CD", "category": "cicd"},

    # Infrastructure
    {"topic": "Terraform for LLM infrastructure: AWS, GCP, Azure resources", "category": "infra"},
    {"topic": "Pulumi for agent infrastructure: TypeScript IaC", "category": "infra"},
    {"topic": "AWS CDK for LLM: Lambda, ECS, SageMaker patterns", "category": "infra"},
    {"topic": "Serverless Framework for agents: functions, events, resources", "category": "infra"},

    # Monitoring & Ops
    {"topic": "Prometheus for LLM metrics: custom metrics, alerting rules", "category": "ops"},
    {"topic": "Grafana dashboards for agents: latency, tokens, costs", "category": "ops"},
    {"topic": "ELK stack for LLM logs: Elasticsearch, Logstash, Kibana", "category": "ops"},
    {"topic": "Datadog for LLM observability: APM, logs, metrics integration", "category": "ops"},

    # Security & Compliance
    {"topic": "Vault for LLM secrets: API keys, credentials, rotation", "category": "security"},
    {"topic": "RBAC for agent APIs: authentication, authorization, policies", "category": "security"},
    {"topic": "SOC2 compliance for LLM: audit logs, data handling, encryption", "category": "security"},
    {"topic": "GDPR for AI: data retention, right to deletion, consent", "category": "security"},

    # Scaling & Performance
    {"topic": "Horizontal pod autoscaler for LLM: CPU, memory, custom metrics", "category": "scaling"},
    {"topic": "KEDA for event-driven scaling: queue depth, HTTP requests", "category": "scaling"},
    {"topic": "Istio service mesh for agents: traffic management, observability", "category": "scaling"},
    {"topic": "Ray Serve for model serving: batching, autoscaling, multi-model", "category": "scaling"},
]


@dataclass
class DeploymentResult:
    topic: str
    category: str
    sources: list
    findings: list
    config: dict
    latency: float


class DeploymentExecutor:
    """Research and generate deployment configurations."""

    def __init__(self):
        self.exa = None
        self.keys = {
            "tavily": os.getenv("TAVILY_API_KEY"),
            "jina": os.getenv("JINA_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        }
        self.stats = {"sources": 0, "findings": 0, "configs": 0}
        self.configs = []

    async def initialize(self):
        from exa_py import Exa
        self.exa = Exa(os.getenv("EXA_API_KEY"))
        print("[OK] Deployment Executor initialized")

    async def research_and_configure(self, topic: str, category: str) -> dict:
        """Research deployment pattern and generate config."""
        result = {"sources": [], "findings": [], "config": {}}

        async with httpx.AsyncClient(timeout=90) as client:
            tasks = [
                self._exa_search(topic),
                self._tavily_search(client, topic),
                self._perplexity_search(client, topic, category),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for r in responses:
                if isinstance(r, dict):
                    result["sources"].extend(r.get("sources", []))
                    result["findings"].extend(r.get("findings", []))

            # Generate configuration
            result["config"] = self._generate_config(category, topic)

        return result

    async def _exa_search(self, topic: str) -> dict:
        try:
            search = self.exa.search_and_contents(topic, type="auto", num_results=4, text=True)
            sources = [{"title": r.title, "url": r.url, "text": r.text[:400] if r.text else ""} for r in search.results]
            findings = [f"[exa] {r.title}" for r in search.results[:2]]
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _tavily_search(self, client: httpx.AsyncClient, topic: str) -> dict:
        try:
            r = await client.post('https://api.tavily.com/search', json={
                'api_key': self.keys["tavily"],
                'query': topic,
                'search_depth': 'advanced',
                'include_answer': True,
                'max_results': 4
            })
            data = r.json()
            sources = [{"title": s.get("title", ""), "url": s.get("url", ""), "text": s.get("content", "")[:400]} for s in data.get("results", [])]
            findings = [f"[tavily] {data.get('answer', '')[:180]}"] if data.get("answer") else []
            return {"sources": sources, "findings": findings}
        except:
            return {"sources": [], "findings": []}

    async def _perplexity_search(self, client: httpx.AsyncClient, topic: str, category: str) -> dict:
        try:
            r = await client.post(
                'https://api.perplexity.ai/chat/completions',
                headers={'Authorization': f'Bearer {self.keys["perplexity"]}'},
                json={
                    'model': 'sonar-pro',
                    'messages': [{'role': 'user', 'content': f"Production deployment guide: {topic}. Include configurations and best practices."}],
                    'return_citations': True
                }
            )
            data = r.json()
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            findings = [f"[perplexity] {content[:200]}"] if content else []
            return {"sources": [], "findings": findings}
        except:
            return {"sources": [], "findings": []}

    def _generate_config(self, category: str, topic: str) -> dict:
        """Generate deployment configuration based on category."""
        configs = {
            "container": {
                "dockerfile": {
                    "base_image": "python:3.11-slim",
                    "multi_stage": True,
                    "gpu_support": "nvidia/cuda:12.1-runtime",
                    "optimizations": ["--no-cache-dir", "multi-stage build", "non-root user"]
                },
                "docker_compose": {
                    "version": "3.8",
                    "services": ["agent", "redis", "qdrant"],
                    "networks": ["agent-network"],
                    "volumes": ["agent-data"]
                },
                "kubernetes": {
                    "deployment": {"replicas": 3, "strategy": "RollingUpdate"},
                    "service": {"type": "ClusterIP", "port": 8000},
                    "ingress": {"class": "nginx", "tls": True}
                }
            },
            "cicd": {
                "github_actions": {
                    "triggers": ["push", "pull_request"],
                    "jobs": ["test", "build", "deploy"],
                    "environments": ["staging", "production"],
                    "secrets": ["API_KEYS", "DOCKER_CREDENTIALS"]
                },
                "stages": ["lint", "test", "build", "push", "deploy"],
                "artifacts": ["docker-image", "test-reports", "coverage"]
            },
            "infra": {
                "terraform": {
                    "providers": ["aws", "kubernetes"],
                    "modules": ["vpc", "eks", "rds", "elasticache"],
                    "backends": ["s3", "dynamodb-lock"]
                },
                "resources": {
                    "compute": "EKS/GKE/AKS",
                    "database": "RDS/CloudSQL",
                    "cache": "ElastiCache/Memorystore",
                    "storage": "S3/GCS"
                }
            },
            "ops": {
                "prometheus": {
                    "metrics": ["llm_request_duration", "llm_tokens_total", "llm_errors_total"],
                    "alerts": ["HighLatency", "ErrorRate", "TokenBudget"]
                },
                "grafana": {
                    "dashboards": ["agent-overview", "cost-tracking", "latency-analysis"],
                    "datasources": ["prometheus", "loki", "elasticsearch"]
                },
                "logging": {
                    "format": "json",
                    "fields": ["timestamp", "level", "trace_id", "span_id", "message"],
                    "retention": "30d"
                }
            },
            "security": {
                "vault": {
                    "secrets_engines": ["kv-v2", "transit"],
                    "auth_methods": ["kubernetes", "approle"],
                    "policies": ["agent-read", "admin-full"]
                },
                "rbac": {
                    "roles": ["admin", "operator", "viewer"],
                    "permissions": ["read", "write", "execute", "admin"]
                },
                "compliance": {
                    "soc2": ["audit-logs", "encryption", "access-control"],
                    "gdpr": ["data-retention", "consent", "deletion"]
                }
            },
            "scaling": {
                "hpa": {
                    "metrics": ["cpu", "memory", "custom/llm_queue_depth"],
                    "min_replicas": 2,
                    "max_replicas": 20,
                    "target_utilization": 70
                },
                "keda": {
                    "triggers": ["prometheus", "rabbitmq", "redis"],
                    "cooldown": 300,
                    "polling_interval": 30
                },
                "service_mesh": {
                    "platform": "istio",
                    "features": ["traffic-splitting", "circuit-breaker", "retry"]
                }
            }
        }

        config = configs.get(category, {})
        config["category"] = category
        config["status"] = "generated"

        self.stats["configs"] += 1
        self.configs.append(config)

        return config

    async def run_iteration(self, topic_data: dict, index: int) -> DeploymentResult:
        topic = topic_data["topic"]
        category = topic_data["category"]

        print(f"\n[{index:02d}] [{category}] {topic[:50]}...")

        start = time.time()
        result = await self.research_and_configure(topic, category)
        latency = time.time() - start

        self.stats["sources"] += len(result["sources"])
        self.stats["findings"] += len(result["findings"])

        print(f"    Src:{len(result['sources'])} Find:{len(result['findings'])} Config:{result['config'].get('status', 'none')} [{latency:.1f}s]")

        return DeploymentResult(
            topic=topic,
            category=category,
            sources=result["sources"],
            findings=result["findings"],
            config=result["config"],
            latency=latency
        )


async def main():
    print("="*70)
    print("PRODUCTION DEPLOYMENT ITERATIONS")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Topics: {len(DEPLOYMENT_TOPICS)}")
    print("="*70)

    executor = DeploymentExecutor()
    await executor.initialize()

    all_results = []

    # Group by category
    by_category = {}
    for t in DEPLOYMENT_TOPICS:
        by_category.setdefault(t["category"], []).append(t)

    iteration = 0
    for category, topics in by_category.items():
        print(f"\n{'='*70}")
        print(f"CATEGORY: {category.upper()}")
        print(f"{'='*70}")

        for topic_data in topics:
            iteration += 1
            try:
                result = await executor.run_iteration(topic_data, iteration)
                all_results.append(result)
            except Exception as e:
                print(f"    [ERR] {str(e)[:50]}")

            await asyncio.sleep(0.3)

    # Summary
    print("\n" + "="*70)
    print("PRODUCTION DEPLOYMENT COMPLETE")
    print("="*70)

    print(f"\n  Iterations: {len(all_results)}")
    print(f"  Total Sources: {executor.stats['sources']}")
    print(f"  Total Findings: {executor.stats['findings']}")
    print(f"  Configs Generated: {executor.stats['configs']}")

    avg_latency = sum(r.latency for r in all_results) / len(all_results) if all_results else 0
    print(f"  Avg Latency: {avg_latency:.1f}s")

    print("\n  BY CATEGORY:")
    cat_stats = {}
    for r in all_results:
        cat_stats[r.category] = cat_stats.get(r.category, 0) + 1
    for cat, count in sorted(cat_stats.items()):
        print(f"    {cat}: {count}")

    print("\n  CONFIG TYPES GENERATED:")
    config_types = set()
    for c in executor.configs:
        config_types.add(c.get("category", "unknown"))
    for ct in sorted(config_types):
        print(f"    - {ct}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "stats": executor.stats,
        "configs": executor.configs,
        "results": [
            {
                "topic": r.topic,
                "category": r.category,
                "sources": len(r.sources),
                "findings": r.findings[:2],
                "config": r.config,
                "latency": r.latency
            }
            for r in all_results
        ]
    }

    with open("production_deployment_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to production_deployment_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
