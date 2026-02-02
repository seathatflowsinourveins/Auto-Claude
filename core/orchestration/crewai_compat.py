"""CrewAI compatibility layer using LangGraph patterns.

CrewAI requires Python >=3.10, <3.14 and cannot install on Python 3.14.
This layer provides equivalent multi-agent orchestration using LangGraph.

Usage:
    from core.orchestration.crewai_compat import CrewCompat, Agent, Task, AgentRole

    researcher = Agent(name="Researcher", role=AgentRole.RESEARCHER, ...)
    crew = CrewCompat(agents=[researcher], tasks=[...])
    result = await crew.kickoff()
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import asyncio

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None


class AgentRole(Enum):
    """Standard agent roles for crew orchestration."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    ANALYST = "analyst"
    PLANNER = "planner"


@dataclass
class Agent:
    """Agent definition compatible with CrewAI patterns."""
    name: str
    role: AgentRole
    goal: str
    backstory: str
    tools: List[Callable] = field(default_factory=list)
    llm_config: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = False

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task using configured LLM provider."""
        prompt = f"""You are {self.name}, a {self.role.value}.

Goal: {self.goal}

Backstory: {self.backstory}

Context from previous tasks:
{self._format_context(context)}

Current Task: {task}

Provide a detailed, high-quality response that fulfills the task requirements.
"""
        # Use provider based on config or default to anthropic
        provider_name = self.llm_config.get("provider", "anthropic")

        try:
            if provider_name == "anthropic":
                from core.providers.anthropic_provider import AnthropicProvider
                provider = AnthropicProvider()
            elif provider_name == "openai":
                from core.providers.openai_provider import OpenAIProvider
                provider = OpenAIProvider()
            else:
                from core.providers.anthropic_provider import AnthropicProvider
                provider = AnthropicProvider()

            return await provider.complete(prompt)
        except Exception as e:
            return f"Agent execution error: {e}"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary for prompt inclusion."""
        if not context:
            return "No prior context available."

        lines = []
        for key, value in context.items():
            lines.append(f"- {key}: {value[:500] if isinstance(value, str) else value}")
        return "\n".join(lines)


@dataclass
class Task:
    """Task definition with dependencies."""
    description: str
    agent: Agent
    expected_output: str
    dependencies: List['Task'] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None


@dataclass
class CrewState:
    """State object for crew execution graph."""
    current_task: Optional[str] = None
    completed_tasks: Dict[str, str] = field(default_factory=dict)
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    iteration: int = 0


class CrewCompat:
    """CrewAI-compatible multi-agent orchestration using LangGraph."""

    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        verbose: bool = False,
        max_iterations: int = 50
    ):
        self.agents = {a.name: a for a in agents}
        self.tasks = tasks
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def _build_graph(self) -> Optional[Any]:
        """Build LangGraph execution graph from tasks."""
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(CrewState)

        for i, task in enumerate(self.tasks):
            node_name = f"task_{i}"
            graph.add_node(node_name, self._create_task_node(task, i))

            if i == 0:
                graph.set_entry_point(node_name)
            else:
                graph.add_edge(f"task_{i-1}", node_name)

        if self.tasks:
            graph.add_edge(f"task_{len(self.tasks)-1}", END)

        return graph.compile()

    def _create_task_node(self, task: Task, index: int):
        """Create a node function for a task."""
        async def node(state: CrewState) -> CrewState:
            if self.verbose:
                print(f"[Crew] Executing task {index}: {task.description[:50]}...")

            agent = task.agent
            context = {**task.context, **state.completed_tasks}

            try:
                result = await agent.execute(task.description, context)
                task.output = result
                state.completed_tasks[f"task_{index}"] = result
                state.agent_outputs[agent.name] = result
                state.iteration += 1

                if self.verbose:
                    print(f"[Crew] Task {index} completed by {agent.name}")
            except Exception as e:
                state.errors.append(f"Task {index} failed: {e}")

            return state
        return node

    async def kickoff(self) -> Dict[str, Any]:
        """Execute the crew workflow."""
        if self.graph:
            initial_state = CrewState()
            final_state = await self.graph.ainvoke(initial_state)
            return {
                "completed_tasks": final_state.completed_tasks,
                "agent_outputs": final_state.agent_outputs,
                "errors": final_state.errors,
                "iterations": final_state.iteration
            }
        else:
            # Fallback sequential execution without LangGraph
            return await self._sequential_execute()

    async def _sequential_execute(self) -> Dict[str, Any]:
        """Fallback sequential execution without LangGraph."""
        state = CrewState()

        for i, task in enumerate(self.tasks):
            if self.verbose:
                print(f"[Crew] Executing task {i}: {task.description[:50]}...")

            context = {**task.context, **state.completed_tasks}

            try:
                result = await task.agent.execute(task.description, context)
                task.output = result
                state.completed_tasks[f"task_{i}"] = result
                state.agent_outputs[task.agent.name] = result
                state.iteration += 1
            except Exception as e:
                state.errors.append(f"Task {i} failed: {e}")

        return {
            "completed_tasks": state.completed_tasks,
            "agent_outputs": state.agent_outputs,
            "errors": state.errors,
            "iterations": state.iteration
        }

    def get_task_outputs(self) -> List[str]:
        """Get all task outputs in order."""
        return [t.output for t in self.tasks if t.output]


CREWAI_COMPAT_AVAILABLE = True
