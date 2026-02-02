#!/usr/bin/env python3
"""
CrewAI Manager - Role-Based Team Orchestration
Manages crews of AI agents with specialized roles.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import CrewAI SDK
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("crewai not available - install with: pip install crewai")


class AgentSpec(BaseModel):
    """Specification for a CrewAI agent."""
    name: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = Field(default_factory=list)
    verbose: bool = Field(default=True)
    allow_delegation: bool = Field(default=False)


class TaskSpec(BaseModel):
    """Specification for a CrewAI task."""
    description: str
    expected_output: str
    agent_name: str
    context_tasks: list[str] = Field(default_factory=list)


class CrewSpec(BaseModel):
    """Specification for a complete crew."""
    name: str
    agents: list[AgentSpec]
    tasks: list[TaskSpec]
    process: str = Field(default="sequential")  # sequential or hierarchical
    verbose: bool = Field(default=True)


class CrewResult(BaseModel):
    """Result from crew execution."""
    success: bool
    output: Optional[str] = None
    task_outputs: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


if CREWAI_AVAILABLE:
    @dataclass
    class ManagedCrew:
        """
        A managed CrewAI crew with configured agents and tasks.

        Wraps CrewAI's Crew class with additional management features.
        """

        spec: CrewSpec
        agents: dict[str, Agent] = field(default_factory=dict)
        tasks: list[Task] = field(default_factory=list)
        crew: Optional[Crew] = field(default=None)

        @classmethod
        def from_spec(cls, spec: CrewSpec) -> "ManagedCrew":
            """Create a managed crew from specification."""
            instance = cls(spec=spec)
            instance._build_agents()
            instance._build_tasks()
            instance._build_crew()
            return instance

        def _build_agents(self) -> None:
            """Build agents from specifications."""
            for agent_spec in self.spec.agents:
                agent = Agent(
                    role=agent_spec.role,
                    goal=agent_spec.goal,
                    backstory=agent_spec.backstory,
                    verbose=agent_spec.verbose,
                    allow_delegation=agent_spec.allow_delegation,
                )
                self.agents[agent_spec.name] = agent

                logger.info("crewai_agent_created", name=agent_spec.name, role=agent_spec.role)

        def _build_tasks(self) -> None:
            """Build tasks from specifications."""
            task_map = {}

            for task_spec in self.spec.tasks:
                agent = self.agents.get(task_spec.agent_name)
                if not agent:
                    logger.warning(
                        "task_agent_not_found",
                        task=task_spec.description[:50],
                        agent=task_spec.agent_name,
                    )
                    continue

                # Handle context dependencies
                context = []
                for ctx_task_desc in task_spec.context_tasks:
                    if ctx_task_desc in task_map:
                        context.append(task_map[ctx_task_desc])

                task = Task(
                    description=task_spec.description,
                    expected_output=task_spec.expected_output,
                    agent=agent,
                    context=context if context else None,
                )
                self.tasks.append(task)
                task_map[task_spec.description] = task

                logger.info("crewai_task_created", description=task_spec.description[:50])

        def _build_crew(self) -> None:
            """Build the crew with agents and tasks."""
            process = (
                Process.hierarchical
                if self.spec.process == "hierarchical"
                else Process.sequential
            )

            self.crew = Crew(
                agents=list(self.agents.values()),
                tasks=self.tasks,
                process=process,
                verbose=self.spec.verbose,
            )

            logger.info(
                "crewai_crew_built",
                name=self.spec.name,
                agents=len(self.agents),
                tasks=len(self.tasks),
                process=self.spec.process,
            )

        def kickoff(self, inputs: Optional[dict[str, Any]] = None) -> CrewResult:
            """
            Execute the crew (synchronous).

            Args:
                inputs: Optional inputs for the crew

            Returns:
                CrewResult with execution details
            """
            if not self.crew:
                return CrewResult(success=False, error="Crew not built")

            logger.info("crewai_kickoff", crew=self.spec.name)

            try:
                result = self.crew.kickoff(inputs=inputs or {})

                # Extract task outputs
                task_outputs = []
                if hasattr(result, 'tasks_output'):
                    for task_output in result.tasks_output:
                        task_outputs.append({
                            "task": str(task_output.description)[:100] if hasattr(task_output, 'description') else "",
                            "output": str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output),
                        })

                return CrewResult(
                    success=True,
                    output=str(result),
                    task_outputs=task_outputs,
                )

            except Exception as e:
                logger.error("crewai_kickoff_failed", error=str(e))
                return CrewResult(success=False, error=str(e))

        async def kickoff_async(
            self,
            inputs: Optional[dict[str, Any]] = None,
        ) -> CrewResult:
            """
            Execute the crew asynchronously.

            Args:
                inputs: Optional inputs for the crew

            Returns:
                CrewResult with execution details
            """
            # Run in thread pool since CrewAI doesn't have native async
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.kickoff(inputs))


@dataclass
class CrewManager:
    """
    Manager for multiple CrewAI crews.

    Provides a high-level interface for creating, managing, and executing
    crews of AI agents.

    Usage:
        manager = CrewManager()
        crew = manager.create_crew(spec)
        result = await manager.run(crew)
    """

    crews: dict[str, Any] = field(default_factory=dict)

    def create_crew(self, spec: CrewSpec) -> Any:
        """
        Create a new crew from specification.

        Args:
            spec: The crew specification

        Returns:
            ManagedCrew instance
        """
        if not CREWAI_AVAILABLE:
            raise ImportError("crewai not installed")

        crew = ManagedCrew.from_spec(spec)
        self.crews[spec.name] = crew

        logger.info("crew_created", name=spec.name)
        return crew

    def create_default_crew(self, name: str = "default") -> Any:
        """
        Create a default research crew.

        Creates a crew with researcher, analyst, and writer agents.
        """
        spec = CrewSpec(
            name=name,
            agents=[
                AgentSpec(
                    name="researcher",
                    role="Senior Research Analyst",
                    goal="Conduct thorough research and gather comprehensive information",
                    backstory="You are an expert researcher with years of experience in data analysis and information gathering.",
                ),
                AgentSpec(
                    name="analyst",
                    role="Data Analyst",
                    goal="Analyze research findings and extract meaningful insights",
                    backstory="You are a skilled analyst who excels at finding patterns and drawing conclusions from data.",
                ),
                AgentSpec(
                    name="writer",
                    role="Technical Writer",
                    goal="Create clear, well-structured reports from analysis",
                    backstory="You are an experienced technical writer who can explain complex topics clearly.",
                ),
            ],
            tasks=[
                TaskSpec(
                    description="Research the given topic thoroughly",
                    expected_output="Comprehensive research findings",
                    agent_name="researcher",
                ),
                TaskSpec(
                    description="Analyze the research findings and identify key insights",
                    expected_output="Key insights and patterns from the research",
                    agent_name="analyst",
                    context_tasks=["Research the given topic thoroughly"],
                ),
                TaskSpec(
                    description="Write a detailed report based on the analysis",
                    expected_output="Well-structured report with findings and recommendations",
                    agent_name="writer",
                    context_tasks=["Analyze the research findings and identify key insights"],
                ),
            ],
            process="sequential",
        )

        return self.create_crew(spec)

    def get_crew(self, name: str) -> Optional[Any]:
        """Get a crew by name."""
        return self.crews.get(name)

    def list_crews(self) -> list[str]:
        """List all crew names."""
        return list(self.crews.keys())

    def delete_crew(self, name: str) -> bool:
        """Delete a crew."""
        if name in self.crews:
            del self.crews[name]
            logger.info("crew_deleted", name=name)
            return True
        return False

    async def run(
        self,
        crew: Any,
        inputs: Optional[dict[str, Any]] = None,
    ) -> CrewResult:
        """
        Run a crew with optional inputs.

        Args:
            crew: The crew to run
            inputs: Optional inputs for the crew

        Returns:
            CrewResult with execution details
        """
        return await crew.kickoff_async(inputs)

    def run_sync(
        self,
        crew: Any,
        inputs: Optional[dict[str, Any]] = None,
    ) -> CrewResult:
        """
        Run a crew synchronously.

        Args:
            crew: The crew to run
            inputs: Optional inputs for the crew

        Returns:
            CrewResult with execution details
        """
        return crew.kickoff(inputs)


def create_manager() -> CrewManager:
    """Factory function to create a CrewManager."""
    return CrewManager()


def create_agent_spec(
    name: str,
    role: str,
    goal: str,
    backstory: str,
    **kwargs,
) -> AgentSpec:
    """Helper function to create an agent specification."""
    return AgentSpec(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        **kwargs,
    )


def create_task_spec(
    description: str,
    expected_output: str,
    agent_name: str,
    context_tasks: Optional[list[str]] = None,
) -> TaskSpec:
    """Helper function to create a task specification."""
    return TaskSpec(
        description=description,
        expected_output=expected_output,
        agent_name=agent_name,
        context_tasks=context_tasks or [],
    )


if __name__ == "__main__":
    async def main():
        """Test the CrewAI manager."""
        print(f"CrewAI available: {CREWAI_AVAILABLE}")

        if CREWAI_AVAILABLE:
            try:
                manager = create_manager()
                crew = manager.create_default_crew("test-crew")

                print(f"Crew created: {manager.list_crews()}")

                # Note: Actual execution requires LLM configuration
                # result = await manager.run(crew, {"topic": "AI agents"})
                # print(f"Result: {result.success}")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Install crewai to use this module")

    asyncio.run(main())
