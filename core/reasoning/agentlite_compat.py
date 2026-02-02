"""AgentLite compatibility - lightweight ReAct agent framework.

AgentLite does not exist on PyPI. This layer provides equivalent
lightweight ReAct-style agent capabilities.

Usage:
    from core.reasoning.agentlite_compat import AgentLiteCompat, Tool

    async def search_web(input: str) -> str:
        return f"Search results for: {input}"

    agent = AgentLiteCompat(
        tools=[Tool("search", "Search the web", search_web)],
        max_iterations=5,
        verbose=True
    )

    result = await agent.run("Find the current Python version")
    print(f"Answer: {result['final_answer']}")
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable, Union, Tuple
from enum import Enum
import re
import asyncio


class ActionType(Enum):
    """Types of actions an agent can take."""
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    FINISH = "finish"


@dataclass
class Action:
    """Represents an agent action."""
    type: ActionType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    result: Optional[str] = None


@dataclass
class Tool:
    """Tool definition for agent use."""
    name: str
    description: str
    func: Callable[..., Awaitable[str]]
    parameters: Dict[str, Any] = field(default_factory=dict)

    async def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        try:
            return await self.func(**kwargs)
        except Exception as e:
            return f"Tool error: {e}"


@dataclass
class AgentState:
    """Tracks agent execution state."""
    goal: str
    thoughts: List[str] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    final_answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 10

    @property
    def is_complete(self) -> bool:
        """Check if agent has finished."""
        return self.final_answer is not None or self.iteration >= self.max_iterations


class AgentLiteCompat:
    """Lightweight ReAct-style agent compatible with AgentLite patterns."""

    ACTION_PATTERN = re.compile(
        r'Action\s*(?:\d+)?:\s*(\w+)\[([^\]]*)\]',
        re.IGNORECASE
    )

    THOUGHT_PATTERN = re.compile(
        r'Thought\s*(?:\d+)?:\s*(.+?)(?=Action|Observation|$)',
        re.DOTALL | re.IGNORECASE
    )

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        self.tools = {t.name.lower(): t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name.lower()] = tool

    def remove_tool(self, name: str) -> bool:
        """Remove a tool by name."""
        name_lower = name.lower()
        if name_lower in self.tools:
            del self.tools[name_lower]
            return True
        return False

    def _build_prompt(self, state: AgentState) -> str:
        """Build the reasoning prompt for the agent."""
        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
        )

        if not tools_desc:
            tools_desc = "No tools available."

        # Build history
        history_lines = []
        for i in range(len(state.thoughts)):
            if i < len(state.thoughts):
                history_lines.append(f"Thought {i+1}: {state.thoughts[i]}")
            if i < len(state.actions):
                action = state.actions[i]
                history_lines.append(f"Action {i+1}: {action.type.value}[{action.content}]")
            if i < len(state.observations):
                history_lines.append(f"Observation {i+1}: {state.observations[i]}")

        history = "\n".join(history_lines) if history_lines else "No history yet."

        return f"""You are a reasoning agent using the ReAct (Reasoning + Acting) pattern.

For each step:
1. Thought: Reason about the current situation and what to do next
2. Action: Take an action using one of these formats:
   - think[your reasoning] - for additional thinking
   - act[tool_name: input] - to use a tool
   - observe[what to check] - to make an observation
   - finish[your final answer] - when you have the answer

Available tools:
{tools_desc}

Goal: {state.goal}

Previous steps:
{history}

Now provide your next thought and action:
Thought {state.iteration + 1}:"""

    def _parse_response(self, response: str) -> Tuple[str, Action]:
        """Parse thought and action from LLM response."""
        # Extract thought
        thought = ""
        thought_match = self.THOUGHT_PATTERN.search(response)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract action
        action_match = self.ACTION_PATTERN.search(response)

        if action_match:
            action_type_str = action_match.group(1).lower()
            action_content = action_match.group(2).strip()

            if action_type_str == "finish":
                return thought, Action(ActionType.FINISH, action_content)
            elif action_type_str == "act":
                # Parse tool:input format
                parts = action_content.split(":", 1)
                tool_name = parts[0].strip()
                tool_input = parts[1].strip() if len(parts) > 1 else ""
                return thought, Action(
                    ActionType.ACT,
                    action_content,
                    tool_name=tool_name,
                    tool_input={"input": tool_input}
                )
            elif action_type_str == "think":
                return thought, Action(ActionType.THINK, action_content)
            elif action_type_str == "observe":
                return thought, Action(ActionType.OBSERVE, action_content)

        # Default to think action with the thought
        return thought, Action(ActionType.THINK, thought or "Processing...")

    async def _execute_action(self, action: Action) -> str:
        """Execute an action and return observation."""
        if action.type == ActionType.ACT and action.tool_name:
            tool_name = action.tool_name.lower()
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                return await tool.execute(**(action.tool_input or {}))
            else:
                return f"Tool '{action.tool_name}' not found. Available: {list(self.tools.keys())}"
        elif action.type == ActionType.THINK:
            return "Thought recorded."
        elif action.type == ActionType.OBSERVE:
            return f"Observed: {action.content}"

        return ""

    async def run(self, goal: str) -> Dict[str, Any]:
        """Run the agent to achieve a goal."""
        state = AgentState(goal=goal, max_iterations=self.max_iterations)

        try:
            from core.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider()
        except ImportError:
            return {
                "success": False,
                "error": "No LLM provider available",
                "goal": goal,
                "final_answer": None,
                "iterations": 0
            }

        while not state.is_complete:
            if self.verbose:
                print(f"[Agent] Iteration {state.iteration + 1}/{state.max_iterations}")

            # Get LLM response
            prompt = self._build_prompt(state)
            try:
                response = await provider.complete(prompt)
            except Exception as e:
                state.observations.append(f"LLM error: {e}")
                break

            # Parse response
            thought, action = self._parse_response(response)

            state.thoughts.append(thought)
            state.actions.append(action)

            if self.verbose:
                print(f"[Agent] Thought: {thought[:100]}...")
                print(f"[Agent] Action: {action.type.value}[{action.content[:50]}...]")

            # Check for finish
            if action.type == ActionType.FINISH:
                state.final_answer = action.content
                break

            # Execute action and get observation
            observation = await self._execute_action(action)
            state.observations.append(observation)
            action.result = observation

            if self.verbose:
                print(f"[Agent] Observation: {observation[:100]}...")

            state.iteration += 1

        return {
            "success": state.final_answer is not None,
            "goal": state.goal,
            "final_answer": state.final_answer,
            "iterations": state.iteration,
            "thoughts": state.thoughts,
            "actions": [{"type": a.type.value, "content": a.content} for a in state.actions],
            "observations": state.observations
        }

    async def run_with_context(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run agent with additional context."""
        enhanced_goal = f"{goal}\n\nContext:\n"
        for key, value in context.items():
            enhanced_goal += f"- {key}: {value}\n"

        return await self.run(enhanced_goal)


# Convenience function for creating tools
def create_tool(
    name: str,
    description: str,
    func: Callable[..., Awaitable[str]]
) -> Tool:
    """Create a tool from an async function."""
    return Tool(name=name, description=description, func=func)


AGENTLITE_COMPAT_AVAILABLE = True
