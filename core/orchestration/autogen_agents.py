#!/usr/bin/env python3
"""
AutoGen Agent Orchestration
Conversational AI agents with multi-agent patterns.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Callable
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import AutoGen SDK
try:
    from autogen import (
        AssistantAgent,
        UserProxyAgent,
        GroupChat,
        GroupChatManager,
        config_list_from_json,
        config_list_from_models,
    )
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    logger.warning("autogen not available - install with: pip install pyautogen")


class LLMConfig(BaseModel):
    """LLM configuration for AutoGen agents."""
    model: str = Field(default="gpt-4")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)


class AgentConfig(BaseModel):
    """Configuration for an AutoGen agent."""
    name: str
    system_message: str
    is_termination_msg: Optional[Callable[[dict], bool]] = None
    human_input_mode: str = Field(default="NEVER")  # NEVER, TERMINATE, ALWAYS
    max_consecutive_auto_reply: int = Field(default=10)
    code_execution_config: Optional[dict[str, Any]] = None


class GroupChatConfig(BaseModel):
    """Configuration for an AutoGen group chat."""
    name: str
    agents: list[str]
    max_round: int = Field(default=12)
    speaker_selection_method: str = Field(default="auto")  # auto, round_robin, random


class ConversationResult(BaseModel):
    """Result from an AutoGen conversation."""
    success: bool
    chat_history: list[dict[str, Any]] = Field(default_factory=list)
    last_message: Optional[str] = None
    error: Optional[str] = None


def get_default_llm_config() -> dict[str, Any]:
    """Get default LLM configuration."""
    # Try to get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        return {
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": api_key,
                }
            ],
            "temperature": 0.7,
        }
    else:
        # Return empty config for testing without API key
        return {
            "config_list": [],
            "temperature": 0.7,
        }


if AUTOGEN_AVAILABLE:
    @dataclass
    class AutoGenConversation:
        """
        Manages an AutoGen conversation between agents.

        Wraps AutoGen's agent classes with additional management features.
        """

        agents: dict[str, Any] = field(default_factory=dict)
        llm_config: dict[str, Any] = field(default_factory=get_default_llm_config)
        chat_history: list[dict[str, Any]] = field(default_factory=list)

        def add_assistant(
            self,
            name: str,
            system_message: str,
            **kwargs,
        ) -> Any:
            """
            Add an assistant agent to the conversation.

            Args:
                name: Agent name
                system_message: System prompt for the agent
                **kwargs: Additional agent configuration

            Returns:
                The created AssistantAgent
            """
            agent = AssistantAgent(
                name=name,
                system_message=system_message,
                llm_config=self.llm_config,
                **kwargs,
            )
            self.agents[name] = agent

            logger.info("autogen_assistant_added", name=name)
            return agent

        def add_user_proxy(
            self,
            name: str = "user_proxy",
            human_input_mode: str = "NEVER",
            max_consecutive_auto_reply: int = 10,
            code_execution_config: Optional[dict[str, Any]] = None,
            **kwargs,
        ) -> Any:
            """
            Add a user proxy agent to the conversation.

            Args:
                name: Agent name
                human_input_mode: Input mode (NEVER, TERMINATE, ALWAYS)
                max_consecutive_auto_reply: Max auto replies
                code_execution_config: Code execution settings
                **kwargs: Additional configuration

            Returns:
                The created UserProxyAgent
            """
            exec_config = code_execution_config or {"use_docker": False}

            agent = UserProxyAgent(
                name=name,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply,
                code_execution_config=exec_config,
                **kwargs,
            )
            self.agents[name] = agent

            logger.info("autogen_user_proxy_added", name=name)
            return agent

        def create_group_chat(
            self,
            agents: Optional[list[str]] = None,
            max_round: int = 12,
            speaker_selection_method: str = "auto",
        ) -> tuple[Any, Any]:
            """
            Create a group chat with multiple agents.

            Args:
                agents: List of agent names to include
                max_round: Maximum conversation rounds
                speaker_selection_method: How to select next speaker

            Returns:
                Tuple of (GroupChat, GroupChatManager)
            """
            if agents:
                agent_list = [self.agents[name] for name in agents if name in self.agents]
            else:
                agent_list = list(self.agents.values())

            if len(agent_list) < 2:
                raise ValueError("Group chat requires at least 2 agents")

            group_chat = GroupChat(
                agents=agent_list,
                messages=[],
                max_round=max_round,
                speaker_selection_method=speaker_selection_method,
            )

            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=self.llm_config,
            )

            logger.info(
                "autogen_group_chat_created",
                agents=len(agent_list),
                max_round=max_round,
            )

            return group_chat, manager

        def initiate_chat(
            self,
            sender_name: str,
            receiver_name: str,
            message: str,
            **kwargs,
        ) -> ConversationResult:
            """
            Initiate a chat between two agents.

            Args:
                sender_name: Name of the sending agent
                receiver_name: Name of the receiving agent
                message: Initial message

            Returns:
                ConversationResult with chat history
            """
            sender = self.agents.get(sender_name)
            receiver = self.agents.get(receiver_name)

            if not sender or not receiver:
                return ConversationResult(
                    success=False,
                    error=f"Agent not found: {sender_name if not sender else receiver_name}",
                )

            logger.info(
                "autogen_chat_initiated",
                sender=sender_name,
                receiver=receiver_name,
            )

            try:
                result = sender.initiate_chat(
                    receiver,
                    message=message,
                    **kwargs,
                )

                # Extract chat history
                history = []
                if hasattr(receiver, 'chat_messages') and sender in receiver.chat_messages:
                    history = receiver.chat_messages[sender]

                last_msg = history[-1]["content"] if history else None

                return ConversationResult(
                    success=True,
                    chat_history=history,
                    last_message=last_msg,
                )

            except Exception as e:
                logger.error("autogen_chat_failed", error=str(e))
                return ConversationResult(success=False, error=str(e))

        async def initiate_chat_async(
            self,
            sender_name: str,
            receiver_name: str,
            message: str,
            **kwargs,
        ) -> ConversationResult:
            """Async version of initiate_chat."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.initiate_chat(sender_name, receiver_name, message, **kwargs),
            )

        def get_agent(self, name: str) -> Optional[Any]:
            """Get an agent by name."""
            return self.agents.get(name)

        def list_agents(self) -> list[str]:
            """List all agent names."""
            return list(self.agents.keys())


@dataclass
class AutoGenOrchestrator:
    """
    Orchestrator for AutoGen conversations and agents.

    Provides a high-level interface for creating and managing
    AutoGen multi-agent systems.

    Usage:
        orchestrator = AutoGenOrchestrator()
        conv = orchestrator.create_conversation()
        conv.add_assistant("coder", "You are an expert programmer")
        conv.add_user_proxy("user")
        result = await orchestrator.run(conv, "user", "coder", "Write hello world")
    """

    conversations: dict[str, AutoGenConversation] = field(default_factory=dict)

    def create_conversation(
        self,
        name: str = "default",
        llm_config: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new conversation.

        Args:
            name: Conversation name
            llm_config: LLM configuration

        Returns:
            AutoGenConversation instance
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError("autogen not installed")

        conv = AutoGenConversation(
            llm_config=llm_config or get_default_llm_config(),
        )
        self.conversations[name] = conv

        logger.info("autogen_conversation_created", name=name)
        return conv

    def create_coding_conversation(
        self,
        name: str = "coding",
    ) -> Any:
        """
        Create a conversation optimized for coding tasks.

        Creates a coder agent and user proxy with code execution enabled.
        """
        conv = self.create_conversation(name)

        conv.add_assistant(
            name="coder",
            system_message="""You are an expert software engineer.
You write clean, efficient, and well-documented code.
When asked to write code, provide complete, working implementations.
If there are errors, debug and fix them.
""",
        )

        conv.add_user_proxy(
            name="executor",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "coding_workspace",
                "use_docker": False,
            },
        )

        return conv

    def create_research_conversation(
        self,
        name: str = "research",
    ) -> Any:
        """
        Create a conversation optimized for research tasks.

        Creates researcher and critic agents.
        """
        conv = self.create_conversation(name)

        conv.add_assistant(
            name="researcher",
            system_message="""You are a senior research analyst.
You gather comprehensive information, analyze data, and provide insights.
Be thorough and cite your sources when applicable.
""",
        )

        conv.add_assistant(
            name="critic",
            system_message="""You are a critical reviewer.
You evaluate research findings for accuracy, completeness, and bias.
Provide constructive feedback and suggest improvements.
""",
        )

        conv.add_user_proxy(
            name="coordinator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
        )

        return conv

    async def run(
        self,
        conversation: Any,
        sender: str,
        receiver: str,
        message: str,
    ) -> ConversationResult:
        """
        Run a conversation between agents.

        Args:
            conversation: The conversation to use
            sender: Sender agent name
            receiver: Receiver agent name
            message: Initial message

        Returns:
            ConversationResult with chat history
        """
        return await conversation.initiate_chat_async(sender, receiver, message)

    def run_sync(
        self,
        conversation: Any,
        sender: str,
        receiver: str,
        message: str,
    ) -> ConversationResult:
        """Synchronous version of run."""
        return conversation.initiate_chat(sender, receiver, message)

    def get_conversation(self, name: str) -> Optional[Any]:
        """Get a conversation by name."""
        return self.conversations.get(name)

    def list_conversations(self) -> list[str]:
        """List all conversation names."""
        return list(self.conversations.keys())

    def delete_conversation(self, name: str) -> bool:
        """Delete a conversation."""
        if name in self.conversations:
            del self.conversations[name]
            return True
        return False


def create_orchestrator() -> AutoGenOrchestrator:
    """Factory function to create an AutoGen orchestrator."""
    return AutoGenOrchestrator()


def create_default_conversation() -> Any:
    """Create a default conversation with basic agents."""
    if not AUTOGEN_AVAILABLE:
        raise ImportError("autogen not installed")

    conv = AutoGenConversation()
    conv.add_assistant(
        name="assistant",
        system_message="You are a helpful AI assistant.",
    )
    conv.add_user_proxy(
        name="user",
        human_input_mode="NEVER",
    )
    return conv


if __name__ == "__main__":
    async def main():
        """Test the AutoGen orchestrator."""
        print(f"AutoGen available: {AUTOGEN_AVAILABLE}")

        if AUTOGEN_AVAILABLE:
            try:
                orchestrator = create_orchestrator()
                conv = orchestrator.create_coding_conversation()

                print(f"Agents: {conv.list_agents()}")

                # Note: Actual execution requires LLM API key
                # result = await orchestrator.run(
                #     conv,
                #     "executor",
                #     "coder",
                #     "Write a Python function that calculates fibonacci numbers"
                # )
                # print(f"Success: {result.success}")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Install pyautogen to use this module")

    asyncio.run(main())
