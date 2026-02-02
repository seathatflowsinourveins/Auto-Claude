#!/usr/bin/env python3
"""
browser-use Integration for UNLEASH Platform
============================================

CRITICAL GAP: This fills the ONLY missing capability in UNLEASH.

browser-use provides AI-driven browser automation via Playwright with:
- Vision-based DOM analysis
- Natural language task description
- Multi-LLM support (Claude, GPT-4, Gemini)
- Stealth anti-detection for web scraping
- BU 2.0 (Jan 2026) with +12% accuracy

Installation: pip install browser-use

Created: 2026-01-31 (V26 Iteration)
Source: https://github.com/browser-use/browser-use
"""

from __future__ import annotations

import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger("browser_use_integration")

# Optional imports with fallback
# browser-use v0.11.5 API (2026-01-31)
BROWSER_USE_AVAILABLE = False
Agent = None
BrowserSession = None
BrowserProfile = None
ChatAnthropic = None

try:
    from browser_use.agent.service import Agent
    from browser_use.browser.session import BrowserSession
    from browser_use.browser.profile import BrowserProfile
    from langchain_anthropic import ChatAnthropic
    BROWSER_USE_AVAILABLE = True
    logger.info("browser-use v0.11.5 imported successfully")
except ImportError as e:
    logger.warning(f"browser-use not installed: {e}. Run: pip install browser-use langchain-anthropic")


@dataclass
class BrowserUseConfig:
    """Configuration for browser-use integration."""
    # Browser settings
    headless: bool = True
    disable_security: bool = False  # Only True for local testing
    proxy: Optional[str] = None

    # Agent settings
    model: str = "claude-sonnet-4-20250514"
    use_vision: bool = True
    max_steps: int = 20

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    # Output settings
    save_screenshots: bool = True
    screenshot_path: Path = field(default_factory=lambda: Path.home() / ".claude" / "browser_screenshots")

    # Session persistence
    cookies_path: Optional[Path] = None
    local_storage_path: Optional[Path] = None


class BrowserUseAgent:
    """
    UNLEASH integration for browser-use SDK.

    Provides AI-driven browser automation for:
    - GitHub navigation and documentation scraping
    - Form filling and multi-page workflows
    - Visual verification of web content
    - Cross-session browser state persistence
    """

    def __init__(self, config: Optional[BrowserUseConfig] = None):
        if not BROWSER_USE_AVAILABLE:
            raise ImportError(
                "browser-use not available. Install with: pip install browser-use langchain-anthropic"
            )

        self.config = config or BrowserUseConfig()
        self._browser: Optional[Browser] = None
        self._llm = None

        # Ensure screenshot directory exists
        self.config.screenshot_path.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _initialize(self):
        """Initialize browser and LLM."""
        # Initialize LLM
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

        if not BROWSER_USE_AVAILABLE:
            raise ImportError("browser-use not available")

        # browser-use v0.11.5 imports (2026-01-31)
        from browser_use.agent.service import Agent as BUAgent
        from browser_use.browser.session import BrowserSession
        from browser_use.browser.profile import BrowserProfile
        from langchain_anthropic import ChatAnthropic as LangChainAnthropic

        self._llm = LangChainAnthropic(
            model=self.config.model,
            api_key=api_key
        )

        # Initialize browser profile
        browser_profile = BrowserProfile(
            headless=self.config.headless,
        )

        # Create browser session
        self._browser = BrowserSession(profile=browser_profile)
        await self._browser.start()

        logger.info("browser-use v0.11.5 initialized successfully")

    async def close(self):
        """Close browser and cleanup."""
        if self._browser:
            await self._browser.close()
            self._browser = None

    async def run_task(
        self,
        task: str,
        url: Optional[str] = None,
        max_steps: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Run a browser automation task.

        Args:
            task: Natural language description of the task
            url: Optional starting URL
            max_steps: Override max steps from config
            callbacks: Optional list of callback functions

        Returns:
            Dict with result, screenshots, and metadata

        Example:
            result = await agent.run_task(
                task="Navigate to GitHub and find the latest Claude SDK release",
                url="https://github.com/anthropics"
            )
        """
        if not self._browser or not self._llm:
            await self._initialize()

        steps = max_steps or self.config.max_steps

        # Create browser-use agent
        agent = Agent(
            task=task,
            llm=self._llm,
            browser=self._browser,
            use_vision=self.config.use_vision,
            max_steps=steps,
        )

        # Navigate to URL if provided
        if url:
            page = await self._browser.get_current_page()
            await page.goto(url)

        # Run the task with retries
        result = None
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                result = await agent.run()
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        if result is None and last_error:
            raise last_error

        # Save screenshot if configured
        screenshot_path = None
        if self.config.save_screenshots:
            screenshot_path = self.config.screenshot_path / f"task_{hash(task) % 10000}.png"
            try:
                page = await self._browser.get_current_page()
                await page.screenshot(path=str(screenshot_path))
            except Exception as e:
                logger.warning(f"Failed to save screenshot: {e}")

        return {
            "result": result,
            "task": task,
            "url": url,
            "steps_taken": agent.steps_taken if hasattr(agent, 'steps_taken') else None,
            "screenshot": str(screenshot_path) if screenshot_path else None,
        }

    async def scrape_documentation(
        self,
        url: str,
        extract_pattern: str = "documentation"
    ) -> Dict[str, Any]:
        """
        Scrape documentation from a website.

        Args:
            url: URL to scrape
            extract_pattern: What to extract (documentation, api, examples)

        Returns:
            Dict with extracted content and metadata
        """
        task = f"""
        Navigate to {url} and extract the {extract_pattern}.

        Instructions:
        1. Load the page and wait for content
        2. Identify the main {extract_pattern} content
        3. Extract text, code examples, and structure
        4. Return the extracted content in markdown format
        """

        return await self.run_task(task, url)

    async def fill_form(
        self,
        url: str,
        form_data: Dict[str, str],
        submit: bool = False
    ) -> Dict[str, Any]:
        """
        Fill out a web form.

        Args:
            url: URL with the form
            form_data: Dict of field_name -> value
            submit: Whether to submit the form

        Returns:
            Dict with result and metadata
        """
        fields = "\n".join([f"- {k}: {v}" for k, v in form_data.items()])

        task = f"""
        Navigate to {url} and fill out the form with these values:
        {fields}

        {"After filling all fields, submit the form." if submit else "Do not submit the form yet."}
        """

        return await self.run_task(task, url)

    async def navigate_github(
        self,
        repo: str,
        action: str = "releases"
    ) -> Dict[str, Any]:
        """
        Navigate GitHub repository.

        Args:
            repo: Repository in format "owner/repo"
            action: What to do (releases, issues, pulls, code)

        Returns:
            Dict with extracted information
        """
        url = f"https://github.com/{repo}"

        task_templates = {
            "releases": f"Navigate to {url}/releases and extract the latest release information including version, date, and changelog.",
            "issues": f"Navigate to {url}/issues and list the top 5 open issues with their titles and labels.",
            "pulls": f"Navigate to {url}/pulls and list recent pull requests with their status.",
            "code": f"Navigate to {url} and describe the repository structure and main technologies used.",
        }

        task = task_templates.get(action, task_templates["code"])
        return await self.run_task(task, url)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def quick_browse(task: str, url: Optional[str] = None) -> str:
    """
    Quick one-off browser task.

    Args:
        task: What to do
        url: Optional starting URL

    Returns:
        Result string

    Example:
        result = await quick_browse(
            "Find the latest version of the Anthropic Python SDK",
            "https://pypi.org/project/anthropic/"
        )
    """
    async with BrowserUseAgent() as agent:
        result = await agent.run_task(task, url)
        return str(result.get("result", ""))


async def scrape_docs(url: str) -> str:
    """Quick documentation scraper."""
    async with BrowserUseAgent() as agent:
        result = await agent.scrape_documentation(url)
        return str(result.get("result", ""))


async def github_releases(repo: str) -> str:
    """Get latest GitHub releases for a repo."""
    async with BrowserUseAgent() as agent:
        result = await agent.navigate_github(repo, "releases")
        return str(result.get("result", ""))


# =============================================================================
# MAIN (Demo)
# =============================================================================

async def demo():
    """Demo the browser-use integration."""
    print("=" * 60)
    print("browser-use Integration Demo")
    print("=" * 60)

    if not BROWSER_USE_AVAILABLE:
        print("ERROR: browser-use not installed")
        print("Run: pip install browser-use langchain-anthropic")
        return

    # Demo: Get latest Anthropic SDK release
    print("\nTask: Find latest Anthropic Python SDK version...")

    try:
        async with BrowserUseAgent(BrowserUseConfig(headless=True)) as agent:
            result = await agent.navigate_github("anthropics/anthropic-sdk-python", "releases")
            print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(demo())
