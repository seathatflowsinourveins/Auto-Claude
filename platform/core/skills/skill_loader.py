"""
Agent Skills Loader - Parse SKILL.md metadata and load skills progressively.

This module implements Anthropic's skill system for Claude agents:
- SKILL.md metadata parsing with frontmatter support
- Progressive disclosure loading (metadata first, full content on demand)
- Directory-based skill discovery
- Integration with claude-flow skill system

SKILL.md Format:
    ---
    name: code-review
    description: Reviews code for quality, security, and best practices
    version: 1.0.0
    author: platform
    tags: [code, review, quality]
    dependencies: []
    triggers:
      - /review
      - /code-review
    ---

    # Code Review Skill

    [Full skill instructions here...]
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class SkillMetadata(BaseModel):
    """Metadata from SKILL.md frontmatter."""

    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(default="", description="Brief skill description")
    version: str = Field(default="1.0.0", description="Skill version")
    author: str = Field(default="", description="Skill author")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    dependencies: List[str] = Field(
        default_factory=list,
        description="Required skills or packages"
    )
    triggers: List[str] = Field(
        default_factory=list,
        description="Commands that invoke this skill (e.g., /review)"
    )
    model_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model capabilities needed"
    )
    context_requirements: List[str] = Field(
        default_factory=list,
        description="Required context (e.g., 'git_repo', 'codebase')"
    )
    progressive_disclosure: bool = Field(
        default=True,
        description="Load full content only when invoked"
    )
    priority: int = Field(default=50, ge=0, le=100, description="Execution priority")
    enabled: bool = Field(default=True, description="Whether skill is active")


class Skill(BaseModel):
    """A loaded skill with metadata and content."""

    metadata: SkillMetadata
    content: str = Field(default="", description="Full skill instructions")
    file_path: Optional[str] = Field(default=None, description="Source file path")
    content_hash: Optional[str] = Field(default=None, description="Content hash for caching")
    loaded_at: Optional[float] = Field(default=None, description="Load timestamp")
    invocation_count: int = Field(default=0, description="Times skill was invoked")

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.metadata.name

    @property
    def is_loaded(self) -> bool:
        """Check if full content is loaded."""
        return bool(self.content)

    def matches_trigger(self, command: str) -> bool:
        """Check if command matches any trigger."""
        command_lower = command.lower().strip()
        for trigger in self.metadata.triggers:
            trigger_lower = trigger.lower().strip()
            # Match exact or as prefix
            if command_lower == trigger_lower or command_lower.startswith(trigger_lower + " "):
                return True
        return False

    def increment_invocation(self) -> None:
        """Record skill invocation."""
        self.invocation_count += 1


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_skill_metadata(content: str) -> tuple[SkillMetadata, str]:
    """Parse SKILL.md content into metadata and body.

    Args:
        content: Full SKILL.md file content

    Returns:
        Tuple of (SkillMetadata, remaining_content)

    Raises:
        ValueError: If frontmatter is invalid
    """
    # Check for YAML frontmatter (--- delimited)
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if match:
        frontmatter_yaml = match.group(1)
        body = match.group(2)

        try:
            frontmatter_data = yaml.safe_load(frontmatter_yaml)
            if not isinstance(frontmatter_data, dict):
                frontmatter_data = {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML frontmatter: {e}")
            frontmatter_data = {}
            body = content
    else:
        # No frontmatter - try to extract from content
        frontmatter_data = _extract_metadata_from_content(content)
        body = content

    # Ensure name is present
    if "name" not in frontmatter_data:
        # Try to extract from first heading
        heading_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        if heading_match:
            frontmatter_data["name"] = _slugify(heading_match.group(1))
        else:
            frontmatter_data["name"] = "unnamed-skill"

    metadata = SkillMetadata(**frontmatter_data)
    return metadata, body.strip()


def _extract_metadata_from_content(content: str) -> dict:
    """Extract metadata from skill content when no frontmatter exists."""
    data = {}

    # Extract name from first heading
    heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if heading_match:
        data["name"] = _slugify(heading_match.group(1))

    # Extract description from first paragraph
    para_match = re.search(r'^#.+\n\n(.+?)(?:\n\n|$)', content, re.MULTILINE)
    if para_match:
        data["description"] = para_match.group(1).strip()[:200]

    # Look for trigger patterns in content
    trigger_pattern = r'(?:triggers?|commands?):\s*`([^`]+)`'
    triggers = re.findall(trigger_pattern, content, re.IGNORECASE)
    if triggers:
        data["triggers"] = triggers

    return data


def _slugify(text: str) -> str:
    """Convert text to slug format."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text


def load_skill_from_file(
    file_path: str | Path,
    progressive: bool = True
) -> Skill:
    """Load a skill from a SKILL.md file.

    Args:
        file_path: Path to SKILL.md file
        progressive: If True, only load metadata initially

    Returns:
        Skill object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {file_path}")

    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    metadata, body = parse_skill_metadata(content)

    skill = Skill(
        metadata=metadata,
        content="" if progressive else body,
        file_path=str(path.absolute()),
        content_hash=content_hash,
        loaded_at=time.time() if not progressive else None,
    )

    return skill


# =============================================================================
# Skill Registry
# =============================================================================

class SkillRegistry:
    """Registry for managing loaded skills.

    Provides skill lookup by name, trigger matching, and caching.
    """

    def __init__(self, loader=None):
        self._skills: Dict[str, Skill] = {}
        self._trigger_index: Dict[str, str] = {}  # trigger -> skill_name
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> skill_names
        self._active_skills: Set[str] = set()
        self._lock = asyncio.Lock()
        self._loader = loader

    def register(self, skill: Skill) -> None:
        """Register a skill in the registry."""
        name = skill.name
        self._skills[name] = skill

        # Index triggers
        for trigger in skill.metadata.triggers:
            trigger_lower = trigger.lower().strip()
            self._trigger_index[trigger_lower] = name

        # Index tags
        for tag in skill.metadata.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(name)

        logger.debug(f"Registered skill: {name}")

    def unregister(self, name: str) -> bool:
        """Remove a skill from the registry."""
        if name not in self._skills:
            return False

        skill = self._skills.pop(name)

        # Remove from trigger index
        for trigger in skill.metadata.triggers:
            trigger_lower = trigger.lower().strip()
            self._trigger_index.pop(trigger_lower, None)

        # Remove from tag index
        for tag in skill.metadata.tags:
            tag_lower = tag.lower()
            if tag_lower in self._tag_index:
                self._tag_index[tag_lower].discard(name)

        return True

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def find_by_trigger(self, command: str) -> Optional[Skill]:
        """Find a skill matching a trigger command."""
        command_lower = command.lower().strip()

        # Check exact match first
        if command_lower in self._trigger_index:
            return self._skills.get(self._trigger_index[command_lower])

        # Check prefix match (command with arguments)
        for trigger, skill_name in self._trigger_index.items():
            if command_lower.startswith(trigger + " "):
                return self._skills.get(skill_name)

        return None

    def find_by_tag(self, tag: str) -> List[Skill]:
        """Find all skills with a tag."""
        tag_lower = tag.lower()
        skill_names = self._tag_index.get(tag_lower, set())
        return [self._skills[name] for name in skill_names if name in self._skills]

    def list_all(self) -> List[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_enabled(self) -> List[Skill]:
        """List enabled skills only."""
        return [s for s in self._skills.values() if s.metadata.enabled]

    def find_relevant(self, query: str, max_skills: int = 5) -> List[Tuple['Skill', float]]:
        """Find skills relevant to a query string.

        Returns list of (skill, relevance_score) tuples sorted by relevance.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []

        for skill in self._skills.values():
            score = 0.0
            name_lower = skill.name.lower()
            desc_lower = (skill.metadata.description or "").lower()
            tags = {t.lower() for t in (skill.metadata.tags or [])}

            # Name match
            for word in query_words:
                if word in name_lower:
                    score += 0.5
                if word in desc_lower:
                    score += 0.3
                if word in tags:
                    score += 0.4

            if score > 0:
                results.append((skill, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_skills]

    def activate(self, skill_name: str) -> bool:
        """Activate a skill."""
        if skill_name in self._skills:
            self._active_skills.add(skill_name)
            return True
        return False

    def deactivate(self, skill_name: str) -> bool:
        """Deactivate a skill."""
        if skill_name in self._active_skills:
            self._active_skills.discard(skill_name)
            return True
        return False

    def get_active_context(self, load_level=None) -> str:
        """Get context string for active skills."""
        parts = []
        for name in self._active_skills:
            skill = self._skills.get(name)
            if skill:
                parts.append(f"{skill.name}: {skill.metadata.description or ''}")
        return "\n".join(parts) if parts else ""

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_skills": len(self._skills),
            "enabled_skills": sum(1 for s in self._skills.values() if s.metadata.enabled),
            "total_triggers": len(self._trigger_index),
            "total_tags": len(self._tag_index),
            "skills_by_tag": {
                tag: len(names) for tag, names in self._tag_index.items()
            },
        }


# =============================================================================
# Skill Loader
# =============================================================================

class SkillLoader:
    """Loader for discovering and loading skills.

    Supports progressive disclosure - metadata is loaded immediately,
    full content is loaded on demand.

    Usage:
        loader = SkillLoader()

        # Discover skills in directory
        await loader.discover_skills("/path/to/skills")

        # Get skill (loads full content if needed)
        skill = await loader.get_skill("code-review")

        # List available skills
        for skill in loader.list_skills():
            print(f"{skill.name}: {skill.metadata.description}")
    """

    # Standard skill file names
    SKILL_FILES = ["SKILL.md", "skill.md", "SKILL.MD"]

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        progressive_loading: bool = True,
    ):
        """Initialize the skill loader.

        Args:
            registry: Optional existing registry (creates new if not provided)
            progressive_loading: Enable progressive disclosure loading
        """
        self._registry = registry or SkillRegistry()
        self._progressive = progressive_loading
        self._discovered_paths: Set[str] = set()
        self._load_callbacks: List[Callable[[Skill], None]] = []

    @property
    def registry(self) -> SkillRegistry:
        """Get the skill registry."""
        return self._registry

    def on_skill_loaded(self, callback: Callable[[Skill], None]) -> None:
        """Register callback for when a skill is fully loaded."""
        self._load_callbacks.append(callback)

    async def discover_skills(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> List[Skill]:
        """Discover and register skills from a directory.

        Args:
            directory: Directory to search
            recursive: Search subdirectories

        Returns:
            List of discovered skills
        """
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Skills directory not found: {directory}")
            return []

        discovered = []

        # Find all skill files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for item in path.glob(pattern):
            if item.is_file() and item.name in self.SKILL_FILES:
                if str(item) in self._discovered_paths:
                    continue

                try:
                    skill = load_skill_from_file(item, progressive=self._progressive)
                    self._registry.register(skill)
                    self._discovered_paths.add(str(item))
                    discovered.append(skill)
                    logger.info(f"Discovered skill: {skill.name} from {item}")
                except Exception as e:
                    logger.error(f"Failed to load skill from {item}: {e}")

        return discovered

    async def load_skill_content(self, skill: Skill) -> bool:
        """Load full content for a skill (progressive disclosure).

        Args:
            skill: Skill to load content for

        Returns:
            True if content was loaded
        """
        if skill.is_loaded:
            return True

        if not skill.file_path:
            logger.warning(f"Skill {skill.name} has no file path")
            return False

        try:
            content = Path(skill.file_path).read_text(encoding="utf-8")
            _, body = parse_skill_metadata(content)
            skill.content = body
            skill.loaded_at = time.time()

            # Notify callbacks
            for callback in self._load_callbacks:
                try:
                    callback(skill)
                except Exception as e:
                    logger.error(f"Skill load callback error: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to load content for {skill.name}: {e}")
            return False

    async def get_skill(
        self,
        name: str,
        load_content: bool = True
    ) -> Optional[Skill]:
        """Get a skill by name, optionally loading full content.

        Args:
            name: Skill name
            load_content: Load full content if not already loaded

        Returns:
            Skill or None if not found
        """
        skill = self._registry.get(name)
        if skill and load_content and not skill.is_loaded:
            await self.load_skill_content(skill)
        return skill

    async def invoke_skill(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[tuple[Skill, str]]:
        """Find and invoke a skill by command.

        Args:
            command: Command that may trigger a skill (e.g., "/review file.py")
            context: Optional context for the skill

        Returns:
            Tuple of (Skill, arguments) or None if no match
        """
        skill = self._registry.find_by_trigger(command)
        if not skill:
            return None

        # Extract arguments (everything after the trigger)
        args = ""
        for trigger in skill.metadata.triggers:
            if command.lower().startswith(trigger.lower()):
                args = command[len(trigger):].strip()
                break

        # Load content if needed
        if not skill.is_loaded:
            await self.load_skill_content(skill)

        skill.increment_invocation()
        return (skill, args)

    def list_skills(self, enabled_only: bool = True) -> List[Skill]:
        """List all skills.

        Args:
            enabled_only: Only return enabled skills

        Returns:
            List of skills
        """
        if enabled_only:
            return self._registry.list_enabled()
        return self._registry.list_all()

    def list_triggers(self) -> Dict[str, str]:
        """List all triggers and their skill names."""
        return dict(self._registry._trigger_index)

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        stats = self._registry.get_stats()
        stats["discovered_paths"] = len(self._discovered_paths)
        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

# Global loader instance
_default_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get the global skill loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


async def discover_skills(directory: str | Path) -> List[Skill]:
    """Discover skills using the global loader."""
    loader = get_skill_loader()
    return await loader.discover_skills(directory)


async def get_skill(name: str) -> Optional[Skill]:
    """Get a skill using the global loader."""
    loader = get_skill_loader()
    return await loader.get_skill(name)


async def invoke_skill(command: str) -> Optional[tuple[Skill, str]]:
    """Invoke a skill using the global loader."""
    loader = get_skill_loader()
    return await loader.invoke_skill(command)
