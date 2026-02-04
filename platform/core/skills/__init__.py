"""
Agent Skills Module - Skill discovery and progressive loading.

This module provides skill management for Claude agents:
- SKILL.md metadata parsing
- Progressive disclosure loading
- Skill discovery from directories
- Integration with claude-flow skill system

Usage:
    from core.skills import SkillLoader, Skill, SkillMetadata

    # Load skills from directory
    loader = SkillLoader()
    await loader.discover_skills("/path/to/skills")

    # Get a specific skill
    skill = await loader.get_skill("code-review")

    # List available skills
    skills = loader.list_skills()
"""

from .skill_loader import (
    Skill,
    SkillMetadata,
    SkillLoader,
    SkillRegistry,
    load_skill_from_file,
    parse_skill_metadata,
)

# Re-export enums and factory from the legacy skills module
# These were originally in core/skills.py before the package was created
from enum import Enum
from typing import Optional
from pathlib import Path


class SkillCategory(str, Enum):
    """Categories of skills."""
    ARCHITECTURE = "architecture"
    CODING = "coding"
    TESTING = "testing"
    DEVOPS = "devops"
    DATA = "data"
    CREATIVE = "creative"
    RESEARCH = "research"
    TRADING = "trading"
    ORCHESTRATION = "orchestration"
    GENERAL = "general"


class SkillLoadLevel(str, Enum):
    """Progressive disclosure levels for skills."""
    METADATA = "metadata"
    SUMMARY = "summary"
    FULL = "full"
    RESOURCES = "resources"


def create_skill_registry(
    skills_root: Optional[Path] = None,
    include_builtins: bool = True,
) -> SkillRegistry:
    """Factory function to create a configured SkillRegistry."""
    loader = SkillLoader(skills_root)
    registry = SkillRegistry(loader)
    return registry


__all__ = [
    "Skill",
    "SkillMetadata",
    "SkillLoader",
    "SkillRegistry",
    "SkillCategory",
    "SkillLoadLevel",
    "load_skill_from_file",
    "parse_skill_metadata",
    "create_skill_registry",
]
