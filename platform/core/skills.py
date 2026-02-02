"""
UAP Skills Module - Dynamic skill loading and management.

Implements the Claude Code skill system pattern:
- Skills are modular packages extending agent capabilities
- SKILL.md files with YAML frontmatter define metadata
- Progressive disclosure: metadata → body → bundled resources
- Dynamic registration and discovery

Based on Anthropic's Claude Agent SDK patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class SkillCategory(str, Enum):
    """Categories of skills."""

    ARCHITECTURE = "architecture"      # System design, patterns
    CODING = "coding"                  # Language-specific expertise
    TESTING = "testing"                # Test frameworks, TDD
    DEVOPS = "devops"                  # CI/CD, infrastructure
    DATA = "data"                      # ETL, databases
    CREATIVE = "creative"              # Visualization, art
    RESEARCH = "research"              # Web search, analysis
    TRADING = "trading"                # Finance, markets
    ORCHESTRATION = "orchestration"    # Multi-agent, coordination
    GENERAL = "general"                # Catch-all


class SkillLoadLevel(str, Enum):
    """Progressive disclosure levels for skills."""

    METADATA = "metadata"      # Name + description only (~100 tokens)
    SUMMARY = "summary"        # First section of SKILL.md (~500 tokens)
    FULL = "full"              # Complete SKILL.md body (<5000 tokens)
    RESOURCES = "resources"    # Include bundled resources (unlimited)


# =============================================================================
# Data Models
# =============================================================================

class SkillMetadata(BaseModel):
    """Metadata from SKILL.md frontmatter."""

    name: str
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    license: Optional[str] = None
    category: SkillCategory = SkillCategory.GENERAL
    tags: List[str] = Field(default_factory=list)
    requires: List[str] = Field(default_factory=list)  # Dependencies


class SkillResource(BaseModel):
    """A bundled resource (script, reference, or asset)."""

    name: str
    path: Path
    resource_type: str  # "script", "reference", "asset"
    size_bytes: int = 0
    content: Optional[str] = None  # Loaded on demand


class Skill(BaseModel):
    """A complete skill definition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: SkillMetadata
    body: str = ""  # Markdown content from SKILL.md
    summary: str = ""  # First section for preview
    path: Path
    scripts: List[SkillResource] = Field(default_factory=list)
    references: List[SkillResource] = Field(default_factory=list)
    assets: List[SkillResource] = Field(default_factory=list)
    loaded_level: SkillLoadLevel = SkillLoadLevel.METADATA

    @property
    def token_estimate(self) -> int:
        """Estimate tokens based on content length (4 chars per token)."""
        total_chars = len(self.body)
        for ref in self.references:
            if ref.content:
                total_chars += len(ref.content)
        return total_chars // 4

    def get_context_for_level(self, level: SkillLoadLevel) -> str:
        """Get skill content appropriate for the requested level."""
        if level == SkillLoadLevel.METADATA:
            return f"**{self.metadata.name}**: {self.metadata.description}"

        if level == SkillLoadLevel.SUMMARY:
            return f"# {self.metadata.name}\n\n{self.summary}"

        if level == SkillLoadLevel.FULL:
            return f"# {self.metadata.name}\n\n{self.body}"

        # RESOURCES level - include everything
        parts = [f"# {self.metadata.name}\n\n{self.body}"]
        if self.references:
            parts.append("\n## References\n")
            for ref in self.references:
                if ref.content:
                    parts.append(f"\n### {ref.name}\n{ref.content}")
        return "\n".join(parts)


# =============================================================================
# Skill Loader
# =============================================================================

class SkillLoader:
    """Loads skills from SKILL.md files."""

    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*\n(.*)$",
        re.DOTALL
    )

    def __init__(self, skills_root: Optional[Path] = None):
        self._skills_root = skills_root or Path.cwd() / "skills"
        self._cache: Dict[str, Skill] = {}

    def load_skill(
        self,
        skill_path: Path,
        level: SkillLoadLevel = SkillLoadLevel.FULL,
    ) -> Optional[Skill]:
        """
        Load a skill from a directory containing SKILL.md.

        Args:
            skill_path: Path to skill directory
            level: How much to load (progressive disclosure)

        Returns:
            Loaded Skill or None if invalid
        """
        skill_file = skill_path / "SKILL.md"
        if not skill_file.exists():
            logger.warning(f"No SKILL.md found in {skill_path}")
            return None

        # Check cache
        cache_key = f"{skill_path}:{level.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            content = skill_file.read_text(encoding="utf-8")
            match = self.FRONTMATTER_PATTERN.match(content)

            if not match:
                logger.warning(f"Invalid SKILL.md format in {skill_path}")
                return None

            frontmatter_str, body = match.groups()

            # Parse YAML frontmatter
            frontmatter = yaml.safe_load(frontmatter_str) or {}
            metadata = SkillMetadata(
                name=frontmatter.get("name", skill_path.name),
                description=frontmatter.get("description", ""),
                version=frontmatter.get("version", "1.0.0"),
                author=frontmatter.get("author"),
                license=frontmatter.get("license"),
                category=SkillCategory(
                    frontmatter.get("category", "general")
                ) if frontmatter.get("category") else SkillCategory.GENERAL,
                tags=frontmatter.get("tags", []),
                requires=frontmatter.get("requires", []),
            )

            # Extract summary (first section)
            summary = self._extract_summary(body)

            # Create skill
            skill = Skill(
                metadata=metadata,
                body=body if level in (SkillLoadLevel.FULL, SkillLoadLevel.RESOURCES) else "",
                summary=summary,
                path=skill_path,
                loaded_level=level,
            )

            # Load resources if requested
            if level == SkillLoadLevel.RESOURCES:
                skill.scripts = self._load_resources(skill_path / "scripts", "script")
                skill.references = self._load_resources(skill_path / "references", "reference")
                skill.assets = self._load_resources(skill_path / "assets", "asset")

            self._cache[cache_key] = skill
            return skill

        except Exception as e:
            logger.error(f"Failed to load skill from {skill_path}: {e}")
            return None

    def _extract_summary(self, body: str, max_lines: int = 20) -> str:
        """Extract first section as summary."""
        lines = body.strip().split("\n")
        summary_lines = []

        for line in lines[:max_lines]:
            if line.startswith("## ") and summary_lines:
                break  # Stop at second heading
            summary_lines.append(line)

        return "\n".join(summary_lines).strip()

    def _load_resources(
        self,
        resource_dir: Path,
        resource_type: str,
    ) -> List[SkillResource]:
        """Load resources from a directory."""
        resources = []

        if not resource_dir.exists():
            return resources

        for file_path in resource_dir.rglob("*"):
            if file_path.is_file():
                content = None
                if resource_type == "reference":
                    try:
                        content = file_path.read_text(encoding="utf-8")
                    except Exception:
                        pass  # Binary file or encoding issue

                resources.append(SkillResource(
                    name=file_path.name,
                    path=file_path,
                    resource_type=resource_type,
                    size_bytes=file_path.stat().st_size,
                    content=content,
                ))

        return resources

    def discover_skills(self) -> List[Path]:
        """Discover all skill directories in the skills root."""
        if not self._skills_root.exists():
            return []

        skills = []
        for path in self._skills_root.iterdir():
            if path.is_dir() and (path / "SKILL.md").exists():
                skills.append(path)

        return skills


# =============================================================================
# Skill Registry
# =============================================================================

@dataclass
class SkillTrigger:
    """Defines when a skill should be activated."""

    patterns: List[str] = field(default_factory=list)  # Regex patterns
    keywords: List[str] = field(default_factory=list)  # Simple keywords
    categories: List[SkillCategory] = field(default_factory=list)
    custom_matcher: Optional[Callable[[str], float]] = None  # Returns relevance 0-1


class SkillRegistry:
    """
    Central registry for skill discovery and selection.

    Implements the skill matching logic that determines which skills
    are relevant for a given task or query.
    """

    def __init__(self, loader: Optional[SkillLoader] = None):
        self._loader = loader or SkillLoader()
        self._skills: Dict[str, Skill] = {}
        self._triggers: Dict[str, SkillTrigger] = {}
        self._active_skills: Set[str] = set()

    def register(
        self,
        skill: Skill,
        trigger: Optional[SkillTrigger] = None,
    ) -> None:
        """Register a skill with optional trigger configuration."""
        self._skills[skill.metadata.name] = skill

        if trigger:
            self._triggers[skill.metadata.name] = trigger
        else:
            # Default trigger based on metadata
            self._triggers[skill.metadata.name] = SkillTrigger(
                keywords=[
                    skill.metadata.name.lower(),
                    *[t.lower() for t in skill.metadata.tags],
                ],
                categories=[skill.metadata.category],
            )

        logger.info(f"Registered skill: {skill.metadata.name}")

    def unregister(self, skill_name: str) -> bool:
        """Unregister a skill by name."""
        if skill_name in self._skills:
            del self._skills[skill_name]
            self._triggers.pop(skill_name, None)
            self._active_skills.discard(skill_name)
            return True
        return False

    def load_all(self, level: SkillLoadLevel = SkillLoadLevel.METADATA) -> int:
        """Load all discovered skills at the specified level."""
        skill_paths = self._loader.discover_skills()
        count = 0

        for path in skill_paths:
            skill = self._loader.load_skill(path, level)
            if skill:
                self.register(skill)
                count += 1

        logger.info(f"Loaded {count} skills from {self._loader._skills_root}")
        return count

    def find_relevant(
        self,
        query: str,
        max_skills: int = 5,
        min_relevance: float = 0.1,
    ) -> List[tuple[Skill, float]]:
        """
        Find skills relevant to a query.

        Args:
            query: User query or task description
            max_skills: Maximum number of skills to return
            min_relevance: Minimum relevance score (0-1)

        Returns:
            List of (skill, relevance) tuples sorted by relevance
        """
        query_lower = query.lower()
        results = []

        for name, skill in self._skills.items():
            trigger = self._triggers.get(name)
            if not trigger:
                continue

            relevance = self._calculate_relevance(query_lower, skill, trigger)

            if relevance >= min_relevance:
                results.append((skill, relevance))

        # Sort by relevance descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_skills]

    def _calculate_relevance(
        self,
        query: str,
        skill: Skill,
        trigger: SkillTrigger,
    ) -> float:
        """Calculate relevance score for a skill given a query."""
        scores = []

        # Custom matcher takes precedence
        if trigger.custom_matcher:
            custom_score = trigger.custom_matcher(query)
            if custom_score > 0:
                return custom_score

        # Keyword matching
        for keyword in trigger.keywords:
            if keyword in query:
                scores.append(0.6)

        # Pattern matching
        for pattern in trigger.patterns:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    scores.append(0.8)
            except re.error:
                pass

        # Description matching (fuzzy)
        desc_words = skill.metadata.description.lower().split()
        query_words = set(query.split())
        overlap = len(set(desc_words) & query_words)
        if overlap > 0:
            scores.append(min(0.5, overlap * 0.1))

        # Name matching
        if skill.metadata.name.lower() in query:
            scores.append(0.9)

        return max(scores) if scores else 0.0

    def activate(self, skill_name: str) -> bool:
        """Mark a skill as active for the current session."""
        if skill_name in self._skills:
            self._active_skills.add(skill_name)
            return True
        return False

    def deactivate(self, skill_name: str) -> bool:
        """Deactivate a skill."""
        if skill_name in self._active_skills:
            self._active_skills.remove(skill_name)
            return True
        return False

    def get_active_context(
        self,
        level: SkillLoadLevel = SkillLoadLevel.SUMMARY,
    ) -> str:
        """Get combined context from all active skills."""
        parts = []

        for name in self._active_skills:
            skill = self._skills.get(name)
            if skill:
                parts.append(skill.get_context_for_level(level))

        return "\n\n---\n\n".join(parts) if parts else ""

    def get(self, skill_name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(skill_name)

    def list_all(self) -> List[SkillMetadata]:
        """List metadata for all registered skills."""
        return [s.metadata for s in self._skills.values()]

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_category: Dict[str, int] = {}
        for skill in self._skills.values():
            cat = skill.metadata.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_skills": len(self._skills),
            "active_skills": len(self._active_skills),
            "by_category": by_category,
            "active_names": list(self._active_skills),
        }


# =============================================================================
# Built-in Skills (Programmatic)
# =============================================================================

def create_builtin_skill(
    name: str,
    description: str,
    body: str,
    category: SkillCategory = SkillCategory.GENERAL,
    tags: Optional[List[str]] = None,
) -> Skill:
    """Create a skill programmatically (without SKILL.md file)."""
    metadata = SkillMetadata(
        name=name,
        description=description,
        category=category,
        tags=tags or [],
    )

    return Skill(
        metadata=metadata,
        body=body,
        summary=body[:500] if len(body) > 500 else body,
        path=Path("."),  # No file path for built-in
        loaded_level=SkillLoadLevel.FULL,
    )


# Pre-defined built-in skills
BUILTIN_SKILLS = {
    "ultrathink": create_builtin_skill(
        name="ultrathink",
        description="Extended deep thinking for complex decisions using full 128K token budget with structured reasoning.",
        body="""
# Ultrathink Pattern

Use ultrathink for complex, multi-faceted problems that benefit from extensive reasoning.

## When to Use
- Architectural decisions with many trade-offs
- Complex debugging with unclear root causes
- Strategic planning with multiple stakeholders
- Research synthesis across many sources

## Process
1. **Frame the Problem**: Define scope, constraints, success criteria
2. **Explore Alternatives**: Generate multiple approaches without commitment
3. **Analyze Trade-offs**: Evaluate each approach systematically
4. **Synthesize**: Combine insights into a coherent recommendation
5. **Validate**: Check reasoning for gaps, biases, and errors

## Token Budget Allocation
- 40% Problem exploration
- 30% Alternative analysis
- 20% Synthesis and validation
- 10% Buffer for course corrections
""",
        category=SkillCategory.ORCHESTRATION,
        tags=["thinking", "reasoning", "complex", "analysis"],
    ),

    "code-review": create_builtin_skill(
        name="code-review",
        description="Comprehensive code review focusing on security, performance, and maintainability.",
        body="""
# Code Review Skill

Systematic review of code changes for quality, security, and best practices.

## Review Checklist
1. **Security**: OWASP Top 10, input validation, secrets exposure
2. **Performance**: O(n) complexity, memory leaks, unnecessary operations
3. **Readability**: Naming, structure, documentation
4. **Testing**: Coverage, edge cases, mocking
5. **Architecture**: SOLID principles, coupling, cohesion

## Severity Levels
- **Critical**: Security vulnerabilities, data loss risks
- **High**: Bugs, performance issues
- **Medium**: Code quality, maintainability
- **Low**: Style, minor improvements

## Output Format
For each finding:
- Location: file:line
- Severity: Critical/High/Medium/Low
- Issue: Description
- Suggestion: How to fix
""",
        category=SkillCategory.CODING,
        tags=["review", "security", "quality", "testing"],
    ),

    "tdd-workflow": create_builtin_skill(
        name="tdd-workflow",
        description="Test-Driven Development workflow with red-green-refactor cycle.",
        body="""
# TDD Workflow

Implement features using Test-Driven Development.

## The Cycle
1. **Red**: Write a failing test that defines desired behavior
2. **Green**: Write minimum code to make the test pass
3. **Refactor**: Clean up while keeping tests green

## Principles
- Write test BEFORE implementation
- One test at a time
- Minimal code to pass
- Refactor only with green tests

## Test Structure (AAA)
- **Arrange**: Set up test data and conditions
- **Act**: Execute the code under test
- **Assert**: Verify expected outcomes

## Tips
- Test behavior, not implementation
- Use descriptive test names: `test_<unit>_<scenario>_<expected>`
- Mock external dependencies
- Keep tests fast and isolated
""",
        category=SkillCategory.TESTING,
        tags=["tdd", "testing", "workflow", "quality"],
    ),
}


# =============================================================================
# Convenience Functions
# =============================================================================

def create_skill_registry(
    skills_root: Optional[Path] = None,
    include_builtins: bool = True,
) -> SkillRegistry:
    """
    Factory function to create a configured SkillRegistry.

    Args:
        skills_root: Root directory for skill files
        include_builtins: Whether to include built-in skills

    Returns:
        Configured SkillRegistry
    """
    loader = SkillLoader(skills_root)
    registry = SkillRegistry(loader)

    if include_builtins:
        for skill in BUILTIN_SKILLS.values():
            registry.register(skill)

    return registry


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the skill system."""
    print("=" * 60)
    print("UAP Skills System Demo")
    print("=" * 60)

    # Create registry with built-ins
    registry = create_skill_registry(include_builtins=True)
    print(f"\nCreated registry with {len(registry._skills)} built-in skills")

    # List skills
    print("\n[Registered Skills]")
    for meta in registry.list_all():
        print(f"  - {meta.name}: {meta.description[:50]}...")

    # Find relevant skills
    print("\n[Skill Matching]")
    queries = [
        "I need to review this code for security issues",
        "Help me think through this architectural decision",
        "Let's write tests first before implementing",
    ]

    for query in queries:
        results = registry.find_relevant(query, max_skills=2)
        print(f"\n  Query: '{query[:40]}...'")
        for skill, score in results:
            print(f"    -> {skill.metadata.name} (relevance: {score:.2f})")

    # Activate a skill
    print("\n[Skill Activation]")
    registry.activate("ultrathink")
    registry.activate("code-review")

    context = registry.get_active_context(SkillLoadLevel.SUMMARY)
    print(f"  Active context ({len(context)} chars):")
    print(f"  {context[:200]}...")

    # Stats
    stats = registry.get_stats()
    print("\n[Registry Stats]")
    print(f"  Total: {stats['total_skills']}")
    print(f"  Active: {stats['active_skills']}")
    print(f"  By category: {stats['by_category']}")


if __name__ == "__main__":
    demo()
