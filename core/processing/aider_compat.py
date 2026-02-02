"""Aider compatibility layer for AI-assisted code editing.

Aider requires Python >=3.10, <3.13 and cannot install on Python 3.14.
This layer provides equivalent git-aware code modification capabilities.

Usage:
    from core.processing.aider_compat import AiderCompat

    aider = AiderCompat("/path/to/repo", auto_commit=True)
    aider.add_file("src/main.py")

    result = await aider.run("Add error handling to the main function")
    print(f"Applied: {result['applied_edits']}")
"""
import subprocess
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class EditBlock:
    """Represents a code edit with search/replace semantics."""
    file_path: str
    search: str  # Original code to find
    replace: str  # New code to replace with

    def apply(self, base_path: Path) -> Tuple[bool, str]:
        """Apply edit to file. Returns (success, message)."""
        full_path = base_path / self.file_path

        if not full_path.exists():
            return False, f"File not found: {self.file_path}"

        try:
            content = full_path.read_text(encoding='utf-8')

            if self.search not in content:
                return False, f"Search text not found in {self.file_path}"

            new_content = content.replace(self.search, self.replace, 1)
            full_path.write_text(new_content, encoding='utf-8')
            return True, f"Applied edit to {self.file_path}"
        except Exception as e:
            return False, f"Error applying edit: {e}"


@dataclass
class AiderSession:
    """Tracks an Aider-style editing session."""
    repo_path: Path
    edited_files: List[str] = field(default_factory=list)
    pending_edits: List[EditBlock] = field(default_factory=list)
    applied_edits: List[EditBlock] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


class AiderCompat:
    """Aider-compatible AI code editing with git integration."""

    EDIT_BLOCK_PATTERN = re.compile(
        r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
        re.DOTALL
    )

    def __init__(
        self,
        repo_path: str,
        auto_commit: bool = True,
        commit_prefix: str = "aider:"
    ):
        self.repo_path = Path(repo_path).resolve()
        self.auto_commit = auto_commit
        self.commit_prefix = commit_prefix
        self.session = AiderSession(repo_path=self.repo_path)

        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

    def _run_git(self, *args: str) -> Tuple[bool, str]:
        """Run a git command and return (success, output)."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except subprocess.TimeoutExpired:
            return False, "Git command timed out"
        except Exception as e:
            return False, str(e)

    def git_status(self) -> str:
        """Get git status."""
        success, output = self._run_git("status", "--porcelain")
        return output if success else ""

    def git_diff(self, file_path: Optional[str] = None) -> str:
        """Get git diff."""
        args = ["diff"]
        if file_path:
            args.append(file_path)
        success, output = self._run_git(*args)
        return output if success else ""

    def git_commit(self, message: str) -> bool:
        """Stage and commit changes."""
        self._run_git("add", "-A")
        success, _ = self._run_git("commit", "-m", f"{self.commit_prefix} {message}")
        return success

    def add_file(self, file_path: str) -> bool:
        """Add file to editing context."""
        full_path = self.repo_path / file_path
        if full_path.exists():
            if file_path not in self.session.edited_files:
                self.session.edited_files.append(file_path)
            return True
        return False

    def add_files(self, file_paths: List[str]) -> List[str]:
        """Add multiple files. Returns list of successfully added files."""
        added = []
        for fp in file_paths:
            if self.add_file(fp):
                added.append(fp)
        return added

    def remove_file(self, file_path: str) -> bool:
        """Remove file from editing context."""
        if file_path in self.session.edited_files:
            self.session.edited_files.remove(file_path)
            return True
        return False

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a file in context."""
        full_path = self.repo_path / file_path
        if full_path.exists():
            try:
                return full_path.read_text(encoding='utf-8')
            except Exception:
                return None
        return None

    def parse_edit_blocks(self, response: str) -> List[EditBlock]:
        """Parse Aider-style edit blocks from LLM response."""
        blocks = []

        for match in self.EDIT_BLOCK_PATTERN.finditer(response):
            search, replace = match.groups()
            search = search.strip()
            replace = replace.strip()

            # Find which file contains this search text
            for file_path in self.session.edited_files:
                content = self.get_file_content(file_path)
                if content and search in content:
                    blocks.append(EditBlock(
                        file_path=file_path,
                        search=search,
                        replace=replace
                    ))
                    break

        return blocks

    def apply_edits(self, edits: List[EditBlock]) -> Dict[str, List[str]]:
        """Apply a list of edits. Returns success/failure by file."""
        results = {"applied": [], "failed": []}

        for edit in edits:
            success, message = edit.apply(self.repo_path)
            if success:
                results["applied"].append(edit.file_path)
                self.session.applied_edits.append(edit)
            else:
                results["failed"].append(f"{edit.file_path}: {message}")

        return results

    async def run(self, instruction: str) -> Dict[str, Any]:
        """Execute an editing instruction using AI."""
        # Gather file contents for context
        file_contents = {}
        for fp in self.session.edited_files:
            content = self.get_file_content(fp)
            if content:
                file_contents[fp] = content

        if not file_contents:
            return {
                "success": False,
                "error": "No files in editing context",
                "applied_edits": [],
                "failed_edits": []
            }

        # Build prompt
        files_section = "\n\n".join(
            f"=== {fp} ===\n```\n{content}\n```"
            for fp, content in file_contents.items()
        )

        prompt = f"""You are an expert code editor. Make the requested changes using SEARCH/REPLACE blocks.

For each change, use this exact format:
<<<<<<< SEARCH
exact code to find (copy from file exactly)
=======
replacement code
>>>>>>> REPLACE

Files in context:
{files_section}

Instruction: {instruction}

Provide SEARCH/REPLACE blocks for all necessary changes. The SEARCH section must match the file exactly."""

        try:
            from core.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider()
            response = await provider.complete(prompt)
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM error: {e}",
                "applied_edits": [],
                "failed_edits": []
            }

        # Parse and apply edits
        edits = self.parse_edit_blocks(response)
        results = self.apply_edits(edits)

        # Auto-commit if enabled
        if self.auto_commit and results["applied"]:
            commit_msg = instruction[:50] if len(instruction) <= 50 else instruction[:47] + "..."
            self.git_commit(commit_msg)

        # Record in history
        self.session.history.append({
            "instruction": instruction,
            "response": response,
            "applied": results["applied"],
            "failed": results["failed"]
        })

        return {
            "success": len(results["applied"]) > 0,
            "applied_edits": results["applied"],
            "failed_edits": results["failed"],
            "diff": self.git_diff(),
            "response": response
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get editing session history."""
        return self.session.history


AIDER_COMPAT_AVAILABLE = True
