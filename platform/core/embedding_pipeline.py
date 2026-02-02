#!/usr/bin/env python3
"""
UNLEASH Embedding Pipeline
==========================
Part of the Unified Code Intelligence Architecture (5-Layer, 2026)

Embeds code files using Voyage-code-3 and stores in Qdrant for semantic search.

Usage:
    python -m platform.core.embedding_pipeline [--full] [--dry-run] [--batch-size N]

Requirements:
    pip install voyageai qdrant-client structlog
    export VOYAGE_API_KEY=your_key
"""

import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ]
)
log = structlog.get_logger()

# Constants
import os as _embedding_os
UNLEASH_ROOT = Path("Z:/insider/AUTO CLAUDE/unleash")
# V45 FIX: Environment-configurable Qdrant URL (was hardcoded)
QDRANT_URL = _embedding_os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = _embedding_os.environ.get("QDRANT_COLLECTION", "unleash_code")
VOYAGE_MODEL = "voyage-code-3"
VOYAGE_DIMENSION = 1024  # Default for voyage-code-3
BATCH_SIZE = 32  # Voyage API batch limit
MAX_TOKENS_PER_CHUNK = 512

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".toml": "toml",
}

EXCLUDE_PATTERNS = [
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
    "archived",
    "archive",
    ".narsil-cache",
    # SDK directories (third-party code)
    "sdks/",
    "everything-claude-code-full",
    "opik-full",
    # Large generated/downloaded directories
    ".cache",
    ".mypy_cache",
    ".ruff_cache",
    "site-packages",
    # Specific large subdirectories
    "research/papers",
    "docs/archive",
]

# Scope-limited directories for initial indexing
CORE_DIRS = [
    "platform/core",
    "platform/tests",
    "platform/scripts",
    "agents",
    "skills",
]


@dataclass
class CodeChunk:
    """Represents a chunk of code for embedding."""
    file_path: Path
    chunk_index: int
    content: str
    language: str
    start_line: int = 0
    end_line: int = 0

    @property
    def id(self) -> str:
        """Generate unique UUID for this chunk (deterministic from content)."""
        # Create a UUID5 from a namespace and the unique identifier
        # This is deterministic: same input = same UUID
        namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace
        return str(uuid.uuid5(namespace, f"{self.file_path}:{self.chunk_index}"))


@dataclass
class PipelineStats:
    """Track embedding pipeline statistics."""
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_embedded: int = 0
    total_tokens: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def rate(self) -> float:
        if self.elapsed_seconds > 0:
            return self.chunks_embedded / self.elapsed_seconds
        return 0.0


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded from indexing."""
    path_str = str(path).replace("\\", "/")
    return any(pattern in path_str for pattern in EXCLUDE_PATTERNS)


def iter_code_files(core_only: bool = False) -> Iterator[Path]:
    """Iterate over all code files in UNLEASH.

    Args:
        core_only: If True, only scan CORE_DIRS instead of entire codebase.
    """
    if core_only:
        # Scan only the core directories for faster initial indexing
        for core_dir in CORE_DIRS:
            core_path = UNLEASH_ROOT / core_dir
            if core_path.exists():
                for ext in SUPPORTED_EXTENSIONS:
                    for file_path in core_path.rglob(f"*{ext}"):
                        if not should_exclude(file_path):
                            yield file_path
    else:
        # Full codebase scan
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in UNLEASH_ROOT.rglob(f"*{ext}"):
                if not should_exclude(file_path):
                    yield file_path


def chunk_code(content: str, file_path: Path, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> list[CodeChunk]:
    """
    Chunk code by lines, trying to keep functions/classes together.

    This is a simple chunker - for production, integrate tree-sitter
    for AST-aware chunking.
    """
    language = SUPPORTED_EXTENSIONS.get(file_path.suffix, "text")
    lines = content.split("\n")
    chunks: list[CodeChunk] = []
    current_lines: list[str] = []
    current_tokens = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        # Rough token estimate (words + punctuation)
        line_tokens = len(line.split()) + line.count("(") + line.count(")")

        # Start new chunk if exceeds limit and we have content
        if current_tokens + line_tokens > max_tokens and current_lines:
            chunk_content = "\n".join(current_lines)
            if chunk_content.strip():  # Don't create empty chunks
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content=chunk_content,
                    language=language,
                    start_line=start_line,
                    end_line=i - 1,
                ))
            current_lines = []
            current_tokens = 0
            start_line = i

        current_lines.append(line)
        current_tokens += line_tokens

    # Don't forget the last chunk
    if current_lines:
        chunk_content = "\n".join(current_lines)
        if chunk_content.strip():
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content=chunk_content,
                language=language,
                start_line=start_line,
                end_line=len(lines),
            ))

    return chunks


class EmbeddingPipeline:
    """Manages code embedding and storage in Qdrant."""

    def __init__(self, dry_run: bool = False, batch_size: int = BATCH_SIZE):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.stats = PipelineStats()
        self._voyage_client = None
        self._qdrant_client = None

    @property
    def voyage(self):
        """Lazy-load Voyage client."""
        if self._voyage_client is None:
            import voyageai  # type: ignore[import-untyped]
            api_key = os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY environment variable not set")
            self._voyage_client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        return self._voyage_client

    @property
    def qdrant(self):
        """Lazy-load Qdrant client."""
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(QDRANT_URL)
        return self._qdrant_client

    def verify_qdrant_collection(self) -> bool:
        """Verify Qdrant collection exists with correct dimensions."""
        try:
            info = self.qdrant.get_collection(QDRANT_COLLECTION)
            # Handle both dict and object access patterns for vectors config
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict):
                actual_dim = vectors_config.get("size", 0)
            else:
                actual_dim = getattr(vectors_config, "size", 0)

            if actual_dim != VOYAGE_DIMENSION:
                log.error(
                    "dimension_mismatch",
                    expected=VOYAGE_DIMENSION,
                    actual=actual_dim,
                    collection=QDRANT_COLLECTION,
                )
                return False
            log.info("qdrant_verified", collection=QDRANT_COLLECTION, dimension=actual_dim)
            return True
        except Exception as e:
            log.error("qdrant_error", error=str(e))
            return False

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Voyage-code-3."""
        if self.dry_run:
            # Return fake embeddings for dry run
            return [[0.0] * VOYAGE_DIMENSION for _ in texts]

        result = self.voyage.embed(
            texts,
            model=VOYAGE_MODEL,
            input_type="document",
        )
        return result.embeddings

    def store_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> None:
        """Store embedded chunks in Qdrant."""
        if self.dry_run:
            log.info("dry_run_store", chunks=len(chunks))
            return

        from qdrant_client.models import PointStruct

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(PointStruct(
                id=chunk.id,
                vector=embedding,
                payload={
                    "file_path": str(chunk.file_path.relative_to(UNLEASH_ROOT)),
                    "language": chunk.language,
                    "chunk_index": chunk.chunk_index,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content[:2000],  # Truncate for storage
                    "indexed_at": datetime.now().isoformat(),
                },
            ))

        self.qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
        )

    def process_file(self, file_path: Path) -> int:
        """Process a single file: chunk, embed, store. Returns chunk count."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                self.stats.files_skipped += 1
                return 0

            chunks = chunk_code(content, file_path)
            if not chunks:
                self.stats.files_skipped += 1
                return 0

            # Process in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                texts = [c.content for c in batch]

                embeddings = self.embed_batch(texts)
                self.store_chunks(batch, embeddings)

                self.stats.chunks_embedded += len(batch)
                self.stats.total_tokens += sum(len(t.split()) for t in texts)

            self.stats.files_processed += 1
            return len(chunks)

        except Exception as e:
            log.error("file_failed", file=str(file_path), error=str(e))
            self.stats.files_failed += 1
            return 0

    def run(self, full: bool = False, core_only: bool = False) -> PipelineStats:
        """Run the full embedding pipeline.

        Args:
            full: If True, clear existing vectors before indexing.
            core_only: If True, only index CORE_DIRS for faster initial setup.
        """
        log.info("pipeline_starting", full=full, dry_run=self.dry_run, core_only=core_only)

        # Verify Qdrant collection
        if not self.dry_run and not self.verify_qdrant_collection():
            raise RuntimeError("Qdrant collection verification failed")

        # Clear collection if full rebuild
        if full and not self.dry_run:
            log.info("clearing_collection", collection=QDRANT_COLLECTION)
            from qdrant_client.models import FilterSelector, Filter
            # Delete all points (recreate would lose config)
            self.qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=FilterSelector(filter=Filter()),
            )

        # Process all files
        files = list(iter_code_files(core_only=core_only))
        log.info("files_found", count=len(files))

        for i, file_path in enumerate(files):
            self.process_file(file_path)

            # Progress logging every 50 files
            if (i + 1) % 50 == 0:
                log.info(
                    "progress",
                    files=f"{i+1}/{len(files)}",
                    chunks=self.stats.chunks_embedded,
                    rate=f"{self.stats.rate:.1f}/s",
                )

        log.info(
            "pipeline_complete",
            files_processed=self.stats.files_processed,
            files_skipped=self.stats.files_skipped,
            files_failed=self.stats.files_failed,
            chunks_embedded=self.stats.chunks_embedded,
            elapsed=f"{self.stats.elapsed_seconds:.1f}s",
        )

        return self.stats


def search_code(query: str, limit: int = 5) -> list[dict]:
    """
    Search for code using semantic similarity.

    This is the main interface for querying embedded code.
    """
    import voyageai  # type: ignore[import-untyped]
    from qdrant_client import QdrantClient

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not set")

    voyage = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
    qdrant = QdrantClient(QDRANT_URL)

    # Embed query
    result = voyage.embed([query], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    # Search Qdrant using query method (works with latest client)
    results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=limit,
    )

    return [
        {
            "file_path": r.payload.get("file_path") if r.payload else None,
            "content": r.payload.get("content") if r.payload else None,
            "language": r.payload.get("language") if r.payload else None,
            "start_line": r.payload.get("start_line") if r.payload else None,
            "score": r.score,
        }
        for r in results.points
    ]


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="UNLEASH Embedding Pipeline")
    parser.add_argument("--full", action="store_true", help="Full re-index (clears collection)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually embed or store")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for embedding")
    parser.add_argument("--core-only", action="store_true", help="Only index core platform directories (faster)")
    parser.add_argument("--search", type=str, help="Search query (test mode)")

    args = parser.parse_args()

    if args.search:
        # Search mode
        results = search_code(args.search)
        print(f"\nSearch results for: {args.search}\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['file_path']}:{r['start_line']} (score: {r['score']:.3f})")
            print(f"   {r['content'][:100]}...")
            print()
        return

    # Embedding mode
    pipeline = EmbeddingPipeline(dry_run=args.dry_run, batch_size=args.batch_size)

    try:
        stats = pipeline.run(full=args.full, core_only=args.core_only)

        # Print summary
        print("\n" + "=" * 60)
        print("EMBEDDING PIPELINE SUMMARY")
        print("=" * 60)
        print(f"  Files processed: {stats.files_processed}")
        print(f"  Files skipped:   {stats.files_skipped}")
        print(f"  Files failed:    {stats.files_failed}")
        print(f"  Chunks embedded: {stats.chunks_embedded}")
        print(f"  Total time:      {stats.elapsed_seconds:.1f}s")
        print(f"  Rate:            {stats.rate:.1f} chunks/s")
        print("=" * 60)

        # Exit with error if any files failed
        if stats.files_failed > 0:
            sys.exit(1)

    except Exception as e:
        log.error("pipeline_error", error=str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
