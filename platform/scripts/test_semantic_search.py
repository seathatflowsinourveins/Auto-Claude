#!/usr/bin/env python3
"""Test semantic search with real embeddings."""

import os
import sys
from pathlib import Path

# Set the API key directly
os.environ["VOYAGE_API_KEY"] = "pa-KCpYL_zzmvoPK1dM6tN5kdCD8e6qnAndC-dSTlCuzK4"

# Import search_code directly from the module file
PROJECT_ROOT = Path("Z:/insider/AUTO CLAUDE/unleash")
PIPELINE_PATH = PROJECT_ROOT / "platform" / "core" / "embedding_pipeline.py"

# Use exec to load the module without import conflicts
import importlib.util
spec = importlib.util.spec_from_file_location("embedding_pipeline", PIPELINE_PATH)
embedding_pipeline = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(embedding_pipeline)  # type: ignore
search_code = embedding_pipeline.search_code

def main():
    queries = [
        "how to search code semantically",
        "embedding pipeline configuration",
        "Qdrant vector database",
        "code chunking and tokenization",
    ]

    print("=" * 60)
    print("SEMANTIC SEARCH VERIFICATION")
    print("=" * 60)

    for query in queries:
        print(f"\n[QUERY] {query}")
        print("-" * 50)

        try:
            results = search_code(query, limit=3)

            if not results:
                print("  [WARN] No results found")
                continue

            for i, r in enumerate(results, 1):
                score = r.get("score", 0)
                file_path = r.get("file_path", "unknown")
                start_line = r.get("start_line", 0)
                content = r.get("content", "")[:80].replace("\n", " ")

                print(f"  {i}. {file_path}:{start_line} (score: {score:.3f})")
                print(f"     {content}...")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
