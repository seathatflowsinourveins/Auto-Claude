"""
Chonkie RAG Chunking Adapter for Unleash Platform (V18)

Provides high-performance text chunking for RAG pipelines using Chonkie.

Key Features (verified 2026-01-31):
1. 33-51x faster than LangChain/LlamaIndex
2. SemanticChunker for topic-coherent retrieval
3. CodeChunker for AST-aware source processing
4. Pipeline API for fluent RAG workflows
5. Native Qdrant/Letta integration

V18 Advanced Features (NEW):
6. NeuralChunker - ML-based topic boundary detection using sentence-transformers
7. LateChunker - Document-level embeddings for semantic richness across chunks
8. SlumberChunker - LLM-driven intelligent agentic chunking via llama-cpp-python
9. Handshakes - Direct pipeline to 9 vector databases (Qdrant, Pinecone, Weaviate, etc.)
10. Recipes - Pre-configured pipeline patterns for common RAG scenarios

Repository: https://github.com/chonkie-inc/chonkie
Docs: https://docs.chonkie.ai/
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import structlog

# Check Chonkie availability
CHONKIE_AVAILABLE = False
CHONKIE_SEMANTIC_AVAILABLE = False

try:
    from chonkie import TokenChunker, WordChunker, SentenceChunker
    from chonkie import RecursiveChunker
    CHONKIE_AVAILABLE = True
except ImportError:
    pass

try:
    from chonkie import SemanticChunker, SDPMChunker
    CHONKIE_SEMANTIC_AVAILABLE = True
except ImportError:
    pass

# Optional: Code chunker
CHONKIE_CODE_AVAILABLE = False
try:
    from chonkie import CodeChunker
    CHONKIE_CODE_AVAILABLE = True
except ImportError:
    pass

# V18: Advanced chunkers
CHONKIE_NEURAL_AVAILABLE = False
try:
    from chonkie import NeuralChunker
    CHONKIE_NEURAL_AVAILABLE = True
except ImportError:
    pass

CHONKIE_LATE_AVAILABLE = False
try:
    from chonkie import LateChunker
    CHONKIE_LATE_AVAILABLE = True
except ImportError:
    pass

CHONKIE_SLUMBER_AVAILABLE = False
try:
    from chonkie import SlumberChunker
    CHONKIE_SLUMBER_AVAILABLE = True
except ImportError:
    pass

# V18: Handshakes for vector databases
CHONKIE_HANDSHAKES_AVAILABLE = False
AVAILABLE_HANDSHAKES: List[str] = []
try:
    from chonkie.handshakes import QdrantHandshake
    CHONKIE_HANDSHAKES_AVAILABLE = True
    AVAILABLE_HANDSHAKES.append("qdrant")
except ImportError:
    pass

try:
    from chonkie.handshakes import ChromaHandshake
    AVAILABLE_HANDSHAKES.append("chroma")
except ImportError:
    pass

try:
    from chonkie.handshakes import PineconeHandshake
    AVAILABLE_HANDSHAKES.append("pinecone")
except ImportError:
    pass

try:
    from chonkie.handshakes import WeaviateHandshake
    AVAILABLE_HANDSHAKES.append("weaviate")
except ImportError:
    pass

try:
    from chonkie.handshakes import MilvusHandshake
    AVAILABLE_HANDSHAKES.append("milvus")
except ImportError:
    pass

try:
    from chonkie.handshakes import LanceDBHandshake
    AVAILABLE_HANDSHAKES.append("lancedb")
except ImportError:
    pass

# Optional: Pipeline API
CHONKIE_PIPELINE_AVAILABLE = False
try:
    from chonkie import Pipeline
    CHONKIE_PIPELINE_AVAILABLE = True
except ImportError:
    pass

# Optional: Refineries
CHONKIE_REFINERY_AVAILABLE = False
try:
    from chonkie.refinery import OverlapRefinery, EmbeddingsRefinery
    CHONKIE_REFINERY_AVAILABLE = True
except ImportError:
    pass

# Register adapter status
try:
    from . import register_adapter
    register_adapter("chonkie", CHONKIE_AVAILABLE, "1.5.4")
except ImportError:
    pass

logger = structlog.get_logger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    TOKEN = "token"           # Fixed token count
    WORD = "word"             # Word boundaries
    SENTENCE = "sentence"     # Sentence boundaries
    RECURSIVE = "recursive"   # Hierarchical splitting
    SEMANTIC = "semantic"     # Embedding similarity
    SDPM = "sdpm"             # Semantic Double-Pass Merge
    CODE = "code"             # AST-aware code chunking
    # V18 Advanced Chunkers
    NEURAL = "neural"         # ML-based topic boundary detection
    LATE = "late"             # Document-level embeddings
    SLUMBER = "slumber"       # LLM-driven agentic chunking


@dataclass
class ChunkResult:
    """Result from chunking operation"""
    text: str
    token_count: int
    start_index: int = 0
    end_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class ChonkieConfig:
    """Configuration for Chonkie adapter"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    embedding_model: str = "minishlab/potion-base-32M"
    similarity_threshold: float = 0.7
    min_sentences_per_chunk: int = 1
    similarity_window: int = 3
    # V18 Advanced options
    neural_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    late_embedding_model: str = "jinaai/jina-embeddings-v3"
    slumber_llm_model: str = "llama-cpp"  # or path to GGUF
    slumber_batch_size: int = 10


class ChonkieAdapter:
    """
    Unified Chonkie adapter for RAG chunking operations.

    Provides multiple chunking strategies optimized for different use cases:
    - TokenChunker: Fixed-size for LLM token limits
    - SemanticChunker: Topic-coherent for retrieval
    - CodeChunker: AST-aware for source code
    - RecursiveChunker: Hierarchical for structured docs

    Example:
        adapter = ChonkieAdapter(config=ChonkieConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=512,
        ))

        chunks = adapter.chunk("Your document text...")
        refined = adapter.add_overlap(chunks, context_size=50)

        # With pipeline
        adapter.pipeline_chunk(
            source_dir="./docs",
            vector_store="qdrant",
            collection_name="knowledge"
        )
    """

    def __init__(self, config: Optional[ChonkieConfig] = None):
        self.config = config or ChonkieConfig()
        self._chunkers: Dict[ChunkingStrategy, Any] = {}
        self._initialized = False

        if CHONKIE_AVAILABLE:
            self._init_chunkers()

    def _init_chunkers(self) -> None:
        """Initialize chunker instances based on availability"""
        try:
            # Always available with base install
            from chonkie import TokenChunker, RecursiveChunker

            self._chunkers[ChunkingStrategy.TOKEN] = TokenChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            self._chunkers[ChunkingStrategy.RECURSIVE] = RecursiveChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            # Semantic chunkers (require chonkie[semantic])
            if CHONKIE_SEMANTIC_AVAILABLE:
                from chonkie import SemanticChunker

                self._chunkers[ChunkingStrategy.SEMANTIC] = SemanticChunker(
                    embedding_model=self.config.embedding_model,
                    threshold=self.config.similarity_threshold,
                    chunk_size=self.config.chunk_size,
                    min_sentences_per_chunk=self.config.min_sentences_per_chunk,
                    similarity_window=self.config.similarity_window,
                )

            # Code chunker (for source files)
            if CHONKIE_CODE_AVAILABLE:
                from chonkie import CodeChunker

                self._chunkers[ChunkingStrategy.CODE] = CodeChunker(
                    chunk_size=self.config.chunk_size,
                )

            # V18: NeuralChunker - ML-based topic boundaries
            if CHONKIE_NEURAL_AVAILABLE:
                from chonkie import NeuralChunker

                self._chunkers[ChunkingStrategy.NEURAL] = NeuralChunker(
                    model=self.config.neural_model,
                    chunk_size=self.config.chunk_size,
                )

            # V18: LateChunker - Document-level embeddings
            if CHONKIE_LATE_AVAILABLE:
                from chonkie import LateChunker

                self._chunkers[ChunkingStrategy.LATE] = LateChunker(
                    embedding_model=self.config.late_embedding_model,
                    chunk_size=self.config.chunk_size,
                )

            # V18: SlumberChunker - LLM-driven agentic chunking
            if CHONKIE_SLUMBER_AVAILABLE:
                from chonkie import SlumberChunker

                self._chunkers[ChunkingStrategy.SLUMBER] = SlumberChunker(
                    model=self.config.slumber_llm_model,
                    batch_size=self.config.slumber_batch_size,
                )

            self._initialized = True
            logger.info(
                "chonkie_initialized",
                strategies=list(self._chunkers.keys()),
                semantic_available=CHONKIE_SEMANTIC_AVAILABLE,
                code_available=CHONKIE_CODE_AVAILABLE,
            )

        except Exception as e:
            logger.error("chonkie_init_failed", error=str(e))

    def chunk(
        self,
        text: str,
        strategy: Optional[ChunkingStrategy] = None,
    ) -> List[ChunkResult]:
        """
        Chunk text using specified strategy.

        Args:
            text: Text to chunk
            strategy: Chunking strategy (defaults to config strategy)

        Returns:
            List of ChunkResult objects
        """
        if not CHONKIE_AVAILABLE or not self._initialized:
            # Fallback: simple character-based chunking
            return self._fallback_chunk(text)

        strategy = strategy or self.config.strategy
        chunker = self._chunkers.get(strategy)

        if chunker is None:
            logger.warning(
                "chunker_not_available",
                strategy=strategy.value,
                falling_back_to="token",
            )
            chunker = self._chunkers.get(ChunkingStrategy.TOKEN)

        if chunker is None:
            return self._fallback_chunk(text)

        try:
            raw_chunks = chunker(text)

            results = []
            for chunk in raw_chunks:
                results.append(ChunkResult(
                    text=chunk.text,
                    token_count=chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text.split()),
                    start_index=chunk.start_index if hasattr(chunk, 'start_index') else 0,
                    end_index=chunk.end_index if hasattr(chunk, 'end_index') else len(chunk.text),
                    metadata={"strategy": strategy.value},
                ))

            logger.info(
                "chunking_complete",
                strategy=strategy.value,
                input_chars=len(text),
                num_chunks=len(results),
            )

            return results

        except Exception as e:
            logger.error("chunking_failed", strategy=strategy.value, error=str(e))
            return self._fallback_chunk(text)

    def _fallback_chunk(self, text: str) -> List[ChunkResult]:
        """Simple fallback chunking when Chonkie not available"""
        chunk_size = self.config.chunk_size * 4  # Approximate chars per chunk
        overlap = self.config.chunk_overlap * 4

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunks.append(ChunkResult(
                text=chunk_text,
                token_count=len(chunk_text.split()),
                start_index=start,
                end_index=end,
                metadata={"strategy": "fallback"},
            ))

            start = end - overlap if end < len(text) else len(text)

        return chunks

    def add_overlap(
        self,
        chunks: List[ChunkResult],
        context_size: int = 50,
        method: str = "suffix",
    ) -> List[ChunkResult]:
        """
        Add overlap context to chunks using OverlapRefinery.

        Args:
            chunks: List of chunks to refine
            context_size: Size of overlap context
            method: "suffix" or "prefix"

        Returns:
            Refined chunks with overlap
        """
        if not CHONKIE_REFINERY_AVAILABLE:
            logger.warning("overlap_refinery_not_available")
            return chunks

        try:
            from chonkie.refinery import OverlapRefinery

            refinery = OverlapRefinery(
                context_size=context_size,
                method=method,
                merge=True,
            )

            # Convert to Chonkie format for refinery
            # (This is a simplified implementation)
            refined = refinery(chunks)

            return refined

        except Exception as e:
            logger.error("overlap_refinery_failed", error=str(e))
            return chunks

    def add_embeddings(
        self,
        chunks: List[ChunkResult],
        model: Optional[str] = None,
    ) -> List[ChunkResult]:
        """
        Add embeddings to chunks using EmbeddingsRefinery.

        Args:
            chunks: List of chunks to embed
            model: Embedding model (defaults to config model)

        Returns:
            Chunks with embeddings attached
        """
        if not CHONKIE_REFINERY_AVAILABLE:
            logger.warning("embeddings_refinery_not_available")
            return chunks

        try:
            from chonkie.refinery import EmbeddingsRefinery

            model = model or self.config.embedding_model
            refinery = EmbeddingsRefinery(embedding_model=model)

            refined = refinery(chunks)

            # Update ChunkResult with embeddings
            for i, chunk in enumerate(refined):
                if hasattr(chunk, 'embedding'):
                    chunks[i].embedding = chunk.embedding

            return chunks

        except Exception as e:
            logger.error("embeddings_refinery_failed", error=str(e))
            return chunks

    def pipeline_chunk(
        self,
        source_dir: str,
        file_extensions: Optional[List[str]] = None,
        vector_store: str = "qdrant",
        collection_name: str = "knowledge",
        store_url: Optional[str] = None,
    ) -> int:
        """
        Run full RAG pipeline: fetch → chunk → embed → store.

        Args:
            source_dir: Directory containing source files
            file_extensions: File extensions to process (default: .md, .txt)
            vector_store: Vector store type ("qdrant", "chroma", "pinecone")
            collection_name: Collection name in vector store
            store_url: URL for vector store

        Returns:
            Number of documents processed
        """
        if not CHONKIE_PIPELINE_AVAILABLE:
            logger.error("chonkie_pipeline_not_available")
            return 0

        try:
            from chonkie import Pipeline

            extensions = file_extensions or [".md", ".txt"]

            pipeline = (
                Pipeline()
                .fetch_from("file", dir=source_dir, ext=extensions)
                .process_with("text")
                .chunk_with(
                    self.config.strategy.value,
                    chunk_size=self.config.chunk_size,
                    threshold=self.config.similarity_threshold,
                )
                .refine_with("overlap", context_size=self.config.chunk_overlap)
                .refine_with("embedding", model=self.config.embedding_model)
            )

            # Add vector store
            if vector_store == "qdrant":
                pipeline = pipeline.store_in(
                    "qdrant",
                    collection_name=collection_name,
                    url=store_url or "http://localhost:6333",
                )
            elif vector_store == "chroma":
                pipeline = pipeline.store_in(
                    "chroma",
                    collection_name=collection_name,
                )

            docs = pipeline.run()

            logger.info(
                "pipeline_complete",
                source_dir=source_dir,
                num_docs=len(docs) if docs else 0,
                vector_store=vector_store,
            )

            return len(docs) if docs else 0

        except Exception as e:
            logger.error("pipeline_failed", error=str(e))
            return 0

    def chunk_code(
        self,
        code: str,
        language: str = "python",
    ) -> List[ChunkResult]:
        """
        Chunk source code with AST-awareness.

        Args:
            code: Source code text
            language: Programming language

        Returns:
            List of code chunks respecting structure
        """
        if not CHONKIE_CODE_AVAILABLE:
            logger.warning("code_chunker_not_available", falling_back_to="recursive")
            return self.chunk(code, strategy=ChunkingStrategy.RECURSIVE)

        return self.chunk(code, strategy=ChunkingStrategy.CODE)

    # ═══════════════════════════════════════════════════════════════════
    # V18: Advanced Chunking Methods
    # ═══════════════════════════════════════════════════════════════════

    def chunk_neural(self, text: str) -> List[ChunkResult]:
        """
        Chunk text using ML-based topic boundary detection (V18).

        Uses sentence-transformers to identify natural topic transitions.
        Best for: diverse document types, news articles, multi-topic content.
        """
        if not CHONKIE_NEURAL_AVAILABLE:
            logger.warning("neural_chunker_not_available", falling_back_to="semantic")
            return self.chunk(text, strategy=ChunkingStrategy.SEMANTIC)
        return self.chunk(text, strategy=ChunkingStrategy.NEURAL)

    def chunk_late(self, text: str) -> List[ChunkResult]:
        """
        Chunk text using document-level embeddings (V18).

        Embeds the full document first, then chunks while preserving
        semantic richness across chunk boundaries.
        Best for: long documents, academic papers, technical documentation.
        """
        if not CHONKIE_LATE_AVAILABLE:
            logger.warning("late_chunker_not_available", falling_back_to="semantic")
            return self.chunk(text, strategy=ChunkingStrategy.SEMANTIC)
        return self.chunk(text, strategy=ChunkingStrategy.LATE)

    def chunk_agentic(self, text: str) -> List[ChunkResult]:
        """
        Chunk text using LLM-driven intelligent chunking (V18).

        Uses an LLM (via llama-cpp-python) to make intelligent chunking
        decisions. Slower but highest quality for complex documents.
        Best for: complex legal docs, contracts, nuanced content.
        """
        if not CHONKIE_SLUMBER_AVAILABLE:
            logger.warning("slumber_chunker_not_available", falling_back_to="semantic")
            return self.chunk(text, strategy=ChunkingStrategy.SEMANTIC)
        return self.chunk(text, strategy=ChunkingStrategy.SLUMBER)

    # ═══════════════════════════════════════════════════════════════════
    # V18: Handshakes - Direct Vector Database Integration
    # ═══════════════════════════════════════════════════════════════════

    def handshake(
        self,
        chunks: List[ChunkResult],
        vector_db: str,
        collection_name: str,
        client: Any = None,
        **kwargs: Any,
    ) -> int:
        """
        Directly pipeline chunks to a vector database using Handshakes (V18).

        Handshakes provide direct integration with 9 vector databases:
        - Qdrant, Pinecone, Weaviate, Milvus, ChromaDB
        - LanceDB, Turbopuffer, Marqo, ApertureDB

        Args:
            chunks: List of chunks to store
            vector_db: Vector database type ("qdrant", "pinecone", etc.)
            collection_name: Collection/index name
            client: Optional pre-configured client
            **kwargs: Additional handshake-specific options

        Returns:
            Number of chunks stored

        Example:
            chunks = adapter.chunk(text)
            stored = adapter.handshake(
                chunks,
                vector_db="qdrant",
                collection_name="knowledge",
                url="http://localhost:6333"
            )
        """
        if not CHONKIE_HANDSHAKES_AVAILABLE:
            logger.error(
                "handshakes_not_available",
                available=AVAILABLE_HANDSHAKES,
                requested=vector_db,
            )
            return 0

        if vector_db not in AVAILABLE_HANDSHAKES:
            logger.error(
                "handshake_not_supported",
                requested=vector_db,
                available=AVAILABLE_HANDSHAKES,
            )
            return 0

        try:
            handshake_class = self._get_handshake_class(vector_db)
            if handshake_class is None:
                return 0

            # Create handshake instance
            handshake = handshake_class(
                collection_name=collection_name,
                client=client,
                **kwargs,
            )

            # Convert ChunkResult to Chonkie Chunk format if needed
            chunk_data = [{"text": c.text, "embedding": c.embedding} for c in chunks]

            # Execute handshake
            result = handshake(chunk_data)

            logger.info(
                "handshake_complete",
                vector_db=vector_db,
                collection=collection_name,
                chunks_stored=len(chunks),
            )

            return len(chunks)

        except Exception as e:
            logger.error(
                "handshake_failed",
                vector_db=vector_db,
                error=str(e),
            )
            return 0

    def _get_handshake_class(self, vector_db: str) -> Optional[Any]:
        """Get the appropriate handshake class for a vector database"""
        try:
            if vector_db == "qdrant":
                from chonkie.handshakes import QdrantHandshake
                return QdrantHandshake
            elif vector_db == "chroma":
                from chonkie.handshakes import ChromaHandshake
                return ChromaHandshake
            elif vector_db == "pinecone":
                from chonkie.handshakes import PineconeHandshake
                return PineconeHandshake
            elif vector_db == "weaviate":
                from chonkie.handshakes import WeaviateHandshake
                return WeaviateHandshake
            elif vector_db == "milvus":
                from chonkie.handshakes import MilvusHandshake
                return MilvusHandshake
            elif vector_db == "lancedb":
                from chonkie.handshakes import LanceDBHandshake
                return LanceDBHandshake
            return None
        except ImportError:
            return None

    def get_available_handshakes(self) -> List[str]:
        """Get list of available handshake integrations"""
        return AVAILABLE_HANDSHAKES.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status"""
        return {
            "chonkie_available": CHONKIE_AVAILABLE,
            "semantic_available": CHONKIE_SEMANTIC_AVAILABLE,
            "code_available": CHONKIE_CODE_AVAILABLE,
            "pipeline_available": CHONKIE_PIPELINE_AVAILABLE,
            "refinery_available": CHONKIE_REFINERY_AVAILABLE,
            # V18 capabilities
            "neural_available": CHONKIE_NEURAL_AVAILABLE,
            "late_available": CHONKIE_LATE_AVAILABLE,
            "slumber_available": CHONKIE_SLUMBER_AVAILABLE,
            "handshakes_available": CHONKIE_HANDSHAKES_AVAILABLE,
            "available_handshakes": AVAILABLE_HANDSHAKES,
            "initialized": self._initialized,
            "strategies": [s.value for s in self._chunkers.keys()],
            "config": {
                "chunk_size": self.config.chunk_size,
                "strategy": self.config.strategy.value,
                "embedding_model": self.config.embedding_model,
                "neural_model": self.config.neural_model,
                "late_embedding_model": self.config.late_embedding_model,
            },
            "version": "V18",
        }


# Global adapter instance
_global_adapter: Optional[ChonkieAdapter] = None


def get_chonkie_adapter() -> ChonkieAdapter:
    """Get or create global Chonkie adapter instance"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = ChonkieAdapter()
    return _global_adapter


def configure_chonkie(config: ChonkieConfig) -> ChonkieAdapter:
    """Configure and return global Chonkie adapter"""
    global _global_adapter
    _global_adapter = ChonkieAdapter(config)
    return _global_adapter


# Convenience functions
def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
) -> List[ChunkResult]:
    """Quick chunk text using default adapter"""
    return get_chonkie_adapter().chunk(text, strategy)


def chunk_code(code: str, language: str = "python") -> List[ChunkResult]:
    """Quick chunk source code"""
    return get_chonkie_adapter().chunk_code(code, language)


__all__ = [
    "ChonkieAdapter",
    "ChonkieConfig",
    "ChunkingStrategy",
    "ChunkResult",
    "get_chonkie_adapter",
    "configure_chonkie",
    "chunk_text",
    "chunk_code",
    # Availability flags
    "CHONKIE_AVAILABLE",
    "CHONKIE_SEMANTIC_AVAILABLE",
    "CHONKIE_CODE_AVAILABLE",
    "CHONKIE_PIPELINE_AVAILABLE",
    # V18 flags
    "CHONKIE_NEURAL_AVAILABLE",
    "CHONKIE_LATE_AVAILABLE",
    "CHONKIE_SLUMBER_AVAILABLE",
    "CHONKIE_HANDSHAKES_AVAILABLE",
    "AVAILABLE_HANDSHAKES",
]
