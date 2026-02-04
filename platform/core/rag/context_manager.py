"""Context Window Manager - Token budget, selection, compression, formatting for RAG."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# === TOKEN COUNTING ===

class TokenCounter:
    """Singleton token counter with tiktoken support and caching."""
    _instance: Optional["TokenCounter"] = None
    _encoder = None
    _available = False

    def __new__(cls) -> "TokenCounter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._try_load()
        return cls._instance

    def _try_load(self) -> None:
        try:
            import tiktoken
            TokenCounter._encoder = tiktoken.get_encoding("cl100k_base")
            TokenCounter._available = True
        except (ImportError, Exception):
            pass

    @property
    def is_accurate(self) -> bool:
        return TokenCounter._available

    @lru_cache(maxsize=10000)
    def count(self, text: str) -> int:
        if not text:
            return 0
        if TokenCounter._available and TokenCounter._encoder:
            return len(TokenCounter._encoder.encode(text))
        # Fallback: ~4 chars/token with adjustments
        base = len(text) / 4
        ws_adj = 1 + (len(re.findall(r'\s', text)) / max(len(text), 1) * 0.3)
        code_adj = 1 + (len(re.findall(r'[{}()\[\];=<>]', text)) / max(len(text), 1) * 0.2)
        return int(base * ws_adj * code_adj)

    def clear_cache(self) -> None:
        self.count.cache_clear()


_counter = TokenCounter()
def count_tokens(text: str) -> int:
    return _counter.count(text)


# === ENUMS AND CONFIG ===

class FormatStyle(str, Enum):
    MARKDOWN = "markdown"
    XML = "xml"
    NUMBERED = "numbered"
    PLAIN = "plain"

class ContentType(str, Enum):
    TEXT = "text"
    CODE = "code"
    DATA = "data"


@dataclass
class ContextConfig:
    max_context_tokens: int = 8000
    query_reserve_tokens: int = 200
    generation_reserve_tokens: int = 500
    relevance_weight: float = 0.5
    recency_weight: float = 0.2
    importance_weight: float = 0.3
    enable_compression: bool = True
    compression_threshold: int = 1000
    target_compression_ratio: float = 0.5
    default_format: FormatStyle = FormatStyle.MARKDOWN
    include_source_attribution: bool = True
    max_source_label_length: int = 50


@dataclass
class ContextItem:
    content: str
    relevance: float = 0.5
    importance: float = 0.5
    source: str = "unknown"
    timestamp: Optional[datetime] = None
    content_type: ContentType = ContentType.TEXT
    must_include: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    compressed_content: Optional[str] = None
    selection_score: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)

    @property
    def effective_content(self) -> str:
        return self.compressed_content or self.content

    @property
    def effective_tokens(self) -> int:
        return count_tokens(self.compressed_content) if self.compressed_content else self.token_count


@dataclass
class FormattedContext:
    content: str
    total_tokens: int
    items_included: int
    items_excluded: int
    tokens_available: int
    tokens_used: int
    format_style: FormatStyle
    sources: List[str]
    compression_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.value if isinstance(v, Enum) else v for k, v in self.__dict__.items()}


# === COMPRESSION ===

class ContextCompressor:
    """Compress long contexts while preserving key information."""

    def compress(self, content: str, target_tokens: int, content_type: ContentType = ContentType.TEXT) -> str:
        if count_tokens(content) <= target_tokens:
            return content
        return self._compress_code(content, target_tokens) if content_type == ContentType.CODE else self._compress_text(content, target_tokens)

    def _compress_text(self, content: str, target_tokens: int) -> str:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip() and len(s) > 10]
        if not sentences:
            return content[:target_tokens * 4]

        scored = [(s, self._score(s, i, len(sentences))) for i, s in enumerate(sentences)]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected, tokens = [], 0
        for s, score in scored:
            st = count_tokens(s)
            if tokens + st <= target_tokens:
                selected.append((s, scored.index((s, score))))
                tokens += st

        selected.sort(key=lambda x: x[1])
        result = " ".join(s for s, _ in selected)
        return result.rstrip(".") + "..." if len(selected) < len(sentences) else result

    def _compress_code(self, content: str, target_tokens: int) -> str:
        lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith(('#', '//'))]
        compressed = '\n'.join(lines)
        if count_tokens(compressed) > target_tokens:
            half = target_tokens * 2
            compressed = f"{compressed[:half]}\n... [truncated] ...\n{compressed[-half:]}"
        return compressed

    def _score(self, sentence: str, pos: int, total: int) -> float:
        score = 0.5 + (0.3 if pos == 0 else 0.2 if pos == total - 1 else 0)
        words = len(sentence.split())
        score += 0.2 if 10 <= words <= 25 else 0
        if any(kw in sentence.lower() for kw in ['important', 'key', 'must', 'critical', 'note']):
            score += 0.15
        return min(1.0, score)


# === SELECTION ===

class ContextSelector:
    """Select optimal context subset within token budget."""

    def __init__(self, config: ContextConfig):
        self.config = config
        self.compressor = ContextCompressor()

    def select(self, items: List[ContextItem], available_tokens: int, query: Optional[str] = None) -> List[ContextItem]:
        if not items:
            return []

        self._score_items(items)
        must = [i for i in items if i.must_include]
        optional = sorted([i for i in items if not i.must_include], key=lambda x: x.selection_score, reverse=True)

        selected, remaining = [], available_tokens

        for item in must + optional:
            tokens = item.effective_tokens
            if tokens > remaining and self.config.enable_compression and tokens > self.config.compression_threshold:
                target = min(int(tokens * self.config.target_compression_ratio), remaining - 50)
                if target > 100:
                    item.compressed_content = self.compressor.compress(item.content, target, item.content_type)
                    tokens = item.effective_tokens

            if tokens <= remaining:
                selected.append(item)
                remaining -= tokens
            if remaining < 100:
                break

        return selected

    def _score_items(self, items: List[ContextItem]) -> None:
        now = datetime.now(timezone.utc)
        for item in items:
            recency = max(0, 1 - ((now - item.timestamp).total_seconds() / 604800)) if item.timestamp else 0.5
            item.selection_score = (
                item.relevance * self.config.relevance_weight +
                item.importance * self.config.importance_weight +
                recency * self.config.recency_weight
            )
            if item.must_include:
                item.selection_score = 10.0


# === FORMATTING ===

class ContextFormatter:
    """Format selected contexts for LLM consumption."""

    def __init__(self, config: ContextConfig):
        self.config = config

    def format(self, items: List[ContextItem], style: FormatStyle) -> str:
        if not items:
            return ""
        formatters = {
            FormatStyle.MARKDOWN: self._markdown,
            FormatStyle.XML: self._xml,
            FormatStyle.NUMBERED: self._numbered,
            FormatStyle.PLAIN: self._plain,
        }
        return formatters.get(style, self._plain)(items)

    def _markdown(self, items: List[ContextItem]) -> str:
        parts = ["## Retrieved Context\n"]
        for i, item in enumerate(items, 1):
            src = self._truncate(item.source)
            header = f"### Source {i}: {src}\n" if self.config.include_source_attribution else f"### Context {i}\n"
            content = f"```\n{item.effective_content}\n```\n" if item.content_type == ContentType.CODE else f"{item.effective_content}\n"
            parts.extend([header, content])
        return "\n".join(parts)

    def _xml(self, items: List[ContextItem]) -> str:
        parts = ["<retrieved_context>"]
        for i, item in enumerate(items, 1):
            parts.extend([f'  <context id="{i}" source="{self._truncate(item.source)}">', f"    {item.effective_content}", "  </context>"])
        parts.append("</retrieved_context>")
        return "\n".join(parts)

    def _numbered(self, items: List[ContextItem]) -> str:
        parts = ["Retrieved Context:"]
        for i, item in enumerate(items, 1):
            prefix = f"\n[{i}] ({self._truncate(item.source)})" if self.config.include_source_attribution else f"\n[{i}]"
            parts.extend([prefix, item.effective_content])
        return "\n".join(parts)

    def _plain(self, items: List[ContextItem]) -> str:
        return "\n\n---\n\n".join(item.effective_content for item in items)

    def _truncate(self, s: str) -> str:
        m = self.config.max_source_label_length
        return s if len(s) <= m else s[:m-3] + "..."


# === MAIN MANAGER ===

class ContextManager:
    """
    Main context window manager for RAG pipelines.

    Usage:
        manager = ContextManager(max_tokens=8000)
        manager.add_context(ContextItem(content="...", relevance=0.9, source="doc1"))
        result = manager.select_and_format(query="What is RAG?", reserved_tokens=500)
    """

    def __init__(self, max_tokens: int = 8000, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig(max_context_tokens=max_tokens)
        self.config.max_context_tokens = max_tokens
        self.selector = ContextSelector(self.config)
        self.formatter = ContextFormatter(self.config)
        self._items: List[ContextItem] = []

    def add_context(self, item: ContextItem) -> None:
        self._items.append(item)

    def add_contexts(self, items: List[ContextItem]) -> None:
        self._items.extend(items)

    def clear(self) -> None:
        self._items.clear()

    def select_and_format(self, query: Optional[str] = None, reserved_tokens: int = 500, format_style: Optional[FormatStyle] = None) -> FormattedContext:
        style = format_style or self.config.default_format
        query_tokens = count_tokens(query) if query else 0
        available = self.config.max_context_tokens - self.config.query_reserve_tokens - reserved_tokens - query_tokens

        selected = self.selector.select(self._items, available, query)
        content = self.formatter.format(selected, style)
        content_tokens = count_tokens(content)

        return FormattedContext(
            content=content, total_tokens=content_tokens, items_included=len(selected),
            items_excluded=len(self._items) - len(selected), tokens_available=available,
            tokens_used=content_tokens, format_style=style,
            sources=[i.source for i in selected],
            compression_applied=any(i.compressed_content for i in selected)
        )

    def get_budget_info(self, reserved_tokens: int = 500) -> Dict[str, int]:
        total = sum(i.token_count for i in self._items)
        available = self.config.max_context_tokens - self.config.query_reserve_tokens - reserved_tokens
        return {"max_tokens": self.config.max_context_tokens, "available": available, "total_item_tokens": total, "items_count": len(self._items), "can_fit_all": total <= available}

    @property
    def items(self) -> List[ContextItem]:
        return self._items.copy()


# === FACTORY FUNCTIONS ===

def create_context_manager(max_tokens: int = 8000, **kwargs) -> ContextManager:
    return ContextManager(max_tokens=max_tokens, config=ContextConfig(max_context_tokens=max_tokens, **kwargs))

def create_context_item(content: str, relevance: float = 0.5, source: str = "unknown", must_include: bool = False, **metadata) -> ContextItem:
    return ContextItem(content=content, relevance=relevance, source=source, must_include=must_include, metadata=metadata)


__all__ = [
    "ContextManager", "ContextSelector", "ContextFormatter", "ContextCompressor", "TokenCounter",
    "ContextConfig", "ContextItem", "FormattedContext", "FormatStyle", "ContentType",
    "create_context_manager", "create_context_item", "count_tokens",
]
