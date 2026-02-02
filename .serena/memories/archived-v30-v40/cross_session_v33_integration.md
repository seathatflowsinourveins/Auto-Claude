# Cross-Session V33 Integration Layer

**Created**: 2026-01-23
**File**: `C:\Users\42\.claude\hooks\cross_session_v33.py`

## Purpose
Enhanced cross-session integration with dynamic thinking budgets, HiAgent hierarchical memory, and pattern extraction from everything-claude-code patterns.

## Key Classes

### ThinkingConfig
```python
@dataclass
class ThinkingConfig:
    budget: int           # Token budget (0-128000)
    model: str            # haiku/sonnet/opus
    prompt_prefix: str    # Guidance for thinking style
    complexity: str       # trivial/low/medium/high/ultrathink
```

### TaskComplexity Enum
```python
class TaskComplexity(Enum):
    TRIVIAL = ("trivial", 0, "haiku", "Respond directly")
    LOW = ("low", 4000, "sonnet", "Outline reasoning in 2-3 sentences")
    MEDIUM = ("medium", 10000, "sonnet", "Explain approach and key decisions")
    HIGH = ("high", 32000, "opus", "Step-by-step, considering alternatives")
    ULTRATHINK = ("ultrathink", 128000, "opus", "Comprehensive multi-angle analysis")
```

### HiAgentMemoryV33
4-tier hierarchical memory manager.

```python
class HiAgentMemoryV33:
    WORKING_LIMIT = 100
    EPISODIC_LIMIT = 1000
    
    def add(self, item: Dict, tier: str = "working")
    def search(self, query: str, tiers: List[str]) -> List[Dict]
    def promote(self, item_id: str, target_tier: str)
    def promote_on_compaction(self)  # Working → Episodic
```

### CrossSessionV33
Main integration class.

```python
class CrossSessionV33:
    def pre_task_setup(self, task: str, context: str = None) -> Dict:
        """
        Returns:
        - thinking_config: ThinkingConfig for the task
        - relevant_memories: From hierarchical search
        - context_fusion: Merged context from all sources
        """
    
    def post_task_capture(self, task: str, result: str, files: List[str]) -> Dict:
        """
        Returns:
        - patterns_extracted: From continuous learning
        - memories_added: New memories created
        - promotions: Memory tier promotions
        """
    
    def pre_compact(self) -> Dict:
        """
        Returns:
        - promoted_items: Working → Episodic promotions
        - summary: Compaction summary for context preservation
        """
```

## Key Functions

### detect_task_complexity
```python
def detect_task_complexity(task: str, context: str = None) -> ThinkingConfig:
    # 1. Check explicit keyword triggers
    if "ultrathink" in task.lower() or "megathink" in task.lower():
        return ThinkingConfig(128000, "opus", "...", "ultrathink")
    
    # 2. Check semantic complexity indicators
    high_indicators = ["architecture", "design", "trade-off", "complex"]
    medium_indicators = ["debug", "fix", "error", "refactor"]
    low_indicators = ["explain", "describe", "what is"]
    trivial_indicators = ["simple", "quick", "just"]
    
    # 3. Return appropriate config
    ...
```

### extract_patterns
```python
def extract_patterns(session_data: Dict) -> Dict:
    """Extract learnings using continuous-learning patterns."""
    return {
        "error_resolution": find_error_fixes(session_data),
        "user_corrections": find_corrections(session_data),
        "workarounds": find_workarounds(session_data),
        "debugging_techniques": find_debug_patterns(session_data)
    }
```

## Hook Integration Points

### Pre-Task Hook
```python
# In session-start hook
v33 = CrossSessionV33()
setup = v33.pre_task_setup(user_task, conversation_context)
# Apply thinking_config to Claude request
# Inject relevant_memories into context
```

### Post-Task Hook
```python
# In post-tool hook (after significant operations)
capture = v33.post_task_capture(task, result, modified_files)
# Save patterns_extracted to learnings
# Add memories_added to appropriate tiers
```

### Pre-Compact Hook
```python
# Before context compaction
compact_prep = v33.pre_compact()
# Ensure promoted_items are preserved
# Include summary in compacted context
```

## Type Annotations Notes
The file has some Pyright warnings (lines 505, 566, 633-635) for Optional parameters.
These are type annotation issues, not runtime errors. The code functions correctly.
Future fix: Add proper type guards for Optional parameters.
