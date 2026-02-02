# Phase 11: Complete SDK Installation & Final Validation

## Current Status: V34 (22/36 SDKs = 61.1%)

### FIXED (100%):
- L0 Protocol: anthropic, openai, mcp ✅
- L5 Observability: langfuse, phoenix, opik, deepeval, ragas, logfire, otel ✅
- L6 Safety: llm_guard, nemoguardrails ✅

### NEED INSTALLATION:

## Step 1: Install L1 Orchestration SDKs
```bash
pip install controlflow crewai pyautogen
```

Validation:
```python
import controlflow
import crewai
import autogen
print("L1 Orchestration OK")
```

## Step 2: Install L3 Structured Output SDKs
```bash
pip install outlines guidance mirascope ell
```

Validation:
```python
import outlines
import guidance
import mirascope
print("L3 Structured OK")
```

## Step 3: Install L7 Processing SDKs
```bash
pip install docling markitdown
```

Validation:
```python
import docling
import markitdown
print("L7 Processing OK")
```

## Step 4: Install L8 Knowledge SDKs
```bash
pip install haystack-ai lightrag
```

Validation:
```python
from haystack import Pipeline
import lightrag
print("L8 Knowledge OK")
```

## Step 5: Fix zep_python (L2 Memory)

Create `core/memory/zep_compat.py`:
```python
"""Zep compatibility layer for Python 3.14+"""
import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class ZepMessage:
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ZepSession:
    session_id: str
    messages: List[ZepMessage] = field(default_factory=list)

class ZepCompat:
    """HTTP-based Zep client compatible with Python 3.14+"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.getzep.com"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
    
    def add_memory(self, session_id: str, messages: List[dict]) -> dict:
        response = self.client.post(
            f"/sessions/{session_id}/memory",
            json={"messages": messages}
        )
        return response.json()
    
    def get_memory(self, session_id: str, lastn: int = 10) -> dict:
        response = self.client.get(
            f"/sessions/{session_id}/memory",
            params={"lastn": lastn}
        )
        return response.json()
    
    def search(self, session_id: str, query: str, limit: int = 5) -> List[dict]:
        response = self.client.post(
            f"/sessions/{session_id}/search",
            json={"query": query, "limit": limit}
        )
        return response.json()
    
    def delete_session(self, session_id: str) -> bool:
        response = self.client.delete(f"/sessions/{session_id}")
        return response.status_code == 200

ZEP_COMPAT_AVAILABLE = True
```

## Step 6: Final Validation Script

Create `scripts/validate_v34_final.py`:
```python
#!/usr/bin/env python3
"""V34 Final Validation - All 36 SDKs"""

import sys
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SDKCheck:
    name: str
    layer: str
    import_path: str
    status: str = "pending"

SDKS = [
    # L0 Protocol
    SDKCheck("anthropic", "L0", "anthropic"),
    SDKCheck("openai", "L0", "openai"),
    SDKCheck("mcp", "L0", "mcp"),
    
    # L1 Orchestration
    SDKCheck("langgraph", "L1", "langgraph"),
    SDKCheck("pydantic_ai", "L1", "pydantic_ai"),
    SDKCheck("instructor", "L1", "instructor"),
    SDKCheck("controlflow", "L1", "controlflow"),
    SDKCheck("crewai", "L1", "crewai"),
    SDKCheck("autogen", "L1", "autogen"),
    
    # L2 Memory
    SDKCheck("mem0", "L2", "mem0"),
    SDKCheck("graphiti_core", "L2", "graphiti_core"),
    SDKCheck("letta", "L2", "letta"),
    SDKCheck("zep_compat", "L2", "core.memory.zep_compat"),
    
    # L3 Structured
    SDKCheck("pydantic", "L3", "pydantic"),
    SDKCheck("outlines", "L3", "outlines"),
    SDKCheck("guidance", "L3", "guidance"),
    SDKCheck("mirascope", "L3", "mirascope"),
    SDKCheck("ell", "L3", "ell"),
    
    # L4 Reasoning
    SDKCheck("dspy", "L4", "dspy"),
    SDKCheck("agentlite", "L4", "agentlite"),
    
    # L5 Observability (FIXED)
    SDKCheck("langfuse_compat", "L5", "core.observability.langfuse_compat"),
    SDKCheck("opik", "L5", "opik"),
    SDKCheck("deepeval", "L5", "deepeval"),
    SDKCheck("ragas", "L5", "ragas"),
    SDKCheck("logfire", "L5", "logfire"),
    SDKCheck("opentelemetry", "L5", "opentelemetry"),
    
    # L6 Safety (FIXED)
    SDKCheck("scanner_compat", "L6", "core.safety.scanner_compat"),
    SDKCheck("rails_compat", "L6", "core.safety.rails_compat"),
    
    # L7 Processing
    SDKCheck("docling", "L7", "docling"),
    SDKCheck("markitdown", "L7", "markitdown"),
    SDKCheck("aider", "L7", "aider"),
    
    # L8 Knowledge
    SDKCheck("llama_index", "L8", "llama_index"),
    SDKCheck("firecrawl", "L8", "firecrawl"),
    SDKCheck("haystack", "L8", "haystack"),
    SDKCheck("lightrag", "L8", "lightrag"),
]

def check_sdk(sdk: SDKCheck) -> bool:
    try:
        __import__(sdk.import_path)
        sdk.status = "✅"
        return True
    except Exception as e:
        sdk.status = f"❌ {str(e)[:30]}"
        return False

def main():
    print("=" * 60)
    print(" V34 FINAL VALIDATION")
    print("=" * 60)
    
    passed = 0
    total = len(SDKS)
    
    for sdk in SDKS:
        if check_sdk(sdk):
            passed += 1
        print(f"[{sdk.layer}] {sdk.name}: {sdk.status}")
    
    print("=" * 60)
    print(f"RESULT: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 V34 COMPLETE - 100% SDK AVAILABILITY!")
        sys.exit(0)
    else:
        print(f"⚠️  Missing {total - passed} SDKs")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Step 7: Run Final Validation
```bash
python scripts/validate_v34_final.py
```

## Success Criteria
- All 36 SDKs importable
- No Pydantic v1 errors
- No Python 3.14 compatibility issues
- Result: V34 100% (36/36)
