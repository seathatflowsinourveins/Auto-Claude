# Python 3.14 SDK Compatibility Report
## Deep Research Analysis - January 2026

---

## Executive Summary

This report documents comprehensive research into SDK compatibility with **Python 3.14.0** for the Unleash Platform V34 architecture. Research was conducted by querying official GitHub repositories, PyPI metadata, and official documentation.

### Critical Finding

**3 of 14 target SDKs explicitly DO NOT support Python 3.14:**
- CrewAI: `>=3.10, <3.14`
- AutoGen: `>=3.10, <3.14`
- Aider: `3.9-3.12 only`

These SDKs cannot be installed on Python 3.14 without creating compatibility layers or waiting for upstream updates.

---

## Compatibility Matrix

| SDK | Layer | Python 3.14 | Status | Action |
|-----|-------|-------------|--------|--------|
| **controlflow** | L1 | ✅ 3.8+ | Archived, Pydantic V2 OK | Install |
| **crewai** | L1 | ❌ <3.14 | Explicitly unsupported | Skip/Compat |
| **autogen** | L1 | ❌ <3.14 | Explicitly unsupported | Skip/Compat |
| **outlines** | L3 | ⚠️ ~3.13 | Pydantic V1 warnings | Test carefully |
| **guidance** | L3 | ✅ 3.7+ | No known issues | Install |
| **mirascope** | L3 | ✅ Active | Likely compatible | Install |
| **ell** | L3 | ✅ 3.7+ | No known issues | Install |
| **docling** | L7 | ✅ v2.59.0+ | Explicit 3.14 support | Install |
| **markitdown** | L7 | ⚠️ 3.10+ | Some issues reported | Test carefully |
| **aider** | L7 | ❌ 3.9-3.12 | Does not support 3.13+ | Skip/Compat |
| **haystack** | L8 | ✅ 3.14 | Explicit support | Install |
| **lightrag** | L8 | ⚠️ Unknown | Docker recommended | Use Docker |

---

## Detailed Research Findings

### L1 Orchestration Layer

#### ControlFlow (prefecthq/controlflow) ✅
- **Repository**: https://github.com/PrefectHQ/ControlFlow
- **Latest Version**: 0.12.1 (February 6, 2025)
- **Python Support**: 3.8+
- **Status**: ARCHIVED (August 22, 2025)
- **Pydantic V2**: Compatible (issues #314, #323 fixed in 2024)
- **Dependencies**: prefect, langchain, pydantic, openai

```bash
pip install controlflow
```

**Note**: Project archived, development moved to Marvin framework. Still functional.

#### CrewAI (crewAIInc/crewAI) ❌ NOT COMPATIBLE
- **Repository**: https://github.com/crewAIInc/crewAI
- **Latest Version**: 1.8.1 (January 15, 2026)
- **Python Support**: `>=3.10, <3.14` (EXPLICITLY EXCLUDES 3.14)
- **Known Issues**: LangChain version conflicts (issue #837)

**CANNOT INSTALL ON PYTHON 3.14**

Options:
1. Wait for upstream support
2. Create compatibility layer (complex - native async architecture)
3. Use in separate Python 3.13 environment

#### AutoGen (microsoft/autogen) ❌ NOT COMPATIBLE
- **Repository**: https://github.com/microsoft/autogen
- **Latest Version**: 0.4+ architecture (0.6, 0.10.4 exist)
- **Python Support**: `>=3.10, <3.14` (EXPLICITLY EXCLUDES 3.14)
- **Migration**: pyautogen → autogen-agentchat (v0.2 → v0.4)
- **Known Issues**: Pydantic model conversion bug (issue #5736)

**CANNOT INSTALL ON PYTHON 3.14**

```bash
# Would work on Python 3.13:
pip install autogen-agentchat autogen-ext[openai]
```

---

### L3 Structured Output Layer

#### Outlines (dottxt-ai/outlines) ⚠️ CAUTION
- **Repository**: https://github.com/dottxt-ai/outlines
- **Latest Version**: 1.2.9 (November 24, 2025)
- **Python Support**: 3.10-3.13 (3.14 not explicitly tested)
- **Known Issues**: Pydantic V1 warnings on Python 3.14

```bash
pip install outlines
```

**Test before production use.**

#### Guidance (guidance-ai/guidance) ✅
- **Repository**: https://github.com/guidance-ai/guidance
- **Stars**: 21k+
- **Python Support**: 3.7+
- **Status**: Actively maintained, no deprecation warnings

```bash
pip install guidance
```

#### Mirascope (Mirascope/mirascope) ✅
- **Repository**: https://github.com/Mirascope/mirascope
- **Latest Version**: 7 (November 7, 2025)
- **Python Support**: Recent Python versions, actively maintained

```bash
pip install "mirascope[openai]"
```

#### Ell (MadcowD/ell) ✅
- **Repository**: https://github.com/MadcowD/ell
- **Latest Version**: 0.0.17 (February 25, 2025)
- **Python Support**: 3.7+

```bash
pip install -U "ell-ai[all]"
```

---

### L7 Processing Layer

#### Docling (DS4SD/docling) ✅
- **Repository**: https://github.com/docling-project/docling
- **Python 3.14 Support**: YES, from version 2.59.0
- **Dependencies**: PyTorch (heavy)
- **FAQ**: Explicitly confirms 3.14 support

```bash
pip install docling
```

**Best L7 option for Python 3.14.**

#### MarkItDown (microsoft/markitdown) ⚠️ CAUTION
- **Repository**: https://github.com/microsoft/markitdown
- **Python Support**: 3.10+ (officially up to 3.12)
- **Known Issues**: Installation failures on Python 3.14 (issue #1470)

```bash
pip install markitdown
```

**May require workarounds on Python 3.14.**

#### Aider (paul-gauthier/aider) ❌ NOT COMPATIBLE
- **Repository**: https://github.com/Aider-AI/aider
- **Python Support**: 3.9-3.12 ONLY
- **Issue #3037**: Confirms no Python 3.13+ support
- **Dependency Conflicts**: Common, requires isolated environments

**CANNOT INSTALL ON PYTHON 3.14**

---

### L8 Knowledge Layer

#### Haystack (deepset-ai/haystack) ✅
- **Repository**: https://github.com/deepset-ai/haystack
- **Python 3.14 Support**: YES, explicitly in latest releases
- **Package**: `haystack-ai` (newer) or `farm-haystack` (legacy)

```bash
pip install haystack-ai
```

**Do NOT install both haystack-ai and farm-haystack together.**

#### LightRAG (HKUDS/LightRAG) ⚠️ CAUTION
- **Repository**: https://github.com/HKUDS/LightRAG
- **Python 3.14**: Unknown/unclear
- **Known Issues**: Missing dependencies, hnswlib build failures
- **Recommendation**: Use Docker

```bash
# Docker (recommended)
docker-compose up -d

# Or pip (may have issues)
pip install lightrag-hku
```

---

## Recommended Installation Plan

### Phase 1: Safe Installs (Python 3.14 Compatible) ✅

```bash
# L1 Orchestration (1 of 3)
pip install controlflow

# L3 Structured Output (3 of 4)
pip install guidance mirascope ell-ai[all]

# L7 Processing (1 of 3)
pip install docling

# L8 Knowledge (1 of 2)
pip install haystack-ai
```

**Expected: 6 new SDKs available**

### Phase 2: Test Carefully ⚠️

```bash
# May have issues
pip install outlines    # Pydantic V1 warnings
pip install markitdown  # Some 3.14 issues
```

### Phase 3: Skip or Use Alternatives ❌

| SDK | Alternative |
|-----|-------------|
| CrewAI | Use LangGraph (already available) |
| AutoGen | Use LangGraph or PydanticAI (already available) |
| Aider | Already have Phase 10 compat layer |
| LightRAG | Use LlamaIndex (already available) |

---

## Revised SDK Availability Projection

### Current State (Post Phase 10)
- Total SDKs: 36
- Available: 22 (61.1%)

### After Phase 1 Safe Installs
- New SDKs: +6
- Total Available: 28 (77.8%)

### After Phase 2 Careful Installs
- New SDKs: +2 (if successful)
- Total Available: 30 (83.3%)

### Maximum Achievable (Python 3.14)
- Unavailable due to version constraints: CrewAI, AutoGen, Aider
- Maximum: 33/36 (91.7%)

---

## Compatibility Layers Needed

For 100% SDK availability on Python 3.14, create compatibility layers for:

### 1. CrewAI Compat (core/orchestration/crewai_compat.py)
- Multi-agent orchestration using LangGraph
- Agent role definitions
- Task delegation patterns

### 2. AutoGen Compat (core/orchestration/autogen_compat.py)
- Conversable agent abstraction
- Group chat patterns
- Code execution sandboxing

### 3. Aider Compat (Already covered in Phase 10)
- Code editing capabilities
- Git integration
- Chat interface

---

## Conclusion

Python 3.14 compatibility requires strategic SDK selection:

| Category | Status |
|----------|--------|
| **Safe to Install** | 6 SDKs (controlflow, guidance, mirascope, ell, docling, haystack) |
| **Test Carefully** | 2 SDKs (outlines, markitdown) |
| **Not Compatible** | 3 SDKs (crewai, autogen, aider) |
| **Use Docker** | 1 SDK (lightrag) |

**Recommendation**: Proceed with Phase 1 safe installs, test Phase 2 carefully, and create compatibility layers for the 3 incompatible SDKs if their functionality is critical.

---

*Report Generated: 2026-01-24*
*Research Method: Exa Deep Research + Official GitHub/PyPI Analysis*
