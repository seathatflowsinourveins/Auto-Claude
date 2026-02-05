# Phase 5: Structured Output Layer Setup

**Layer 3 - 4 Structured Output SDKs**

Execute this prompt in Claude Code CLI to implement the Structured Output Layer.

---

## Pre-Flight Validation

Before beginning, verify Phase 4 is complete:

```bash
# Verify Phase 4 validation passes
cd /path/to/unleash
python scripts/validate_phase4.py

# Expected: "Phase 4 Validation PASSED"
```

Quick verification commands:

```bash
# Check Phase 4 Memory Layer
python -c "from core.memory import MemoryProviderFactory; print('Phase 4 OK')"

# Check Phase 3 Orchestration
python -c "from core.orchestration import UnifiedOrchestrator; print('Phase 3 OK')"

# Check Phase 2 Protocol
python -c "from core import LLMGateway; print('Phase 2 OK')"
```

**Required Checks:**
- [ ] Memory Layer operational (`core/memory/providers.py`)
- [ ] Orchestration Layer functional (`core/orchestration/`)
- [ ] LLM Gateway working (`core/llm_gateway.py`)
- [ ] All Phase 4 SDKs installed

---

## Phase 5 Objectives

Implement the Structured Output Layer with 4 core SDKs:
1. **instructor** - Structured LLM outputs with Pydantic validation
2. **baml** - Type-safe LLM functions with schema validation
3. **outlines** - Constrained text generation with grammar/regex
4. **pydantic-ai** - Pydantic-native AI agent framework

---

## Step 1: Install Phase 5 Dependencies

```bash
# Activate virtual environment first
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate

# Install structured output layer dependencies
uv pip install instructor outlines pydantic-ai

# BAML has a separate installation process
pip install baml-py

# Verify installation
python -c "import instructor; import outlines; import pydantic_ai; print('Structured Output SDKs installed')"
```

---

## Step 2: Create Structured Directory Structure

```bash
mkdir -p core/structured
```

---

## Step 3: Create Instructor Integration

Create `core/structured/instructor_chains.py`:

```python
#!/usr/bin/env python3
"""
Instructor Integration for Structured LLM Outputs
Pydantic-validated responses from LLMs with automatic retries.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import Instructor
try:
    import instructor
    from instructor import Instructor
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logger.warning("instructor not available - install with: pip install instructor")

# Import LLM clients
try:
    from anthropic import Anthropic
    from openai import OpenAI
    CLIENTS_AVAILABLE = True
except ImportError:
    CLIENTS_AVAILABLE = False


# Type variable for generic response models
T = TypeVar("T", bound=BaseModel)


# ============================================
# Common Response Models
# ============================================

class Sentiment(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ClassificationResult(BaseModel):
    """Generic classification result."""
    label: str = Field(..., description="Classification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Explanation for classification")


class Entity(BaseModel):
    """Extracted entity."""
    text: str = Field(..., description="Entity text")
    entity_type: str = Field(..., description="Type of entity (person, org, location, etc.)")
    start_index: Optional[int] = Field(None, description="Start position in text")
    end_index: Optional[int] = Field(None, description="End position in text")


class ExtractionResult(BaseModel):
    """Extraction result with multiple entities."""
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    summary: str = Field(..., description="Summary of extraction")


class SentimentResult(BaseModel):
    """Sentiment analysis result."""
    sentiment: Sentiment = Field(..., description="Overall sentiment")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in sentiment")
    key_phrases: list[str] = Field(default_factory=list, description="Key phrases influencing sentiment")
    reasoning: str = Field(..., description="Explanation for sentiment")


class SummaryResult(BaseModel):
    """Text summarization result."""
    summary: str = Field(..., description="Concise summary")
    key_points: list[str] = Field(default_factory=list, description="Key points from text")
    word_count: int = Field(..., ge=1, description="Word count of summary")
    
    @field_validator("word_count", mode="before")
    @classmethod
    def calculate_word_count(cls, v, info):
        """Auto-calculate word count if not provided."""
        if v is None and "summary" in info.data:
            return len(info.data["summary"].split())
        return v


class QuestionAnswer(BaseModel):
    """Question answering result."""
    question: str = Field(..., description="The question asked")
    answer: str = Field(..., description="The answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in answer")
    sources: list[str] = Field(default_factory=list, description="Sources used")
    is_answerable: bool = Field(True, description="Whether question is answerable")


class CodeGenerationResult(BaseModel):
    """Code generation result."""
    code: str = Field(..., description="Generated code")
    language: str = Field(..., description="Programming language")
    explanation: str = Field(..., description="Explanation of the code")
    dependencies: list[str] = Field(default_factory=list, description="Required dependencies")


class DataExtractionResult(BaseModel):
    """Structured data extraction from unstructured text."""
    data: dict[str, Any] = Field(default_factory=dict, description="Extracted structured data")
    schema_version: str = Field(default="1.0", description="Schema version")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Data completeness score")
    missing_fields: list[str] = Field(default_factory=list, description="Fields that couldn't be extracted")


# ============================================
# Instructor Client
# ============================================

@dataclass
class InstructorClient:
    """
    Instructor-based structured output client.
    
    Provides validated, typed responses from LLMs using Pydantic models.
    
    Usage:
        client = InstructorClient()
        result = client.extract(
            model_class=SentimentResult,
            prompt="Analyze sentiment: I love this product!",
        )
    """
    
    provider: str = "anthropic"  # anthropic or openai
    model: str = "claude-3-5-sonnet-20241022"
    max_retries: int = 3
    _client: Optional[Any] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the instructor-patched client."""
        if not INSTRUCTOR_AVAILABLE:
            raise ImportError("instructor not installed - pip install instructor")
        
        if not CLIENTS_AVAILABLE:
            raise ImportError("anthropic and openai clients required")
        
        self._setup_client()
        logger.info("instructor_client_initialized", provider=self.provider, model=self.model)
    
    def _setup_client(self) -> None:
        """Setup the instructor-patched LLM client."""
        if self.provider == "anthropic":
            base_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self._client = instructor.from_anthropic(base_client)
        elif self.provider == "openai":
            base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._client = instructor.from_openai(base_client)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def extract(
        self,
        model_class: Type[T],
        prompt: str,
        system: str = "You are a helpful assistant that extracts structured data.",
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> T:
        """
        Extract structured data using a Pydantic model.
        
        Args:
            model_class: Pydantic model class for the response
            prompt: User prompt
            system: System message
            model: Model to use (defaults to instance model)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            **kwargs: Additional API parameters
            
        Returns:
            Validated Pydantic model instance
        """
        model = model or self.model
        
        try:
            if self.provider == "anthropic":
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    system=system,
                    response_model=model_class,
                    max_retries=self.max_retries,
                    **kwargs,
                )
            else:  # openai
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    response_model=model_class,
                    max_retries=self.max_retries,
                    temperature=temperature,
                    **kwargs,
                )
            
            logger.info(
                "instructor_extraction_success",
                model_class=model_class.__name__,
                provider=self.provider,
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "instructor_extraction_failed",
                model_class=model_class.__name__,
                error=str(e),
            )
            raise
    
    def classify(
        self,
        text: str,
        labels: list[str],
        model: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify text into given labels.
        
        Args:
            text: Text to classify
            labels: Possible classification labels
            model: Model to use
            
        Returns:
            Classification result
        """
        class DynamicClassification(BaseModel):
            label: str = Field(..., description=f"One of: {', '.join(labels)}")
            confidence: float = Field(..., ge=0.0, le=1.0)
            reasoning: str
        
        prompt = f"""Classify the following text into one of these categories: {', '.join(labels)}

Text: {text}

Provide the classification label, confidence score (0-1), and reasoning."""
        
        result = self.extract(
            model_class=DynamicClassification,
            prompt=prompt,
            system="You are an expert text classifier.",
            model=model,
        )
        
        return ClassificationResult(
            label=result.label,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
    
    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        model: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract named entities from text.
        
        Args:
            text: Text to extract entities from
            entity_types: Types of entities to extract
            model: Model to use
            
        Returns:
            Extraction result with entities
        """
        prompt = f"""Extract all {', '.join(entity_types)} entities from the following text.

Text: {text}

For each entity, provide the text, type, and position if possible."""
        
        return self.extract(
            model_class=ExtractionResult,
            prompt=prompt,
            system="You are an expert named entity recognizer.",
            model=model,
        )
    
    def analyze_sentiment(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            model: Model to use
            
        Returns:
            Sentiment analysis result
        """
        prompt = f"""Analyze the sentiment of the following text.

Text: {text}

Provide the overall sentiment (positive/negative/neutral), confidence, key phrases, and reasoning."""
        
        return self.extract(
            model_class=SentimentResult,
            prompt=prompt,
            system="You are an expert sentiment analyst.",
            model=model,
        )
    
    def summarize(
        self,
        text: str,
        max_words: int = 100,
        model: Optional[str] = None,
    ) -> SummaryResult:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            max_words: Maximum words in summary
            model: Model to use
            
        Returns:
            Summary result
        """
        prompt = f"""Summarize the following text in {max_words} words or less.

Text: {text}

Provide a concise summary and list key points."""
        
        return self.extract(
            model_class=SummaryResult,
            prompt=prompt,
            system="You are an expert summarizer.",
            model=model,
        )
    
    def answer_question(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
    ) -> QuestionAnswer:
        """
        Answer a question given context.
        
        Args:
            question: Question to answer
            context: Context to use for answering
            model: Model to use
            
        Returns:
            Question answer result
        """
        prompt = f"""Answer the following question based on the context provided.

Context: {context}

Question: {question}

If the question cannot be answered from the context, indicate that."""
        
        return self.extract(
            model_class=QuestionAnswer,
            prompt=prompt,
            system="You are an expert question answerer.",
            model=model,
        )
    
    def generate_code(
        self,
        description: str,
        language: str = "python",
        model: Optional[str] = None,
    ) -> CodeGenerationResult:
        """
        Generate code from description.
        
        Args:
            description: Description of code to generate
            language: Programming language
            model: Model to use
            
        Returns:
            Code generation result
        """
        prompt = f"""Generate {language} code that accomplishes the following:

{description}

Provide the code, explanation, and any required dependencies."""
        
        return self.extract(
            model_class=CodeGenerationResult,
            prompt=prompt,
            system=f"You are an expert {language} programmer.",
            model=model,
        )
    
    def extract_structured_data(
        self,
        text: str,
        schema: dict[str, str],
        model: Optional[str] = None,
    ) -> DataExtractionResult:
        """
        Extract structured data according to a schema.
        
        Args:
            text: Text to extract from
            schema: Dict mapping field names to descriptions
            model: Model to use
            
        Returns:
            Extracted structured data
        """
        schema_desc = "\n".join([f"- {k}: {v}" for k, v in schema.items()])
        
        prompt = f"""Extract the following fields from the text:

{schema_desc}

Text: {text}

Extract all fields that are present. Note any fields that cannot be extracted."""
        
        return self.extract(
            model_class=DataExtractionResult,
            prompt=prompt,
            system="You are an expert data extractor.",
            model=model,
        )


# ============================================
# Convenience Functions
# ============================================

def quick_extract(
    model_class: Type[T],
    prompt: str,
    provider: str = "anthropic",
) -> T:
    """Quick extraction with defaults."""
    client = InstructorClient(provider=provider)
    return client.extract(model_class=model_class, prompt=prompt)


def analyze_text(text: str, task: str = "sentiment") -> BaseModel:
    """
    Analyze text for various tasks.
    
    Args:
        text: Text to analyze
        task: One of: sentiment, entities, summary
        
    Returns:
        Analysis result
    """
    client = InstructorClient()
    
    if task == "sentiment":
        return client.analyze_sentiment(text)
    elif task == "entities":
        return client.extract_entities(text, ["person", "organization", "location"])
    elif task == "summary":
        return client.summarize(text)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    print("Instructor Chains Module")
    print("-" * 40)
    
    if not INSTRUCTOR_AVAILABLE:
        print("Instructor not installed. Install with: pip install instructor")
    else:
        print("Available features:")
        print("  - InstructorClient: Main client for structured extraction")
        print("  - quick_extract(): Quick extraction with defaults")
        print("  - analyze_text(): Analyze text for sentiment/entities/summary")
        print("\nExample usage:")
        print("  client = InstructorClient()")
        print("  result = client.analyze_sentiment('I love this!')")
```

---

## Step 4: Create BAML Type-Safe Functions

Create `core/structured/baml_functions.py`:

```python
#!/usr/bin/env python3
"""
BAML Type-Safe Functions
Type-safe LLM function definitions with schema validation.
"""

from __future__ import annotations

import os
import json
from typing import Any, Optional, Type, TypeVar, Generic, Callable
from dataclasses import dataclass, field
from functools import wraps

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import BAML
try:
    from baml_py import BAML, baml_function
    BAML_AVAILABLE = True
except ImportError:
    BAML_AVAILABLE = False
    logger.warning("baml-py not available - install with: pip install baml-py")


T = TypeVar("T", bound=BaseModel)


# ============================================
# Type Definitions for BAML
# ============================================

class TypeDef(BaseModel):
    """Type definition for BAML schemas."""
    name: str
    description: str
    fields: dict[str, str] = Field(default_factory=dict)
    examples: list[dict[str, Any]] = Field(default_factory=list)


class FunctionDef(BaseModel):
    """Function definition for BAML."""
    name: str
    description: str
    input_type: str
    output_type: str
    prompt_template: str
    examples: list[dict[str, Any]] = Field(default_factory=list)


# ============================================
# BAML Schema Registry
# ============================================

class SchemaRegistry:
    """Registry for BAML type and function schemas."""
    
    def __init__(self):
        self._types: dict[str, TypeDef] = {}
        self._functions: dict[str, FunctionDef] = {}
    
    def register_type(self, type_def: TypeDef) -> None:
        """Register a type definition."""
        self._types[type_def.name] = type_def
        logger.info("type_registered", name=type_def.name)
    
    def register_function(self, func_def: FunctionDef) -> None:
        """Register a function definition."""
        self._functions[func_def.name] = func_def
        logger.info("function_registered", name=func_def.name)
    
    def get_type(self, name: str) -> Optional[TypeDef]:
        """Get a registered type."""
        return self._types.get(name)
    
    def get_function(self, name: str) -> Optional[FunctionDef]:
        """Get a registered function."""
        return self._functions.get(name)
    
    def list_types(self) -> list[str]:
        """List all registered types."""
        return list(self._types.keys())
    
    def list_functions(self) -> list[str]:
        """List all registered functions."""
        return list(self._functions.keys())
    
    def to_baml_schema(self) -> str:
        """Generate BAML schema string."""
        schema_parts = []
        
        # Generate type definitions
        for type_def in self._types.values():
            fields = "\n  ".join([f"{k}: {v}" for k, v in type_def.fields.items()])
            schema_parts.append(f"""
type {type_def.name} {{
  {fields}
}}
""")
        
        # Generate function definitions
        for func_def in self._functions.values():
            schema_parts.append(f"""
function {func_def.name}({func_def.input_type}) -> {func_def.output_type} {{
  prompt {{
    {func_def.prompt_template}
  }}
}}
""")
        
        return "\n".join(schema_parts)


# Global registry instance
_registry = SchemaRegistry()


def get_registry() -> SchemaRegistry:
    """Get the global schema registry."""
    return _registry


# ============================================
# BAML Client
# ============================================

@dataclass
class BAMLClient:
    """
    BAML-style type-safe LLM client.
    
    Provides type-safe function calls with schema validation.
    Even without the full BAML runtime, this provides similar
    benefits through Pydantic validation.
    
    Usage:
        client = BAMLClient()
        
        @client.function(
            input_type="str",
            output_type="SentimentResult",
        )
        def analyze_sentiment(text: str) -> SentimentResult:
            '''Analyze sentiment of text.'''
            pass
        
        result = analyze_sentiment("I love this!")
    """
    
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    registry: SchemaRegistry = field(default_factory=get_registry)
    _llm_client: Optional[Any] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the client."""
        self._setup_client()
        logger.info("baml_client_initialized", provider=self.provider)
    
    def _setup_client(self) -> None:
        """Setup LLM client for function execution."""
        try:
            if self.provider == "anthropic":
                from anthropic import Anthropic
                self._llm_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            elif self.provider == "openai":
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError as e:
            logger.warning("llm_client_import_failed", error=str(e))
    
    def _call_llm(self, prompt: str, system: str = "") -> str:
        """Call the LLM and get response."""
        if self._llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        if self.provider == "anthropic":
            response = self._llm_client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        else:  # openai
            response = self._llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
    
    def function(
        self,
        input_type: str,
        output_type: Type[T],
        prompt_template: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to create a type-safe BAML-style function.
        
        Args:
            input_type: Description of input type
            output_type: Pydantic model for output
            prompt_template: Optional custom prompt template
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Build prompt from function args
                func_name = func.__name__
                doc = func.__doc__ or f"Execute {func_name}"
                
                # Create prompt
                if prompt_template:
                    prompt = prompt_template.format(*args, **kwargs)
                else:
                    prompt = f"""{doc}

Input: {json.dumps(args[0] if args else kwargs, indent=2)}

Respond with valid JSON matching this schema:
{json.dumps(output_type.model_json_schema(), indent=2)}"""
                
                # Call LLM
                response = self._call_llm(
                    prompt=prompt,
                    system=f"You are executing the function '{func_name}'. Always respond with valid JSON.",
                )
                
                # Parse and validate response
                try:
                    # Try to extract JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        data = json.loads(json_str)
                        return output_type(**data)
                    else:
                        raise ValueError("No JSON found in response")
                except Exception as e:
                    logger.error("baml_function_parse_error", func=func_name, error=str(e))
                    raise
            
            # Register function
            self.registry.register_function(FunctionDef(
                name=func.__name__,
                description=func.__doc__ or "",
                input_type=input_type,
                output_type=output_type.__name__,
                prompt_template=prompt_template or "",
            ))
            
            return wrapper
        return decorator
    
    def define_type(self, model_class: Type[BaseModel]) -> Type[BaseModel]:
        """
        Register a Pydantic model as a BAML type.
        
        Args:
            model_class: Pydantic model to register
            
        Returns:
            The same model class (for decorator use)
        """
        schema = model_class.model_json_schema()
        
        fields = {}
        for field_name, field_info in schema.get("properties", {}).items():
            field_type = field_info.get("type", "any")
            description = field_info.get("description", "")
            fields[field_name] = f"{field_type}  # {description}" if description else field_type
        
        self.registry.register_type(TypeDef(
            name=model_class.__name__,
            description=schema.get("description", ""),
            fields=fields,
        ))
        
        return model_class


# ============================================
# Pre-built BAML Functions
# ============================================

# Sample output types
class BAMLSentiment(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(..., description="positive, negative, or neutral")
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score")
    explanation: str = Field(..., description="Explanation")


class BAMLClassification(BaseModel):
    """Classification result."""
    category: str = Field(..., description="Classification category")
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list)


class BAMLExtraction(BaseModel):
    """Data extraction result."""
    extracted: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)


class BAMLSummary(BaseModel):
    """Summary result."""
    summary: str
    topics: list[str] = Field(default_factory=list)
    word_count: int


def create_baml_functions() -> BAMLClient:
    """Create a BAML client with pre-registered functions."""
    client = BAMLClient()
    
    # Register types
    client.define_type(BAMLSentiment)
    client.define_type(BAMLClassification)
    client.define_type(BAMLExtraction)
    client.define_type(BAMLSummary)
    
    return client


if __name__ == "__main__":
    print("BAML Functions Module")
    print("-" * 40)
    print("Available features:")
    print("  - BAMLClient: Type-safe LLM function client")
    print("  - SchemaRegistry: Registry for types and functions")
    print("  - @client.function decorator: Create type-safe functions")
    print("\nExample usage:")
    print("  client = BAMLClient()")
    print("  @client.function(input_type='str', output_type=BAMLSentiment)")
    print("  def analyze(text: str) -> BAMLSentiment: pass")
```

---

## Step 5: Create Outlines Constrained Generation

Create `core/structured/outlines_constraints.py`:

```python
#!/usr/bin/env python3
"""
Outlines Constrained Generation
Grammar and regex-guided text generation for structured outputs.
"""

from __future__ import annotations

import os
import re
import json
from typing import Any, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import Outlines
try:
    import outlines
    from outlines import models, generate
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    logger.warning("outlines not available - install with: pip install outlines")


T = TypeVar("T", bound=BaseModel)


# ============================================
# Constraint Types
# ============================================

class ConstraintType(str, Enum):
    """Types of generation constraints."""
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    GRAMMAR = "grammar"
    CHOICE = "choice"
    INTEGER = "integer"
    FLOAT = "float"


class Constraint(BaseModel):
    """Generation constraint specification."""
    type: ConstraintType
    value: Any  # Regex pattern, schema, grammar, or choices
    description: str = ""


# ============================================
# Common Regex Patterns
# ============================================

class RegexPatterns:
    """Common regex patterns for constrained generation."""
    
    # Basic types
    INTEGER = r"-?\d+"
    POSITIVE_INTEGER = r"\d+"
    FLOAT = r"-?\d+\.?\d*"
    POSITIVE_FLOAT = r"\d+\.?\d*"
    
    # Identifiers
    EMAIL = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    URL = r"https?://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=-]*)?"
    PHONE_US = r"\d{3}-\d{3}-\d{4}"
    UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    
    # Dates and times
    DATE_ISO = r"\d{4}-\d{2}-\d{2}"
    TIME_24H = r"\d{2}:\d{2}(?::\d{2})?"
    DATETIME_ISO = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
    
    # Codes
    ZIP_US = r"\d{5}(?:-\d{4})?"
    SSN = r"\d{3}-\d{2}-\d{4}"
    CREDIT_CARD = r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}"
    
    # Text patterns
    WORD = r"[a-zA-Z]+"
    SENTENCE = r"[A-Z][^.!?]*[.!?]"
    SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
    
    # Structured
    JSON_OBJECT = r"\{[^}]*\}"
    JSON_ARRAY = r"\[[^\]]*\]"
    BOOLEAN = r"true|false"
    YES_NO = r"yes|no"


# ============================================
# Grammar Definitions
# ============================================

class GrammarTemplates:
    """Common grammar templates for constrained generation."""
    
    SENTIMENT = """
    root ::= sentiment
    sentiment ::= "positive" | "negative" | "neutral"
    """
    
    RATING = """
    root ::= rating
    rating ::= "1" | "2" | "3" | "4" | "5"
    """
    
    YESNO = """
    root ::= answer
    answer ::= "yes" | "no" | "maybe"
    """
    
    JSON_SIMPLE = """
    root ::= object
    object ::= "{" ws members? ws "}"
    members ::= pair ("," ws pair)*
    pair ::= string ":" ws value
    value ::= string | number | "true" | "false" | "null" | object | array
    array ::= "[" ws elements? ws "]"
    elements ::= value ("," ws value)*
    string ::= '"' [^"\\]* '"'
    number ::= "-"? [0-9]+ ("." [0-9]+)?
    ws ::= [ \\t\\n]*
    """


# ============================================
# Outlines Generator
# ============================================

@dataclass
class OutlinesGenerator:
    """
    Outlines-based constrained text generator.
    
    Provides grammar and regex-guided generation for structured outputs.
    
    Usage:
        generator = OutlinesGenerator()
        result = generator.generate_regex(
            prompt="What is 2+2?",
            pattern=r"\\d+",
        )
    """
    
    model_name: str = "mistralai/Mistral-7B-v0.1"  # Default local model
    device: str = "auto"
    _model: Optional[Any] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the generator."""
        if not OUTLINES_AVAILABLE:
            raise ImportError("outlines not installed - pip install outlines")
        
        logger.info("outlines_generator_initialized", model=self.model_name)
    
    def _get_model(self) -> Any:
        """Lazy load the model."""
        if self._model is None:
            logger.info("loading_model", model=self.model_name)
            try:
                self._model = models.transformers(self.model_name, device=self.device)
            except Exception as e:
                logger.warning("model_load_failed", error=str(e))
                # Fall back to a simple mock for environments without GPU
                self._model = None
        return self._model
    
    def generate_regex(
        self,
        prompt: str,
        pattern: str,
        max_tokens: int = 100,
    ) -> str:
        """
        Generate text constrained by a regex pattern.
        
        Args:
            prompt: Input prompt
            pattern: Regex pattern to match
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text matching the pattern
        """
        model = self._get_model()
        
        if model is None:
            # Fallback for environments without model
            return self._fallback_regex(prompt, pattern)
        
        try:
            generator = generate.regex(model, pattern)
            result = generator(prompt, max_tokens=max_tokens)
            
            logger.info("regex_generation_success", pattern=pattern[:50])
            return result
            
        except Exception as e:
            logger.error("regex_generation_failed", error=str(e))
            return self._fallback_regex(prompt, pattern)
    
    def _fallback_regex(self, prompt: str, pattern: str) -> str:
        """Fallback regex generation using LLM."""
        try:
            from core.llm_gateway import LLMGateway, Message
            
            gateway = LLMGateway()
            response = gateway.complete_sync(
                messages=[
                    Message(role="system", content=f"Respond with text that matches this regex: {pattern}"),
                    Message(role="user", content=prompt),
                ]
            )
            
            # Validate against pattern
            match = re.search(pattern, response.content)
            if match:
                return match.group(0)
            return response.content
            
        except Exception as e:
            logger.error("fallback_regex_failed", error=str(e))
            raise
    
    def generate_json(
        self,
        prompt: str,
        schema: Union[Type[BaseModel], dict[str, Any]],
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        """
        Generate JSON constrained by a schema.
        
        Args:
            prompt: Input prompt
            schema: Pydantic model or JSON schema dict
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated JSON matching the schema
        """
        model = self._get_model()
        
        # Get schema dict
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
        else:
            schema_dict = schema
        
        if model is None:
            return self._fallback_json(prompt, schema_dict)
        
        try:
            generator = generate.json(model, schema_dict)
            result = generator(prompt, max_tokens=max_tokens)
            
            logger.info("json_generation_success")
            return result
            
        except Exception as e:
            logger.error("json_generation_failed", error=str(e))
            return self._fallback_json(prompt, schema_dict)
    
    def _fallback_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Fallback JSON generation using LLM."""
        try:
            from core.llm_gateway import LLMGateway, Message
            
            gateway = LLMGateway()
            response = gateway.complete_sync(
                messages=[
                    Message(
                        role="system",
                        content=f"Respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
                    ),
                    Message(role="user", content=prompt),
                ]
            )
            
            # Extract and parse JSON
            json_start = response.content.find("{")
            json_end = response.content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response.content[json_start:json_end])
            return json.loads(response.content)
            
        except Exception as e:
            logger.error("fallback_json_failed", error=str(e))
            raise
    
    def generate_choice(
        self,
        prompt: str,
        choices: list[str],
    ) -> str:
        """
        Generate text constrained to specific choices.
        
        Args:
            prompt: Input prompt
            choices: List of valid choices
            
        Returns:
            One of the provided choices
        """
        model = self._get_model()
        
        if model is None:
            return self._fallback_choice(prompt, choices)
        
        try:
            generator = generate.choice(model, choices)
            result = generator(prompt)
            
            logger.info("choice_generation_success", choice=result)
            return result
            
        except Exception as e:
            logger.error("choice_generation_failed", error=str(e))
            return self._fallback_choice(prompt, choices)
    
    def _fallback_choice(self, prompt: str, choices: list[str]) -> str:
        """Fallback choice generation using LLM."""
        try:
            from core.llm_gateway import LLMGateway, Message
            
            gateway = LLMGateway()
            response = gateway.complete_sync(
                messages=[
                    Message(
                        role="system",
                        content=f"Respond with exactly one of these options: {', '.join(choices)}"
                    ),
                    Message(role="user", content=prompt),
                ]
            )
            
            # Find matching choice
            content = response.content.strip().lower()
            for choice in choices:
                if choice.lower() in content:
                    return choice
            return choices[0]  # Default to first choice
            
        except Exception as e:
            logger.error("fallback_choice_failed", error=str(e))
            raise
    
    def generate_grammar(
        self,
        prompt: str,
        grammar: str,
        max_tokens: int = 200,
    ) -> str:
        """
        Generate text constrained by a grammar.
        
        Args:
            prompt: Input prompt
            grammar: Grammar specification (EBNF-like)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text matching the grammar
        """
        model = self._get_model()
        
        if model is None:
            # Grammar generation requires the model
            raise RuntimeError("Grammar generation requires a local model")
        
        try:
            generator = generate.cfg(model, grammar)
            result = generator(prompt, max_tokens=max_tokens)
            
            logger.info("grammar_generation_success")
            return result
            
        except Exception as e:
            logger.error("grammar_generation_failed", error=str(e))
            raise
    
    def generate_structured(
        self,
        prompt: str,
        model_class: Type[T],
        max_tokens: int = 500,
    ) -> T:
        """
        Generate structured output as a Pydantic model.
        
        Args:
            prompt: Input prompt
            model_class: Pydantic model class
            max_tokens: Maximum tokens to generate
            
        Returns:
            Validated Pydantic model instance
        """
        result_dict = self.generate_json(
            prompt=prompt,
            schema=model_class,
            max_tokens=max_tokens,
        )
        
        return model_class(**result_dict)


# ============================================
# Convenience Functions
# ============================================

def constrain_to_pattern(
    prompt: str,
    pattern: str,
    model_name: Optional[str] = None,
) -> str:
    """Generate text constrained to a regex pattern."""
    generator = OutlinesGenerator(model_name=model_name or "mistralai/Mistral-7B-v0.1")
    return generator.generate_regex(prompt, pattern)


def constrain_to_choices(
    prompt: str,
    choices: list[str],
    model_name: Optional[str] = None,
) -> str:
    """Generate text constrained to specific choices."""
    generator = OutlinesGenerator(model_name=model_name or "mistralai/Mistral-7B-v0.1")
    return generator.generate_choice(prompt, choices)


def constrain_to_schema(
    prompt: str,
    schema: Type[BaseModel],
    model_name: Optional[str] = None,
) -> BaseModel:
    """Generate JSON constrained to a Pydantic schema."""
    generator = OutlinesGenerator(model_name=model_name or "mistralai/Mistral-7B-v0.1")
    return generator.generate_structured(prompt, schema)


if __name__ == "__main__":
    print("Outlines Constraints Module")
    print("-" * 40)
    
    if not OUTLINES_AVAILABLE:
        print("Outlines not installed. Install with: pip install outlines")
    else:
        print("Available features:")
        print("  - OutlinesGenerator: Main constrained generation client")
        print("  - RegexPatterns: Common regex patterns")
        print("  - GrammarTemplates: Common grammar templates")
        print("\nExample usage:")
        print("  generator = OutlinesGenerator()")
        print("  result = generator.generate_choice('Is this positive?', ['yes', 'no'])")
```

---

## Step 6: Create PydanticAI Agent Wrapper

Create `core/structured/pydantic_agents.py`:

```python
#!/usr/bin/env python3
"""
PydanticAI Agent Wrapper
Pydantic-native AI agent framework with memory and orchestration integration.
"""

from __future__ import annotations

import os
import asyncio
from typing import Any, Optional, Type, TypeVar, Callable, Generic
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import PydanticAI
try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIModel
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    logger.warning("pydantic-ai not available - install with: pip install pydantic-ai")


T = TypeVar("T", bound=BaseModel)


# ============================================
# State and Context Models
# ============================================

class AgentState(BaseModel):
    """State maintained by the agent."""
    conversation_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentConfig(BaseModel):
    """Configuration for a PydanticAI agent."""
    name: str
    system_prompt: str
    model: str = "claude-3-5-sonnet-20241022"
    provider: str = "anthropic"  # anthropic or openai
    temperature: float = 0.7
    max_tokens: int = 4096
    tools_enabled: bool = True
    memory_enabled: bool = True


class ToolDefinition(BaseModel):
    """Definition of an agent tool."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Result from agent execution."""
    output: Any
    state: AgentState
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    execution_time_ms: int = 0


# ============================================
# PydanticAI Agent Wrapper
# ============================================

@dataclass
class PydanticAIAgent:
    """
    PydanticAI-based agent with memory and orchestration integration.
    
    Provides a Pydantic-native agent framework that integrates with
    the Unleash memory layer (Phase 4) and orchestration layer (Phase 3).
    
    Usage:
        agent = PydanticAIAgent(
            config=AgentConfig(
                name="assistant",
                system_prompt="You are a helpful assistant.",
            )
        )
        result = await agent.run("Hello!")
    """
    
    config: AgentConfig
    state: AgentState = field(default_factory=AgentState)
    _agent: Optional[Any] = field(default=None, init=False)
    _tools: dict[str, Callable] = field(default_factory=dict, init=False)
    _memory_provider: Optional[Any] = field(default=None, init=False)
    
    def __post_init__(self) -> None:
        """Initialize the agent."""
        if not PYDANTIC_AI_AVAILABLE:
            raise ImportError("pydantic-ai not installed - pip install pydantic-ai")
        
        self._setup_agent()
        self._setup_memory()
        
        logger.info(
            "pydantic_agent_initialized",
            name=self.config.name,
            model=self.config.model,
        )
    
    def _setup_agent(self) -> None:
        """Setup the PydanticAI agent."""
        # Create model instance
        if self.config.provider == "anthropic":
            model = AnthropicModel(
                self.config.model,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
        else:  # openai
            model = OpenAIModel(
                self.config.model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        
        # Create agent
        self._agent = Agent(
            model,
            system_prompt=self.config.system_prompt,
        )
    
    def _setup_memory(self) -> None:
        """Setup memory integration from Phase 4."""
        if not self.config.memory_enabled:
            return
        
        try:
            from core.memory import MemoryProviderFactory
            
            # Try to get a memory provider
            self._memory_provider = MemoryProviderFactory.create("letta")
            logger.info("memory_provider_connected", provider="letta")
            
        except ImportError:
            logger.warning("memory_layer_not_available")
        except Exception as e:
            logger.warning("memory_provider_failed", error=str(e))
    
    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a tool for the agent to use.
        
        Args:
            func: The function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
        """
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        self._tools[tool_name] = func
        
        # Register with PydanticAI agent
        if self._agent and self.config.tools_enabled:
            self._agent.tool(func)
        
        logger.info("tool_registered", name=tool_name)
    
    async def run(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Run the agent with a message.
        
        Args:
            message: User message
            context: Optional additional context
            
        Returns:
            Agent execution result
        """
        start_time = datetime.utcnow()
        
        # Update state
        self.state.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        if context:
            self.state.context.update(context)
        
        # Load memory context if available
        memory_context = await self._load_memory_context(message)
        
        # Build full context
        full_context = {
            **self.state.context,
            "memory": memory_context,
        }
        
        try:
            # Run the agent
            result = await self._agent.run(
                message,
                deps=full_context,
            )
            
            # Extract output
            output = result.data
            
            # Record assistant message
            self.state.messages.append({
                "role": "assistant",
                "content": str(output),
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            # Save to memory if enabled
            await self._save_to_memory(message, str(output))
            
            # Calculate execution time
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            self.state.updated_at = datetime.utcnow().isoformat()
            
            logger.info(
                "agent_run_success",
                name=self.config.name,
                execution_time_ms=execution_time,
            )
            
            return AgentResult(
                output=output,
                state=self.state,
                tool_calls=self.state.tool_calls,
                execution_time_ms=execution_time,
            )
            
        except Exception as e:
            logger.error("agent_run_failed", name=self.config.name, error=str(e))
            raise
    
    async def _load_memory_context(self, query: str) -> dict[str, Any]:
        """Load relevant context from memory."""
        if not self._memory_provider:
            return {}
        
        try:
            # Search memory for relevant context
            memories = await self._memory_provider.search(query, limit=5)
            return {"relevant_memories": memories}
        except Exception as e:
            logger.warning("memory_load_failed", error=str(e))
            return {}
    
    async def _save_to_memory(self, query: str, response: str) -> None:
        """Save interaction to memory."""
        if not self._memory_provider:
            return
        
        try:
            await self._memory_provider.add(
                content=f"Q: {query}\nA: {response}",
                metadata={"conversation_id": self.state.conversation_id},
            )
        except Exception as e:
            logger.warning("memory_save_failed", error=str(e))
    
    def run_sync(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(message, context))
    
    async def with_orchestration(
        self,
        message: str,
        orchestrator_name: str = "claude_flow",
    ) -> AgentResult:
        """
        Run agent with orchestration from Phase 3.
        
        Args:
            message: User message
            orchestrator_name: Which orchestrator to use
            
        Returns:
            Agent result
        """
        try:
            from core.orchestration import UnifiedOrchestrator, OrchestrationFramework
            
            orchestrator = UnifiedOrchestrator()
            
            # Map name to framework
            framework_map = {
                "claude_flow": OrchestrationFramework.CLAUDE_FLOW,
                "langgraph": OrchestrationFramework.LANGGRAPH,
                "crewai": OrchestrationFramework.CREWAI,
                "autogen": OrchestrationFramework.AUTOGEN,
            }
            
            framework = framework_map.get(orchestrator_name, OrchestrationFramework.CLAUDE_FLOW)
            orch = orchestrator.get_orchestrator(framework)
            
            # Use orchestration for complex workflows
            # This is a simplified integration
            result = await self.run(message, context={"orchestrator": orchestrator_name})
            
            return result
            
        except ImportError:
            logger.warning("orchestration_layer_not_available")
            return await self.run(message)
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def reset_state(self) -> None:
        """Reset agent state."""
        self.state = AgentState()
        logger.info("agent_state_reset", name=self.config.name)


# ============================================
# Pre-built Agent Templates
# ============================================

def create_assistant_agent(
    name: str = "assistant",
    system_prompt: Optional[str] = None,
) -> PydanticAIAgent:
    """Create a general-purpose assistant agent."""
    return PydanticAIAgent(
        config=AgentConfig(
            name=name,
            system_prompt=system_prompt or """You are a helpful AI assistant. 
You provide clear, accurate, and helpful responses.
You can use tools when needed to accomplish tasks.""",
        )
    )


def create_researcher_agent(name: str = "researcher") -> PydanticAIAgent:
    """Create a research-focused agent."""
    return PydanticAIAgent(
        config=AgentConfig(
            name=name,
            system_prompt="""You are a research specialist AI.
Your job is to thoroughly research topics, synthesize information,
and provide well-sourced, factual responses.
Always cite your reasoning and note any uncertainties.""",
        )
    )


def create_coder_agent(name: str = "coder") -> PydanticAIAgent:
    """Create a coding-focused agent."""
    return PydanticAIAgent(
        config=AgentConfig(
            name=name,
            system_prompt="""You are an expert programmer AI.
You write clean, efficient, well-documented code.
You follow best practices and explain your implementations.
You can work with multiple programming languages.""",
        )
    )


def create_analyst_agent(name: str = "analyst") -> PydanticAIAgent:
    """Create a data analysis agent."""
    return PydanticAIAgent(
        config=AgentConfig(
            name=name,
            system_prompt="""You are a data analyst AI.
You analyze data, identify patterns, and provide insights.
You present findings clearly with supporting evidence.
You can suggest visualizations and statistical analyses.""",
        )
    )


# ============================================
# Multi-Agent Coordination
# ============================================

class AgentTeam:
    """Coordinate multiple PydanticAI agents."""
    
    def __init__(self, name: str = "team"):
        self.name = name
        self.agents: dict[str, PydanticAIAgent] = {}
        self.history: list[dict[str, Any]] = []
    
    def add_agent(self, agent: PydanticAIAgent) -> None:
        """Add an agent to the team."""
        self.agents[agent.config.name] = agent
        logger.info("agent_added_to_team", team=self.name, agent=agent.config.name)
    
    async def delegate(
        self,
        message: str,
        agent_name: str,
        context: Optional[dict[str, Any]] = None,
    ) -> AgentResult:
        """Delegate a task to a specific agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not in team")
        
        agent = self.agents[agent_name]
        result = await agent.run(message, context)
        
        self.history.append({
            "agent": agent_name,
            "message": message,
            "output": result.output,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return result
    
    async def round_robin(
        self,
        message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[AgentResult]:
        """Run message through all agents."""
        results = []
        
        for agent_name, agent in self.agents.items():
            result = await agent.run(message, context)
            results.append(result)
            
            # Add previous result to context for next agent
            context = context or {}
            context[f"response_from_{agent_name}"] = result.output
        
        return results
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get team interaction history."""
        return self.history


if __name__ == "__main__":
    print("PydanticAI Agents Module")
    print("-" * 40)
    
    if not PYDANTIC_AI_AVAILABLE:
        print("PydanticAI not installed. Install with: pip install pydantic-ai")
    else:
        print("Available features:")
        print("  - PydanticAIAgent: Main agent class")
        print("  - AgentTeam: Multi-agent coordination")
        print("  - create_assistant_agent(): General assistant")
        print("  - create_researcher_agent(): Research specialist")
        print("  - create_coder_agent(): Programming expert")
        print("  - create_analyst_agent(): Data analyst")
        print("\nExample usage:")
        print("  agent = create_assistant_agent()")
        print("  result = await agent.run('Hello!')")
```

---

## Step 7: Create Unified Interface

Create `core/structured/__init__.py`:

```python
#!/usr/bin/env python3
"""
Unified Structured Output Interface
Layer 3 - Structured Output across 4 frameworks.
"""

from __future__ import annotations

from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ============================================
# Framework Availability
# ============================================

class StructuredFramework(str, Enum):
    """Available structured output frameworks."""
    INSTRUCTOR = "instructor"
    BAML = "baml"
    OUTLINES = "outlines"
    PYDANTIC_AI = "pydantic_ai"


def get_available_frameworks() -> dict[str, bool]:
    """Check which frameworks are available."""
    availability = {}
    
    try:
        import instructor
        availability["instructor"] = True
    except ImportError:
        availability["instructor"] = False
    
    try:
        from baml_py import BAML
        availability["baml"] = True
    except ImportError:
        availability["baml"] = False
    
    try:
        import outlines
        availability["outlines"] = True
    except ImportError:
        availability["outlines"] = False
    
    try:
        from pydantic_ai import Agent
        availability["pydantic_ai"] = True
    except ImportError:
        availability["pydantic_ai"] = False
    
    return availability


# ============================================
# Import Framework Components
# ============================================

# Instructor
try:
    from core.structured.instructor_chains import (
        InstructorClient,
        ClassificationResult,
        Entity,
        ExtractionResult,
        SentimentResult,
        SummaryResult,
        QuestionAnswer,
        CodeGenerationResult,
        DataExtractionResult,
        quick_extract,
        analyze_text,
        INSTRUCTOR_AVAILABLE,
    )
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    InstructorClient = None
    ClassificationResult = None
    Entity = None
    ExtractionResult = None
    SentimentResult = None
    SummaryResult = None
    QuestionAnswer = None
    CodeGenerationResult = None
    DataExtractionResult = None
    quick_extract = None
    analyze_text = None

# BAML
try:
    from core.structured.baml_functions import (
        BAMLClient,
        SchemaRegistry,
        TypeDef,
        FunctionDef,
        BAMLSentiment,
        BAMLClassification,
        BAMLExtraction,
        BAMLSummary,
        create_baml_functions,
        get_registry,
        BAML_AVAILABLE,
    )
except ImportError:
    BAML_AVAILABLE = False
    BAMLClient = None
    SchemaRegistry = None
    TypeDef = None
    FunctionDef = None
    BAMLSentiment = None
    BAMLClassification = None
    BAMLExtraction = None
    BAMLSummary = None
    create_baml_functions = None
    get_registry = None

# Outlines
try:
    from core.structured.outlines_constraints import (
        OutlinesGenerator,
        Constraint,
        ConstraintType,
        RegexPatterns,
        GrammarTemplates,
        constrain_to_pattern,
        constrain_to_choices,
        constrain_to_schema,
        OUTLINES_AVAILABLE,
    )
except ImportError:
    OUTLINES_AVAILABLE = False
    OutlinesGenerator = None
    Constraint = None
    ConstraintType = None
    RegexPatterns = None
    GrammarTemplates = None
    constrain_to_pattern = None
    constrain_to_choices = None
    constrain_to_schema = None

# PydanticAI
try:
    from core.structured.pydantic_agents import (
        PydanticAIAgent,
        AgentState,
        AgentConfig,
        AgentResult,
        AgentTeam,
        create_assistant_agent,
        create_researcher_agent,
        create_coder_agent,
        create_analyst_agent,
        PYDANTIC_AI_AVAILABLE,
    )
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    PydanticAIAgent = None
    AgentState = None
    AgentConfig = None
    AgentResult = None
    AgentTeam = None
    create_assistant_agent = None
    create_researcher_agent = None
    create_coder_agent = None
    create_analyst_agent = None


# ============================================
# Unified Structured Output Interface
# ============================================

@dataclass
class StructuredOutputFactory:
    """
    Factory for creating structured output clients.
    
    Provides a unified interface to work with any supported framework.
    """
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all structured output frameworks."""
        return {
            "available_frameworks": get_available_frameworks(),
            "instructor": {
                "available": INSTRUCTOR_AVAILABLE,
                "features": ["pydantic validation", "automatic retries", "streaming"],
            },
            "baml": {
                "available": BAML_AVAILABLE,
                "features": ["type-safe functions", "schema registry", "validation"],
            },
            "outlines": {
                "available": OUTLINES_AVAILABLE,
                "features": ["regex constraints", "grammar generation", "json schema"],
            },
            "pydantic_ai": {
                "available": PYDANTIC_AI_AVAILABLE,
                "features": ["native agents", "tool use", "memory integration"],
            },
        }
    
    def get_client(self, framework: StructuredFramework, **kwargs) -> Any:
        """
        Get a client for the specified framework.
        
        Args:
            framework: Which framework to use
            **kwargs: Framework-specific arguments
        
        Returns:
            Framework-specific client instance
        """
        if framework == StructuredFramework.INSTRUCTOR:
            if not INSTRUCTOR_AVAILABLE:
                raise ImportError("instructor not installed")
            return InstructorClient(**kwargs)
        
        elif framework == StructuredFramework.BAML:
            if not BAML_AVAILABLE:
                raise ImportError("baml-py not installed")
            return BAMLClient(**kwargs)
        
        elif framework == StructuredFramework.OUTLINES:
            if not OUTLINES_AVAILABLE:
                raise ImportError("outlines not installed")
            return OutlinesGenerator(**kwargs)
        
        elif framework == StructuredFramework.PYDANTIC_AI:
            if not PYDANTIC_AI_AVAILABLE:
                raise ImportError("pydantic-ai not installed")
            config = AgentConfig(
                name=kwargs.get("name", "agent"),
                system_prompt=kwargs.get("system_prompt", "You are a helpful assistant."),
                **{k: v for k, v in kwargs.items() if k not in ["name", "system_prompt"]}
            )
            return PydanticAIAgent(config=config)
        
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def recommend_framework(self, use_case: str) -> StructuredFramework:
        """
        Recommend a framework based on use case.
        
        Args:
            use_case: Description of the use case
        
        Returns:
            Recommended framework
        """
        use_case_lower = use_case.lower()
        
        # Check for keywords
        if any(kw in use_case_lower for kw in ["validate", "pydantic", "extract", "structured"]):
            if INSTRUCTOR_AVAILABLE:
                return StructuredFramework.INSTRUCTOR
        
        if any(kw in use_case_lower for kw in ["type-safe", "function", "schema"]):
            if BAML_AVAILABLE:
                return StructuredFramework.BAML
        
        if any(kw in use_case_lower for kw in ["regex", "grammar", "constrain", "choice"]):
            if OUTLINES_AVAILABLE:
                return StructuredFramework.OUTLINES
        
        if any(kw in use_case_lower for kw in ["agent", "memory", "tool", "conversation"]):
            if PYDANTIC_AI_AVAILABLE:
                return StructuredFramework.PYDANTIC_AI
        
        # Default to instructor if available (most general-purpose)
        if INSTRUCTOR_AVAILABLE:
            return StructuredFramework.INSTRUCTOR
        
        # Fallback
        available = get_available_frameworks()
        for fw, is_available in available.items():
            if is_available:
                return StructuredFramework(fw)
        
        raise RuntimeError("No structured output frameworks available")


# ============================================
# Module Exports
# ============================================

__all__ = [
    # Enums
    "StructuredFramework",
    
    # Availability checks
    "get_available_frameworks",
    "INSTRUCTOR_AVAILABLE",
    "BAML_AVAILABLE",
    "OUTLINES_AVAILABLE",
    "PYDANTIC_AI_AVAILABLE",
    
    # Factory
    "StructuredOutputFactory",
    
    # Instructor
    "InstructorClient",
    "ClassificationResult",
    "Entity",
    "ExtractionResult",
    "SentimentResult",
    "SummaryResult",
    "QuestionAnswer",
    "CodeGenerationResult",
    "DataExtractionResult",
    "quick_extract",
    "analyze_text",
    
    # BAML
    "BAMLClient",
    "SchemaRegistry",
    "TypeDef",
    "FunctionDef",
    "BAMLSentiment",
    "BAMLClassification",
    "BAMLExtraction",
    "BAMLSummary",
    "create_baml_functions",
    "get_registry",
    
    # Outlines
    "OutlinesGenerator",
    "Constraint",
    "ConstraintType",
    "RegexPatterns",
    "GrammarTemplates",
    "constrain_to_pattern",
    "constrain_to_choices",
    "constrain_to_schema",
    
    # PydanticAI
    "PydanticAIAgent",
    "AgentState",
    "AgentConfig",
    "AgentResult",
    "AgentTeam",
    "create_assistant_agent",
    "create_researcher_agent",
    "create_coder_agent",
    "create_analyst_agent",
]


if __name__ == "__main__":
    print("Structured Output Layer Status")
    print("=" * 50)
    
    factory = StructuredOutputFactory()
    status = factory.get_status()
    
    for framework, available in status["available_frameworks"].items():
        status_str = "✓" if available else "✗"
        info = status[framework]
        print(f"\n{status_str} {framework.upper()}")
        if available:
            print(f"  Features: {', '.join(info['features'])}")
        else:
            print(f"  Status: Not installed")
```

---

## Step 8: Create Validation Script

Create `scripts/validate_phase5.py`:

```python
#!/usr/bin/env python3
"""
Phase 5 Structured Output Layer Validation Script
Validates all Layer 3 structured output components.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_sdk_import(name: str, package: str) -> Tuple[bool, str]:
    """Check if an SDK can be imported."""
    try:
        __import__(package)
        return True, f"{name} importable"
    except ImportError as e:
        return False, f"{name} not installed (optional)"


def check_module_exists(module_path: str, description: str) -> Tuple[bool, str]:
    """Check if a module file exists."""
    full_path = project_root / module_path.replace(".", "/")
    py_path = full_path.with_suffix(".py")
    
    if py_path.exists():
        return True, f"{description} exists"
    elif full_path.is_dir() and (full_path / "__init__.py").exists():
        return True, f"{description} exists"
    return False, f"{description} not found"


def check_structured_imports() -> Tuple[bool, str]:
    """Check if structured module imports work."""
    try:
        from core.structured import (
            StructuredOutputFactory,
            StructuredFramework,
            get_available_frameworks,
        )
        return True, "Core imports work"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_framework_availability() -> dict:
    """Check which frameworks are available."""
    try:
        from core.structured import get_available_frameworks
        return get_available_frameworks()
    except:
        return {}


def check_instructor() -> Tuple[bool, str]:
    """Check Instructor integration."""
    try:
        from core.structured.instructor_chains import (
            InstructorClient,
            INSTRUCTOR_AVAILABLE,
        )
        if INSTRUCTOR_AVAILABLE:
            return True, "Instructor operational"
        return False, "Instructor SDK not installed"
    except ImportError as e:
        return False, f"Instructor error: {e}"


def check_outlines() -> Tuple[bool, str]:
    """Check Outlines integration."""
    try:
        from core.structured.outlines_constraints import (
            OutlinesGenerator,
            OUTLINES_AVAILABLE,
        )
        if OUTLINES_AVAILABLE:
            return True, "Outlines operational"
        return False, "Outlines SDK not installed"
    except ImportError as e:
        return False, f"Outlines error: {e}"


def check_pydantic_ai() -> Tuple[bool, str]:
    """Check PydanticAI integration."""
    try:
        from core.structured.pydantic_agents import (
            PydanticAIAgent,
            PYDANTIC_AI_AVAILABLE,
        )
        if PYDANTIC_AI_AVAILABLE:
            return True, "PydanticAI operational"
        return False, "PydanticAI SDK not installed"
    except ImportError as e:
        return False, f"PydanticAI error: {e}"


def main():
    """Run all Phase 5 validation checks."""
    # Dynamic import to handle missing rich gracefully
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
    except ImportError:
        print("rich not installed - using basic output")
        print("\nPhase 5: Structured Output Layer Validation")
        print("=" * 50)
        
        # Basic checks without rich
        checks_passed = True
        
        # Module checks
        modules = [
            "core/structured/__init__.py",
            "core/structured/instructor_chains.py",
            "core/structured/baml_functions.py",
            "core/structured/outlines_constraints.py",
            "core/structured/pydantic_agents.py",
        ]
        
        print("\nModule Checks:")
        for module in modules:
            exists = (project_root / module).exists()
            status = "OK" if exists else "MISSING"
            print(f"  {module}: {status}")
            if not exists:
                checks_passed = False
        
        # Import checks
        print("\nImport Checks:")
        passed, details = check_structured_imports()
        print(f"  Structured Module: {'OK' if passed else 'FAIL'} - {details}")
        if not passed:
            checks_passed = False
        
        if checks_passed:
            print("\n[PASS] Phase 5 Validation PASSED")
            return 0
        else:
            print("\n[FAIL] Phase 5 Validation FAILED")
            return 1

    # Force UTF-8 for Windows console compatibility
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    console = Console(force_terminal=True)
    console.print(Panel.fit(
        "[bold blue]Phase 5: Structured Output Layer Validation[/bold blue]\n"
        "[dim]Layer 3 - 4 Structured Output SDKs[/dim]",
        border_style="blue"
    ))
    console.print()

    # SDK Checks
    sdk_table = Table(title="SDK Availability", show_header=True, header_style="bold magenta")
    sdk_table.add_column("SDK", style="cyan")
    sdk_table.add_column("Status", style="green")
    sdk_table.add_column("Details")

    sdk_checks = [
        ("Instructor", "instructor"),
        ("BAML", "baml_py"),
        ("Outlines", "outlines"),
        ("PydanticAI", "pydantic_ai"),
    ]

    sdk_available_count = 0
    for name, package in sdk_checks:
        passed, details = check_sdk_import(name, package)
        status = "[green]PASS[/green]" if passed else "[yellow]SKIP[/yellow]"
        sdk_table.add_row(name, status, details)
        if passed:
            sdk_available_count += 1

    console.print(sdk_table)
    console.print()

    # Module Checks
    module_table = Table(title="Structured Modules", show_header=True, header_style="bold magenta")
    module_table.add_column("Module", style="cyan")
    module_table.add_column("Status", style="green")
    module_table.add_column("Details")

    module_checks = [
        ("core/structured/__init__.py", "Unified Interface"),
        ("core/structured/instructor_chains.py", "Instructor Chains"),
        ("core/structured/baml_functions.py", "BAML Functions"),
        ("core/structured/outlines_constraints.py", "Outlines Constraints"),
        ("core/structured/pydantic_agents.py", "PydanticAI Agents"),
    ]

    modules_passed = True
    for module, description in module_checks:
        passed, details = check_module_exists(module, description)
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        module_table.add_row(module, status, details)
        if not passed:
            modules_passed = False

    console.print(module_table)
    console.print()

    # Functional Checks
    func_table = Table(title="Functional Validation", show_header=True, header_style="bold magenta")
    func_table.add_column("Check", style="cyan")
    func_table.add_column("Status", style="green")
    func_table.add_column("Details")

    func_checks = [
        ("Structured Imports", check_structured_imports),
        ("Instructor", check_instructor),
        ("Outlines", check_outlines),
        ("PydanticAI", check_pydantic_ai),
    ]

    func_passed = True
    for name, check_fn in func_checks:
        passed, details = check_fn()
        # Yellow for optional SDK not installed, red for actual failures
        if "not installed" in details.lower():
            status = "[yellow]SKIP[/yellow]"
        elif passed:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"
            func_passed = False
        func_table.add_row(name, status, details)

    console.print(func_table)
    console.print()

    # Framework Availability Summary
    frameworks = check_framework_availability()
    if frameworks:
        fw_table = Table(title="Framework Status", show_header=True, header_style="bold cyan")
        fw_table.add_column("Framework", style="cyan")
        fw_table.add_column("Available", style="green")
        
        for fw, available in frameworks.items():
            status = "[green]Yes[/green]" if available else "[yellow]No[/yellow]"
            fw_table.add_row(fw, status)
        
        console.print(fw_table)
        console.print()

    # Summary
    all_required_passed = modules_passed and func_passed

    if all_required_passed:
        console.print(Panel.fit(
            "[bold green]Phase 5 Validation PASSED[/bold green]\n\n"
            f"Structured Output Layer (Layer 3) is operational.\n"
            f"- {sdk_available_count}/4 optional SDKs installed\n"
            f"- All module files created\n"
            f"- Core functionality verified\n\n"
            "[dim]Ready for Phase 6: Guardrails Layer[/dim]",
            border_style="green"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold red]Phase 5 Validation FAILED[/bold red]\n\n"
            "Some required checks did not pass.\n"
            "Please review the tables above and fix issues.",
            border_style="red"
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Success Criteria

After execution, verify:

- [ ] `core/structured/` directory exists
- [ ] `core/structured/__init__.py` - Unified interface
- [ ] `core/structured/instructor_chains.py` - Instructor integration
- [ ] `core/structured/baml_functions.py` - BAML functions
- [ ] `core/structured/outlines_constraints.py` - Outlines constraints
- [ ] `core/structured/pydantic_agents.py` - PydanticAI agents
- [ ] `scripts/validate_phase5.py` - Validation script
- [ ] `python scripts/validate_phase5.py` passes

---

## Rollback

If issues occur:

```bash
# Remove structured directory
rm -rf core/structured

# Remove validation script
rm scripts/validate_phase5.py
```

---

## Notes

- **Instructor** is the most general-purpose - recommended for most structured extraction
- **BAML** provides type-safe function definitions with schema validation
- **Outlines** excels at constrained generation with regexes and grammars
- **PydanticAI** integrates with Phase 4 memory and Phase 3 orchestration
- All frameworks support Pydantic models for type safety
- SDKs are optional - the layer works with available frameworks

---

## Installation Commands

```bash
# Install all Layer 3 SDKs
pip install instructor outlines pydantic-ai baml-py

# Or install individually as needed
pip install instructor      # Structured LLM outputs
pip install outlines        # Constrained generation
pip install pydantic-ai     # Pydantic-native agents
pip install baml-py         # Type-safe functions
```

---

## Integration with Previous Phases

The Structured Output Layer integrates with:

1. **Phase 2 (Protocol)**: Uses LLM Gateway for completions
2. **Phase 3 (Orchestration)**: PydanticAI agents can use orchestrators
3. **Phase 4 (Memory)**: PydanticAI agents connect to memory providers

---

**End of Phase 5 Prompt**
