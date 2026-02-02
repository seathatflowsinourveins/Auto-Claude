"""
DSPy Adapter for Unleashed Platform

DSPy (Stanford NLP) provides declarative prompt programming with automatic optimization.
Key features:
- Modular, composable pipelines
- Automatic prompt and weight optimization
- Production-ready with 1,400+ dependents

Repository: https://github.com/stanfordnlp/dspy
Stars: 31,600 | Version: 3.1.0 | License: MIT
"""

import os
import sys
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Check DSPy availability
DSPY_AVAILABLE = False
dspy = None

try:
    import dspy as _dspy
    dspy = _dspy
    DSPY_AVAILABLE = True
except ImportError:
    pass

# Register adapter status
from . import register_adapter
register_adapter("dspy", DSPY_AVAILABLE, "3.1.0" if DSPY_AVAILABLE else None)


@dataclass
class OptimizationResult:
    """Result from DSPy optimization."""
    optimized_program: Any
    metrics: Dict[str, float]
    iterations: int
    improvement: float
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CompilationResult:
    """Result from DSPy compilation."""
    compiled_module: Any
    signature: str
    num_examples: int
    optimization_time: float


class DSPyAdapter:
    """
    Adapter for DSPy prompt optimization framework.

    DSPy replaces brittle prompt strings with modular, declarative programs
    that can be automatically optimized for any LLM.
    """

    def __init__(
        self,
        model: str = "claude-3-opus",
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize DSPy adapter.

        Args:
            model: LLM model to use (default: claude-3-opus)
            api_key: API key for the model provider
            cache_dir: Directory for caching compiled programs
        """
        self._available = DSPY_AVAILABLE
        self.model_name = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.cache_dir = cache_dir
        self._lm = None
        self._programs: Dict[str, Any] = {}

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "available": self._available,
            "model": self.model_name,
            "configured": self._lm is not None,
            "programs_loaded": len(self._programs),
        }

    def _check_available(self):
        """Check if DSPy is available, raise error if not."""
        if not self._available:
            raise ImportError(
                "DSPy is not installed. Install with: pip install dspy-ai"
            )

    def configure(self, **kwargs):
        """Configure DSPy settings."""
        self._check_available()
        if "claude" in self.model_name.lower():
            self._lm = dspy.Claude(
                model=self.model_name,
                api_key=self.api_key,
                **kwargs
            )
        elif "gpt" in self.model_name.lower():
            self._lm = dspy.OpenAI(
                model=self.model_name,
                api_key=kwargs.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
                **kwargs
            )
        else:
            # Generic LM configuration
            self._lm = dspy.LM(
                model=self.model_name,
                **kwargs
            )

        dspy.configure(lm=self._lm)
        return self

    def create_signature(
        self,
        inputs: List[str],
        outputs: List[str],
        instructions: Optional[str] = None,
    ) -> str:
        """
        Create a DSPy signature string.

        Args:
            inputs: List of input field names
            outputs: List of output field names
            instructions: Optional task instructions

        Returns:
            DSPy signature string
        """
        input_str = ", ".join(inputs)
        output_str = ", ".join(outputs)

        if instructions:
            return f'"{instructions}" {input_str} -> {output_str}'
        return f"{input_str} -> {output_str}"

    def create_module(
        self,
        name: str,
        signature: str,
        module_type: str = "Predict",
    ) -> Any:
        """
        Create a DSPy module with the given signature.

        Args:
            name: Name for this module
            signature: DSPy signature string
            module_type: Type of module (Predict, ChainOfThought, etc.)

        Returns:
            DSPy module
        """
        module_classes = {
            "Predict": dspy.Predict,
            "ChainOfThought": dspy.ChainOfThought,
            "ChainOfThoughtWithHint": dspy.ChainOfThoughtWithHint,
            "ProgramOfThought": dspy.ProgramOfThought,
            "ReAct": dspy.ReAct,
            "MultiChainComparison": dspy.MultiChainComparison,
        }

        module_cls = module_classes.get(module_type, dspy.Predict)
        module = module_cls(signature)

        self._programs[name] = module
        return module

    def create_chain(
        self,
        steps: List[Dict[str, Any]],
        name: str = "chain",
    ) -> Any:
        """
        Create a chain of DSPy modules.

        Args:
            steps: List of step configurations
                   Each step: {"name": str, "signature": str, "type": str}
            name: Name for the chain

        Returns:
            DSPy Module chain
        """
        class ChainModule(dspy.Module):
            def __init__(self, steps_config):
                super().__init__()
                self.steps = []
                module_classes = {
                    "Predict": dspy.Predict,
                    "ChainOfThought": dspy.ChainOfThought,
                }
                for step in steps_config:
                    module_cls = module_classes.get(step.get("type", "Predict"), dspy.Predict)
                    module = module_cls(step["signature"])
                    setattr(self, step["name"], module)
                    self.steps.append(step["name"])

            def forward(self, **kwargs):
                result = kwargs
                for step_name in self.steps:
                    module = getattr(self, step_name)
                    result = module(**result)
                return result

        chain = ChainModule(steps)
        self._programs[name] = chain
        return chain

    async def optimize(
        self,
        program: Any,
        trainset: List[Any],
        metric: Callable,
        optimizer: str = "BootstrapFewShot",
        num_threads: int = 4,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
    ) -> OptimizationResult:
        """
        Optimize a DSPy program using the specified optimizer.

        Args:
            program: DSPy program/module to optimize
            trainset: Training examples
            metric: Evaluation metric function
            optimizer: Optimizer type (BootstrapFewShot, MIPRO, etc.)
            num_threads: Number of parallel threads
            max_bootstrapped_demos: Max bootstrapped demonstrations
            max_labeled_demos: Max labeled demonstrations

        Returns:
            OptimizationResult with optimized program and metrics
        """
        # DSPy optimizers (verified from official docs 2026-01-30)
        # BootstrapFewShot: Standard few-shot learning
        # BootstrapFewShotWithRandomSearch: Random search over demonstrations
        # KNNFewShot: K-nearest neighbors for demonstration selection
        # GEPA: Genetic Evolution with Prompt Adaptation (newest)
        # MIPro: Multi-hop In-context optimization
        optimizers = {
            "BootstrapFewShot": dspy.BootstrapFewShot,
            "BootstrapFewShotWithRandomSearch": dspy.BootstrapFewShotWithRandomSearch,
            "KNNFewShot": getattr(dspy, "KNNFewShot", dspy.BootstrapFewShot),
            "GEPA": getattr(dspy, "GEPA", dspy.BootstrapFewShot),
            "MIPRO": getattr(dspy, "MIPRO", getattr(dspy, "MIPROv2", dspy.BootstrapFewShot)),
            "Ensemble": getattr(dspy, "Ensemble", None),
        }

        optimizer_cls = optimizers.get(optimizer, dspy.BootstrapFewShot)

        if optimizer_cls is None:
            raise ValueError(f"Optimizer {optimizer} not available")

        teleprompter = optimizer_cls(
            metric=metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            num_threads=num_threads,
        )

        # Track optimization history
        history = []
        original_score = self._evaluate(program, trainset[:10], metric)

        # Compile/optimize
        optimized = teleprompter.compile(program, trainset=trainset)

        # Evaluate improvement
        optimized_score = self._evaluate(optimized, trainset[:10], metric)

        return OptimizationResult(
            optimized_program=optimized,
            metrics={
                "original_score": original_score,
                "optimized_score": optimized_score,
            },
            iterations=len(trainset),
            improvement=optimized_score - original_score,
            history=history,
        )

    def _evaluate(
        self,
        program: Any,
        examples: List[Any],
        metric: Callable,
    ) -> float:
        """Evaluate program on examples."""
        scores = []
        for example in examples:
            try:
                prediction = program(**example.inputs())
                score = metric(example, prediction)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def compile_for_inference(
        self,
        program: Any,
        examples: Optional[List[Any]] = None,
    ) -> CompilationResult:
        """
        Compile a program for fast inference.

        Args:
            program: DSPy program to compile
            examples: Optional examples for few-shot compilation

        Returns:
            CompilationResult with compiled module
        """
        import time
        start_time = time.time()

        # Get program signature
        signature = str(getattr(program, "signature", "unknown"))

        compiled = program
        num_examples = len(examples) if examples else 0

        if examples:
            # Use examples as demonstrations
            teleprompter = dspy.LabeledFewShot(k=min(len(examples), 4))
            compiled = teleprompter.compile(program, trainset=examples)

        compilation_time = time.time() - start_time

        return CompilationResult(
            compiled_module=compiled,
            signature=signature,
            num_examples=num_examples,
            optimization_time=compilation_time,
        )

    def create_rag_module(
        self,
        retriever: Any,
        signature: str = "context, question -> answer",
    ) -> Any:
        """
        Create a RAG module with DSPy.

        Args:
            retriever: Retriever function/object
            signature: Answer signature

        Returns:
            DSPy RAG module
        """
        class RAGModule(dspy.Module):
            def __init__(self, retrieve_fn, sig):
                super().__init__()
                self.retrieve = retrieve_fn
                self.generate = dspy.ChainOfThought(sig)

            def forward(self, question: str, k: int = 3):
                context = self.retrieve(question, k=k)
                return self.generate(context=context, question=question)

        return RAGModule(retriever, signature)

    def save_program(self, name: str, path: str):
        """Save a compiled program to disk."""
        if name not in self._programs:
            raise ValueError(f"Program '{name}' not found")

        program = self._programs[name]
        program.save(path)

    def load_program(self, name: str, path: str) -> Any:
        """Load a compiled program from disk."""
        # Create a dummy module to load into
        program = dspy.Module()
        program.load(path)
        self._programs[name] = program
        return program

    def get_program(self, name: str) -> Optional[Any]:
        """Get a registered program by name."""
        return self._programs.get(name)

    def list_programs(self) -> List[str]:
        """List all registered programs."""
        return list(self._programs.keys())


# Convenience functions for common patterns
def create_qa_module(question_field: str = "question") -> Any:
    """Create a simple Q&A module."""
    if not DSPY_AVAILABLE:
        return None
    return dspy.ChainOfThought(f"{question_field} -> answer")


def create_summarize_module() -> Any:
    """Create a summarization module."""
    if not DSPY_AVAILABLE:
        return None
    return dspy.Predict("document -> summary")


def create_classify_module(classes: List[str]) -> Any:
    """Create a classification module."""
    if not DSPY_AVAILABLE:
        return None
    classes_str = ", ".join(classes)
    return dspy.Predict(f'"Classify into one of: {classes_str}" text -> category')
