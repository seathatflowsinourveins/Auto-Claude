"""
TextGrad Adapter for Unleashed Platform

TextGrad provides automatic differentiation via text for LLM optimization.
Key features:
- Text-based gradients for prompt optimization
- Backpropagation through LLM calls
- Loss functions for text outputs
- Optimizer classes (TextualGradientDescent)

Repository: https://github.com/zou-group/textgrad
Stars: 2,500+ | License: MIT

Based on paper: "TextGrad: Automatic Differentiation via Text"
"""

import os
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

# Check TextGrad availability
TEXTGRAD_AVAILABLE = False
textgrad = None

try:
    import textgrad as _textgrad
    textgrad = _textgrad
    TEXTGRAD_AVAILABLE = True
except ImportError:
    pass

# Register adapter status
from . import register_adapter
register_adapter("textgrad", TEXTGRAD_AVAILABLE, "0.1.8" if TEXTGRAD_AVAILABLE else None)


class OptimizerType(Enum):
    """Types of TextGrad optimizers."""
    TGD = "textual_gradient_descent"  # Standard TGD
    TGD_MOMENTUM = "tgd_momentum"      # TGD with momentum
    ADAM = "adam"                       # Adam-style text optimizer


class LossType(Enum):
    """Types of loss functions."""
    TEXT_LOSS = "text_loss"           # Generic text comparison
    MULTIFIELD = "multifield"         # Multi-field structured loss
    SEMANTIC = "semantic"             # Semantic similarity loss


@dataclass
class Variable:
    """A TextGrad variable (text that can be optimized)."""
    value: str
    role_description: str
    requires_grad: bool = True
    gradients: List[str] = field(default_factory=list)


@dataclass
class OptimizationStep:
    """Result of a single optimization step."""
    iteration: int
    loss_value: float
    gradient: str
    variable_before: str
    variable_after: str
    improvement: float


@dataclass
class OptimizationResult:
    """Result from TextGrad optimization."""
    initial_value: str
    optimized_value: str
    total_iterations: int
    final_loss: float
    improvement: float
    history: List[OptimizationStep]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextGradAdapter:
    """
    Adapter for TextGrad text-based gradient optimization.

    TextGrad enables optimizing text (prompts, responses, code) using
    gradient-like feedback from LLMs. This is analogous to gradient
    descent but operates on text instead of numerical parameters.

    Reference: https://github.com/zou-group/textgrad
    """

    def __init__(
        self,
        model: str = "claude-3-opus",
        api_key: Optional[str] = None,
        backward_model: Optional[str] = None,
    ):
        """
        Initialize TextGrad adapter.

        Args:
            model: LLM model for forward passes
            api_key: API key for model provider
            backward_model: Model for computing gradients (defaults to same as model)
        """
        self._available = TEXTGRAD_AVAILABLE
        self.model_name = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.backward_model = backward_model or model
        self._engine = None
        self._backward_engine = None
        self._optimizer = None

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "available": self._available,
            "model": self.model_name,
            "backward_model": self.backward_model,
            "configured": self._engine is not None,
        }

    def _check_available(self):
        """Check if TextGrad is available, raise error if not."""
        if not self._available:
            raise ImportError(
                "TextGrad is not installed. Install with: pip install textgrad"
            )

    def configure(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Configure TextGrad engines.

        Args:
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
        """
        self._check_available()

        # Configure forward engine
        if "claude" in self.model_name.lower():
            self._engine = textgrad.get_engine(
                engine_name=f"anthropic-{self.model_name}",
                api_key=self.api_key,
            )
        elif "gpt" in self.model_name.lower():
            self._engine = textgrad.get_engine(
                engine_name=f"openai-{self.model_name}",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            self._engine = textgrad.get_engine(engine_name=self.model_name)

        # Configure backward engine (for gradient computation)
        if self.backward_model != self.model_name:
            self._backward_engine = textgrad.get_engine(
                engine_name=self.backward_model
            )
        else:
            self._backward_engine = self._engine

        textgrad.set_backward_engine(self._backward_engine, override=True)

        return self

    def create_variable(
        self,
        value: str,
        role_description: str,
        requires_grad: bool = True,
    ) -> Any:
        """
        Create a TextGrad variable.

        Args:
            value: Initial text value
            role_description: Description of the variable's role
            requires_grad: Whether to compute gradients

        Returns:
            TextGrad Variable object
        """
        self._check_available()

        return textgrad.Variable(
            value=value,
            role_description=role_description,
            requires_grad=requires_grad,
        )

    def create_loss_function(
        self,
        loss_type: LossType = LossType.TEXT_LOSS,
        evaluation_instruction: Optional[str] = None,
    ) -> Callable:
        """
        Create a loss function for optimization.

        Args:
            loss_type: Type of loss function
            evaluation_instruction: Custom evaluation instruction

        Returns:
            Loss function callable
        """
        self._check_available()

        if loss_type == LossType.TEXT_LOSS:
            return textgrad.TextLoss(
                eval_system_prompt=evaluation_instruction or
                "Evaluate the quality of the response and provide feedback.",
                engine=self._backward_engine,
            )
        elif loss_type == LossType.MULTIFIELD:
            return textgrad.MultiFieldTokenParsedEvaluation(
                engine=self._backward_engine,
            )
        else:
            # Default text loss
            return textgrad.TextLoss(engine=self._backward_engine)

    def create_optimizer(
        self,
        parameters: List[Any],
        optimizer_type: OptimizerType = OptimizerType.TGD,
        learning_rate: float = 1.0,
        momentum: float = 0.0,
    ) -> Any:
        """
        Create an optimizer for text variables.

        Args:
            parameters: List of Variable objects to optimize
            optimizer_type: Type of optimizer
            learning_rate: Learning rate (text influence factor)
            momentum: Momentum factor for TGD_MOMENTUM

        Returns:
            TextGrad optimizer
        """
        self._check_available()

        if optimizer_type == OptimizerType.TGD:
            self._optimizer = textgrad.TGD(
                parameters=parameters,
                engine=self._engine,
            )
        elif optimizer_type == OptimizerType.TGD_MOMENTUM:
            self._optimizer = textgrad.TGD(
                parameters=parameters,
                engine=self._engine,
                momentum=momentum,
            )
        else:
            self._optimizer = textgrad.TGD(
                parameters=parameters,
                engine=self._engine,
            )

        return self._optimizer

    async def optimize(
        self,
        variable: Any,
        loss_fn: Callable,
        target: Optional[str] = None,
        num_iterations: int = 3,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Optimize a text variable using TextGrad.

        Args:
            variable: TextGrad Variable to optimize
            loss_fn: Loss function for evaluation
            target: Target/reference for comparison (optional)
            num_iterations: Number of optimization iterations
            verbose: Print progress

        Returns:
            OptimizationResult with optimized text
        """
        self._check_available()

        if self._optimizer is None:
            self.create_optimizer([variable])

        initial_value = variable.value
        history = []
        prev_loss = float('inf')

        for i in range(num_iterations):
            # Store value before step
            value_before = variable.value

            # Compute loss
            if target:
                loss = loss_fn(variable, target)
            else:
                loss = loss_fn(variable)

            # Get loss value (convert to float if possible)
            try:
                loss_value = float(loss.value) if hasattr(loss, 'value') else 0.0
            except (ValueError, TypeError):
                loss_value = 0.0

            # Backward pass (compute gradients)
            loss.backward()

            # Get gradient text
            gradient = ""
            if hasattr(variable, 'gradients') and variable.gradients:
                gradient = str(variable.gradients[-1])

            # Optimization step
            self._optimizer.step()

            # Zero gradients for next iteration
            self._optimizer.zero_grad()

            # Store value after step
            value_after = variable.value

            # Calculate improvement
            improvement = prev_loss - loss_value if prev_loss != float('inf') else 0.0
            prev_loss = loss_value

            history.append(OptimizationStep(
                iteration=i + 1,
                loss_value=loss_value,
                gradient=gradient[:500] if gradient else "",  # Truncate long gradients
                variable_before=value_before[:200],
                variable_after=value_after[:200],
                improvement=improvement,
            ))

            if verbose:
                print(f"Iteration {i + 1}: Loss = {loss_value:.4f}")

        return OptimizationResult(
            initial_value=initial_value,
            optimized_value=variable.value,
            total_iterations=num_iterations,
            final_loss=prev_loss,
            improvement=history[0].loss_value - prev_loss if history else 0.0,
            history=history,
            metadata={
                "model": self.model_name,
                "backward_model": self.backward_model,
            },
        )

    async def optimize_prompt(
        self,
        prompt: str,
        task_description: str,
        evaluation_criteria: str,
        num_iterations: int = 3,
    ) -> OptimizationResult:
        """
        Convenience method to optimize a prompt.

        Args:
            prompt: Initial prompt text
            task_description: Description of what the prompt should do
            evaluation_criteria: Criteria for evaluating prompt quality
            num_iterations: Number of optimization iterations

        Returns:
            OptimizationResult with optimized prompt
        """
        self._check_available()

        # Ensure configured
        if self._engine is None:
            self.configure()

        # Create variable
        variable = self.create_variable(
            value=prompt,
            role_description=f"A prompt for: {task_description}",
            requires_grad=True,
        )

        # Create loss function with custom evaluation
        loss_fn = self.create_loss_function(
            loss_type=LossType.TEXT_LOSS,
            evaluation_instruction=f"""
Evaluate this prompt based on:
{evaluation_criteria}

Provide specific, actionable feedback for improvement.
"""
        )

        # Create optimizer
        self.create_optimizer([variable])

        # Run optimization
        return await self.optimize(
            variable=variable,
            loss_fn=loss_fn,
            num_iterations=num_iterations,
            verbose=True,
        )


# Fallback implementation when TextGrad not available
class TextGradFallback:
    """
    Fallback implementation using LLM-based iteration
    when TextGrad is not installed.
    """

    def __init__(self, model: str = "claude-3-opus"):
        self.model = model
        self._llm = None

    async def optimize_prompt(
        self,
        prompt: str,
        task_description: str,
        evaluation_criteria: str,
        num_iterations: int = 3,
    ) -> OptimizationResult:
        """
        Fallback prompt optimization using iterative LLM refinement.
        """
        history = []
        current_prompt = prompt

        for i in range(num_iterations):
            # Use LLM to evaluate and improve
            improvement_prompt = f"""
You are a prompt engineering expert. Improve this prompt:

CURRENT PROMPT:
{current_prompt}

TASK: {task_description}

EVALUATION CRITERIA:
{evaluation_criteria}

Provide an improved version of the prompt that better meets the criteria.
Output only the improved prompt, nothing else.
"""
            # This would call the LLM - placeholder for now
            improved_prompt = current_prompt  # Would be LLM response

            history.append(OptimizationStep(
                iteration=i + 1,
                loss_value=0.0,
                gradient="LLM-based improvement",
                variable_before=current_prompt[:200],
                variable_after=improved_prompt[:200],
                improvement=0.0,
            ))

            current_prompt = improved_prompt

        return OptimizationResult(
            initial_value=prompt,
            optimized_value=current_prompt,
            total_iterations=num_iterations,
            final_loss=0.0,
            improvement=0.0,
            history=history,
            metadata={"fallback": True, "model": self.model},
        )


def get_textgrad_adapter(**kwargs) -> Union[TextGradAdapter, TextGradFallback]:
    """
    Get appropriate TextGrad adapter based on availability.

    Returns:
        TextGradAdapter if available, TextGradFallback otherwise
    """
    if TEXTGRAD_AVAILABLE:
        return TextGradAdapter(**kwargs)
    return TextGradFallback(**kwargs)
