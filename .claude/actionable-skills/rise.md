# /rise - Recursive Iterative Self-Improvement Skill

## Description
Apply RISE (Recursive Improvement of Self-Enhancement) for iterative refinement of solutions. Each iteration builds on the previous with explicit self-critique.

## When to Use
- First solution attempt didn't fully satisfy requirements
- Code needs optimization or polish
- Output quality can be improved
- Complex task requiring multiple passes

## RISE Protocol (What I Actually Do)

### Step 1: Generate Initial Solution
```
TOOLS:
- Write/Edit: Create first pass
- TodoWrite: Note what I attempted

OUTPUT: Working but potentially suboptimal solution
```

### Step 2: Self-Critique
```
INTERNAL PROCESS:
- Does this actually solve the problem?
- What edge cases did I miss?
- Is the code efficient? Readable?
- Does it match the user's style?

OUTPUT: List of improvement opportunities
```

### Step 3: Improve
```
TOOLS:
- Edit: Refine existing code
- Read: Reference related files for consistency

OUTPUT: Improved solution addressing critique
```

### Step 4: Evaluate
```
INTERNAL PROCESS:
- Compare to original requirements
- Measure improvement vs critique points
- Decide: Iterate again or complete?

If score < threshold:
  → Return to Step 2
Else:
  → Mark complete
```

## Iteration Tracking

Each RISE cycle should update TodoWrite:
```
Iteration 1: Initial implementation (Score: 6/10)
  - Missing input validation
  - No error handling

Iteration 2: Added validation (Score: 8/10)
  - Still missing edge case for empty input

Iteration 3: Complete (Score: 9/10)
  - All requirements met
```

## Stopping Conditions
- Score reaches acceptable threshold (typically 8+/10)
- User signals satisfaction
- Diminishing returns (improvement < 0.5 per iteration)
- Maximum iterations reached (typically 3-5)

## Example

```
User: "Write a function to parse user input"

RISE Iteration 1:
def parse_input(data):
    return data.strip().lower()

Self-Critique: No validation, no error handling, hardcoded behavior

RISE Iteration 2:
def parse_input(data: str, normalize: bool = True) -> str:
    if data is None:
        raise ValueError("Input cannot be None")
    result = data.strip()
    if normalize:
        result = result.lower()
    return result

Self-Critique: Better, but what about empty strings?

RISE Iteration 3:
def parse_input(data: str | None, normalize: bool = True) -> str:
    if data is None or not data.strip():
        raise ValueError("Input cannot be empty")
    result = data.strip()
    if normalize:
        result = result.lower()
    return result

Score: 9/10 - Complete
```

## Integration with V30.8

This skill implements Section 38 (RISE Framework) of CLAUDE_SELF_ENHANCEMENT_V30.md

Key insight: RISE requires explicit self-critique between iterations - not just regenerating.
