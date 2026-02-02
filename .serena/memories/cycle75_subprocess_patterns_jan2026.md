# Cycle 75: subprocess Module Patterns (January 2026)

## Overview

The `subprocess` module spawns new processes, connects to their I/O pipes, and obtains return codes. It replaces older modules like `os.system` and `os.spawn*`.

**Key Principle**: Use `subprocess.run()` for simple cases, `Popen` for full control.

---

## 1. subprocess.run() - Recommended High-Level API (Python 3.5+)

### Basic Usage

```python
import subprocess

# Simple command execution
result = subprocess.run(["ls", "-l"])
print(result.returncode)  # 0 on success

# Capture output
result = subprocess.run(
    ["git", "status"],
    capture_output=True,  # Shorthand for stdout=PIPE, stderr=PIPE
    text=True,            # Decode output as string (not bytes)
    check=True            # Raise CalledProcessError on non-zero exit
)
print(result.stdout)
print(result.stderr)

# With timeout
result = subprocess.run(
    ["long_running_command"],
    timeout=30,  # seconds
    capture_output=True
)
```

### Key Parameters

```python
subprocess.run(
    args,                    # Command as list or string (if shell=True)
    stdin=None,              # Input: None, PIPE, DEVNULL, file descriptor
    stdout=None,             # Output: None, PIPE, DEVNULL, file descriptor
    stderr=None,             # Error: None, PIPE, DEVNULL, STDOUT
    capture_output=False,    # Shorthand for stdout=PIPE, stderr=PIPE
    shell=False,             # Execute through shell (SECURITY RISK)
    cwd=None,                # Working directory
    timeout=None,            # Seconds before TimeoutExpired
    check=False,             # Raise CalledProcessError on non-zero exit
    encoding=None,           # Text encoding (enables text mode)
    errors=None,             # Encoding error handling
    text=None,               # Alias for universal_newlines (text mode)
    env=None,                # Environment variables (replaces current)
    input=None,              # Data to send to stdin
)
```

### CompletedProcess Object

```python
result = subprocess.run(["echo", "hello"], capture_output=True, text=True)

result.args         # ['echo', 'hello']
result.returncode   # 0
result.stdout       # 'hello\n'
result.stderr       # ''
result.check_returncode()  # Raises CalledProcessError if non-zero
```

---

## 2. Popen - Low-Level Process Control

### Basic Popen Usage

```python
from subprocess import Popen, PIPE, DEVNULL

# Start process without waiting
proc = Popen(["long_task"], stdout=PIPE, stderr=PIPE)

# Do other work while process runs...

# Wait and get output
stdout, stderr = proc.communicate(timeout=60)
print(f"Exit code: {proc.returncode}")
```

### Popen as Context Manager

```python
with Popen(["command"], stdout=PIPE) as proc:
    output = proc.stdout.read()
# Process is automatically waited for on exit
```

### Interactive Process Communication

```python
proc = Popen(
    ["python", "-i"],
    stdin=PIPE,
    stdout=PIPE,
    stderr=PIPE,
    text=True
)

# Send input and get output
stdout, stderr = proc.communicate(input="print('hello')\nexit()\n")
```

### Non-Blocking Output Reading

```python
import select

proc = Popen(["tail", "-f", "log.txt"], stdout=PIPE, text=True)

while True:
    # Check if output is available (Unix only)
    readable, _, _ = select.select([proc.stdout], [], [], 1.0)
    if readable:
        line = proc.stdout.readline()
        if line:
            print(line, end='')
    
    # Check if process ended
    if proc.poll() is not None:
        break
```

---

## 3. Popen Methods and Attributes

### Methods

```python
proc = Popen(["command"])

proc.poll()           # Check if terminated, returns returncode or None
proc.wait(timeout=30) # Wait for termination (can deadlock with PIPE!)
proc.communicate(input=None, timeout=None)  # Send input, read output, wait

proc.send_signal(signal.SIGTERM)  # Send signal
proc.terminate()      # Send SIGTERM (Unix) / TerminateProcess (Windows)
proc.kill()           # Send SIGKILL (Unix) / same as terminate (Windows)
```

### Attributes

```python
proc.args        # The args argument
proc.stdin       # StreamWriter if stdin=PIPE
proc.stdout      # StreamReader if stdout=PIPE
proc.stderr      # StreamReader if stderr=PIPE
proc.pid         # Process ID
proc.returncode  # Exit code (None if still running)
```

---

## 4. SECURITY: Avoiding Command Injection

### DANGEROUS: shell=True with User Input

```python
# VULNERABLE - Command injection!
user_input = "; rm -rf /"
subprocess.run(f"echo {user_input}", shell=True)  # DISASTER!

# VULNERABLE - Even with f-strings
filename = user_input
subprocess.run(f"cat {filename}", shell=True)  # NEVER DO THIS
```

### SAFE: Use List Arguments (No Shell)

```python
# SAFE - Arguments are escaped automatically
user_input = "; rm -rf /"
subprocess.run(["echo", user_input])  # Prints literal "; rm -rf /"

# SAFE - Filename with special characters
filename = "file with spaces.txt"
subprocess.run(["cat", filename])  # Works correctly
```

### When Shell is Required: Use shlex.quote()

```python
import shlex

# If you MUST use shell=True (rarely needed)
user_input = "file with spaces.txt"
safe_input = shlex.quote(user_input)
subprocess.run(f"cat {safe_input}", shell=True)  # Quoted safely

# Better: Parse complex commands with shlex.split()
command_line = 'grep "search term" file.txt'
args = shlex.split(command_line)
subprocess.run(args)  # ['grep', 'search term', 'file.txt']
```

### PEP 787: T-Strings for Safer Subprocess (Python 3.15+)

```python
# Future: Template strings with automatic escaping
# subprocess.run(t"echo {user_input}")  # Coming in Python 3.15
```

---

## 5. Environment Variables

### Modifying Environment

```python
import os

# Add to current environment
env = os.environ.copy()
env["MY_VAR"] = "value"
env["PATH"] = f"/custom/bin:{env['PATH']}"

result = subprocess.run(["my_command"], env=env)

# Minimal environment (CAREFUL: may break things)
result = subprocess.run(
    ["command"],
    env={"PATH": "/usr/bin", "HOME": "/tmp"}
)
```

### Windows Considerations

```python
# On Windows, env MUST include SystemRoot for many commands
env = os.environ.copy()
env["MY_VAR"] = "value"
# SystemRoot is preserved from os.environ.copy()
```

---

## 6. Timeout Handling

### Basic Timeout

```python
try:
    result = subprocess.run(
        ["slow_command"],
        timeout=30,
        capture_output=True
    )
except subprocess.TimeoutExpired as e:
    print(f"Command timed out after {e.timeout}s")
    print(f"Partial stdout: {e.stdout}")
    print(f"Partial stderr: {e.stderr}")
```

### Timeout with Popen (Proper Cleanup)

```python
proc = subprocess.Popen(["long_task"], stdout=PIPE, stderr=PIPE)

try:
    stdout, stderr = proc.communicate(timeout=30)
except subprocess.TimeoutExpired:
    proc.kill()  # MUST kill on timeout
    stdout, stderr = proc.communicate()  # Finish reading pipes
    raise  # Re-raise or handle
```

---

## 7. Pipe Patterns

### Shell Pipeline Equivalent

```python
# Shell: dmesg | grep hda
p1 = Popen(["dmesg"], stdout=PIPE)
p2 = Popen(["grep", "hda"], stdin=p1.stdout, stdout=PIPE)
p1.stdout.close()  # Allow p1 to receive SIGPIPE if p2 exits
output = p2.communicate()[0]
```

### Redirect stderr to stdout

```python
result = subprocess.run(
    ["command"],
    stdout=PIPE,
    stderr=subprocess.STDOUT,  # Merge stderr into stdout
    text=True
)
print(result.stdout)  # Contains both stdout and stderr
```

### Suppress Output

```python
result = subprocess.run(
    ["noisy_command"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
```

---

## 8. asyncio Subprocess (Async/Await)

### Basic Async Subprocess

```python
import asyncio

async def run_command(cmd: str) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return stdout, stderr

# Run it
stdout, stderr = asyncio.run(run_command("ls -la"))
```

### Async with Executable (Safer)

```python
async def run_safe(program: str, *args: str) -> tuple[bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        program, *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    return await proc.communicate()

stdout, stderr = asyncio.run(run_safe("git", "status"))
```

### Async Timeout

```python
async def run_with_timeout(cmd: list[str], timeout: float):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
        return stdout, stderr
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()  # Clean up
        raise

# Usage
try:
    out, err = asyncio.run(run_with_timeout(["slow"], 5.0))
except asyncio.TimeoutError:
    print("Command timed out")
```

### Parallel Subprocess Execution

```python
async def main():
    # Run multiple commands in parallel
    results = await asyncio.gather(
        run_safe("ls", "-la"),
        run_safe("pwd"),
        run_safe("whoami"),
    )
    for stdout, stderr in results:
        print(stdout.decode())

asyncio.run(main())
```

---

## 9. Windows-Specific Patterns

### Hide Console Window

```python
import subprocess
import sys

if sys.platform == "win32":
    # CREATE_NO_WINDOW prevents console window from appearing
    result = subprocess.run(
        ["command"],
        creationflags=subprocess.CREATE_NO_WINDOW,
        capture_output=True
    )
```

### Run Detached Process

```python
if sys.platform == "win32":
    proc = Popen(
        ["background_task"],
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    )
```

### STARTUPINFO for GUI Apps

```python
if sys.platform == "win32":
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    
    proc = Popen(["app.exe"], startupinfo=startupinfo)
```

---

## 10. Exception Handling

### Exception Hierarchy

```python
subprocess.SubprocessError        # Base class
├── subprocess.TimeoutExpired     # Timeout exceeded
└── subprocess.CalledProcessError # Non-zero exit with check=True
```

### Comprehensive Error Handling

```python
def run_safely(cmd: list[str], timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        return result.stdout
    
    except FileNotFoundError:
        raise RuntimeError(f"Command not found: {cmd[0]}")
    
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Command timed out after {e.timeout}s")
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed with code {e.returncode}: {e.stderr}"
        )
```

---

## 11. Best Practices Summary

### DO:
- Use `subprocess.run()` for simple cases
- Pass commands as lists: `["cmd", "arg1", "arg2"]`
- Use `capture_output=True` and `text=True` for readable output
- Use `check=True` to catch failures
- Always handle `TimeoutExpired` when using timeouts
- Kill and communicate after timeout with Popen

### DON'T:
- Use `shell=True` with untrusted input (command injection!)
- Use `stdout=PIPE` with `wait()` (deadlock risk)
- Forget to close `p1.stdout` in pipelines
- Ignore return codes without `check=True`
- Use deprecated `os.system()` or `os.popen()`

---

## 12. Quick Reference

| Task | Code |
|------|------|
| Run and wait | `subprocess.run(["cmd"])` |
| Capture output | `subprocess.run(cmd, capture_output=True, text=True)` |
| Check exit code | `subprocess.run(cmd, check=True)` |
| With timeout | `subprocess.run(cmd, timeout=30)` |
| Send input | `subprocess.run(cmd, input="data", text=True)` |
| Custom env | `subprocess.run(cmd, env={...})` |
| Working dir | `subprocess.run(cmd, cwd="/path")` |
| Suppress output | `subprocess.run(cmd, stdout=DEVNULL, stderr=DEVNULL)` |
| Merge stderr | `subprocess.run(cmd, stdout=PIPE, stderr=STDOUT)` |
| Async exec | `await asyncio.create_subprocess_exec(*cmd)` |

---

## References

- Python 3.14 subprocess documentation
- PEP 324 - subprocess module proposal
- PEP 787 - Safer subprocess with t-strings (deferred to 3.15)
- Semgrep command injection prevention guide
