# SDK Integration Guide for Claude Code CLI

> **Curated SDKs for autonomous Claude Code CLI workflows**
> 
> This guide covers production-ready SDKs optimized for agentic development patterns.

---

## Quick Reference

| Domain | SDK | Install | Language |
|--------|-----|---------|----------|
| File Watching | chokidar | `npm i chokidar` | TypeScript |
| Git Operations | simple-git | `npm i simple-git` | TypeScript |
| AST Parsing | ast-grep | `cargo install ast-grep` | Rust/CLI |
| Testing | Vitest | `npm i vitest` | TypeScript |
| HTTP Client | httpx | `pip install httpx[http2]` | Python |
| MCP Server | FastMCP | `pip install fastmcp` | Python |
| Memory | mem0 | `pip install mem0ai` | Python |
| Multi-Agent | claude-flow | `npx claude-flow init` | TypeScript |

---

## Core SDKs

### 1. File System - chokidar

High-performance file system watcher optimized for large codebases.

#### Quick Start

```typescript
import chokidar from 'chokidar';

// Initialize watcher with performance optimizations
const watcher = chokidar.watch('src/**/*.ts', {
  persistent: true,
  ignoreInitial: true,
  awaitWriteFinish: { stabilityThreshold: 100 }
});

// React to file changes
watcher.on('change', (path: string) => console.log(`Changed: ${path}`));
watcher.on('add', (path: string) => console.log(`Added: ${path}`));
watcher.on('unlink', (path: string) => console.log(`Deleted: ${path}`));
```

#### Configuration Options

```typescript
interface ChokidarConfig {
  persistent: boolean;           // Keep process running
  ignoreInitial: boolean;        // Skip initial add events
  followSymlinks: boolean;       // Follow symlinked directories
  depth: number;                 // Subdirectory traversal depth
  awaitWriteFinish: {
    stabilityThreshold: number;  // Wait for write completion (ms)
    pollInterval: number;        // Polling interval (ms)
  };
  ignored: string | RegExp | (string | RegExp)[];  // Ignore patterns
  usePolling: boolean;           // Use polling (for network mounts)
  interval: number;              // Polling interval
}
```

#### Agentic Usage Example

```typescript
import chokidar from 'chokidar';
import { EventEmitter } from 'events';

interface FileChange {
  type: 'add' | 'change' | 'unlink';
  path: string;
  timestamp: Date;
}

class AgentFileWatcher extends EventEmitter {
  private watcher: chokidar.FSWatcher;
  private changeBuffer: FileChange[] = [];
  private debounceTimer: NodeJS.Timeout | null = null;

  constructor(patterns: string[], private debounceMs: number = 500) {
    super();
    this.watcher = chokidar.watch(patterns, {
      persistent: true,
      ignoreInitial: true,
      awaitWriteFinish: { stabilityThreshold: 100 },
      ignored: [
        '**/node_modules/**',
        '**/.git/**',
        '**/dist/**',
        '**/*.log'
      ]
    });

    this.setupListeners();
  }

  private setupListeners(): void {
    const events: Array<'add' | 'change' | 'unlink'> = ['add', 'change', 'unlink'];
    
    events.forEach(event => {
      this.watcher.on(event, (path: string) => {
        this.changeBuffer.push({
          type: event,
          path,
          timestamp: new Date()
        });
        this.scheduleEmit();
      });
    });
  }

  private scheduleEmit(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    
    this.debounceTimer = setTimeout(() => {
      const changes = [...this.changeBuffer];
      this.changeBuffer = [];
      this.emit('batch', changes);
    }, this.debounceMs);
  }

  async close(): Promise<void> {
    await this.watcher.close();
  }
}

// Usage in Claude Code CLI agent
const watcher = new AgentFileWatcher(['src/**/*.ts', 'tests/**/*.ts']);

watcher.on('batch', (changes: FileChange[]) => {
  console.log(`Processing ${changes.length} file changes`);
  // Trigger test runs, linting, or analysis
  changes.forEach(change => {
    if (change.type === 'change' && change.path.endsWith('.test.ts')) {
      console.log(`Test file changed: ${change.path}`);
    }
  });
});
```

---

### 2. Git Operations - simple-git

Programmatic Git operations with async/await support.

#### Quick Start

```typescript
import simpleGit, { SimpleGit, StatusResult } from 'simple-git';

const git: SimpleGit = simpleGit();

// Stage and commit changes
await git.add('.').commit('feat: automated change');

// Get diff from last commit
const diff: string = await git.diff(['HEAD~1']);

// Check status
const status: StatusResult = await git.status();
console.log(`Modified files: ${status.modified.length}`);
```

#### Configuration Options

```typescript
import simpleGit, { SimpleGitOptions } from 'simple-git';

const options: Partial<SimpleGitOptions> = {
  baseDir: process.cwd(),      // Working directory
  binary: 'git',               // Git binary path
  maxConcurrentProcesses: 6,   // Parallel operations
  trimmed: true,               // Trim output strings
  timeout: {
    block: 30000               // Timeout for operations (ms)
  }
};

const git = simpleGit(options);
```

#### Agentic Usage Example

```typescript
import simpleGit, { SimpleGit, DiffResult, StatusResult } from 'simple-git';

interface CommitInfo {
  hash: string;
  message: string;
  files: string[];
}

interface BranchAnalysis {
  current: string;
  ahead: number;
  behind: number;
  modified: string[];
  staged: string[];
}

class AgentGitOperations {
  private git: SimpleGit;

  constructor(workDir: string = process.cwd()) {
    this.git = simpleGit({
      baseDir: workDir,
      maxConcurrentProcesses: 6,
      trimmed: true
    });
  }

  async analyzeBranch(): Promise<BranchAnalysis> {
    const status: StatusResult = await this.git.status();
    
    return {
      current: status.current ?? 'unknown',
      ahead: status.ahead,
      behind: status.behind,
      modified: status.modified,
      staged: status.staged
    };
  }

  async getRecentCommits(count: number = 10): Promise<CommitInfo[]> {
    const log = await this.git.log({ maxCount: count });
    
    return log.all.map(commit => ({
      hash: commit.hash,
      message: commit.message,
      files: [] // Would need git show for full file list
    }));
  }

  async smartCommit(message: string, files?: string[]): Promise<string> {
    if (files && files.length > 0) {
      await this.git.add(files);
    } else {
      await this.git.add('.');
    }
    
    const result = await this.git.commit(message);
    return result.commit;
  }

  async getDiffSummary(target: string = 'HEAD~1'): Promise<DiffResult> {
    return await this.git.diffSummary([target]);
  }

  async createFeatureBranch(name: string): Promise<void> {
    const branchName = `feature/${name.toLowerCase().replace(/\s+/g, '-')}`;
    await this.git.checkoutLocalBranch(branchName);
  }

  async safeRebase(targetBranch: string = 'main'): Promise<boolean> {
    try {
      await this.git.fetch();
      await this.git.rebase([targetBranch]);
      return true;
    } catch (error) {
      await this.git.rebase(['--abort']);
      return false;
    }
  }
}

// Usage in Claude Code CLI agent
const gitOps = new AgentGitOperations();

async function automatedCommitWorkflow(): Promise<void> {
  const analysis = await gitOps.analyzeBranch();
  
  if (analysis.modified.length > 0) {
    console.log(`Found ${analysis.modified.length} modified files`);
    
    // Categorize changes
    const testChanges = analysis.modified.filter(f => f.includes('test'));
    const srcChanges = analysis.modified.filter(f => f.includes('src'));
    
    if (testChanges.length > 0) {
      await gitOps.smartCommit('test: update test files', testChanges);
    }
    
    if (srcChanges.length > 0) {
      await gitOps.smartCommit('feat: update source files', srcChanges);
    }
  }
}
```

---

### 3. AST Parsing - ast-grep

Lightning-fast structural code search and rewriting using AST patterns.

#### Quick Start

```bash
# Install via cargo
cargo install ast-grep

# Basic search
ast-grep -p 'console.log($MSG)' --lang typescript

# JSON output for parsing
ast-grep -p 'function $NAME($$$ARGS)' --lang typescript --json
```

#### Python Integration

```python
import subprocess
import json
from typing import TypedDict, Optional
from pathlib import Path


class ASTMatch(TypedDict):
    file: str
    range: dict
    text: str
    metaVariables: dict


def search_pattern(
    pattern: str,
    lang: str = "python",
    path: str = "."
) -> list[ASTMatch]:
    """Search for AST pattern in codebase."""
    result = subprocess.run(
        ["ast-grep", "-l", lang, "-p", pattern, "--json", path],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout) if result.stdout else []


def search_rule(rule_file: str, path: str = ".") -> list[ASTMatch]:
    """Search using a YAML rule file."""
    result = subprocess.run(
        ["ast-grep", "scan", "-r", rule_file, "--json", path],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout) if result.stdout else []


# Example usage
matches = search_pattern("def $FUNC($$$ARGS):", "python", "src/")
for match in matches:
    print(f"Found function in {match['file']}: {match['text'][:50]}...")
```

#### Configuration (sgconfig.yml)

```yaml
ruleDirs:
  - rules/

languageGlobs:
  python: "**/*.py"
  typescript: ["**/*.ts", "**/*.tsx"]
  
testConfigs:
  - testDir: tests/
```

#### Agentic Usage Example

```python
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CodeLocation:
    file: str
    start_line: int
    end_line: int
    start_col: int
    end_col: int


@dataclass
class ASTSearchResult:
    pattern: str
    language: str
    matches: list[dict]
    
    @property
    def count(self) -> int:
        return len(self.matches)
    
    def get_locations(self) -> list[CodeLocation]:
        locations = []
        for match in self.matches:
            rng = match.get('range', {})
            locations.append(CodeLocation(
                file=match.get('file', ''),
                start_line=rng.get('start', {}).get('line', 0),
                end_line=rng.get('end', {}).get('line', 0),
                start_col=rng.get('start', {}).get('column', 0),
                end_col=rng.get('end', {}).get('column', 0)
            ))
        return locations


class AgentASTSearch:
    """AST-based code search for Claude Code CLI agents."""
    
    LANGUAGE_MAP = {
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.js': 'javascript',
        '.jsx': 'jsx',
        '.rs': 'rust',
        '.go': 'go'
    }
    
    def __init__(self, workspace: str = "."):
        self.workspace = Path(workspace)
    
    def search(
        self,
        pattern: str,
        lang: str,
        path: Optional[str] = None
    ) -> ASTSearchResult:
        """Execute AST pattern search."""
        search_path = path or str(self.workspace)
        
        result = subprocess.run(
            ["ast-grep", "-l", lang, "-p", pattern, "--json", search_path],
            capture_output=True,
            text=True,
            cwd=str(self.workspace)
        )
        
        matches = json.loads(result.stdout) if result.stdout else []
        return ASTSearchResult(
            pattern=pattern,
            language=lang,
            matches=matches
        )
    
    def find_functions(self, lang: str = "python") -> ASTSearchResult:
        """Find all function definitions."""
        patterns = {
            'python': 'def $NAME($$$ARGS):',
            'typescript': 'function $NAME($$$ARGS) { $$$BODY }',
            'rust': 'fn $NAME($$$ARGS) { $$$BODY }'
        }
        return self.search(patterns.get(lang, patterns['python']), lang)
    
    def find_classes(self, lang: str = "python") -> ASTSearchResult:
        """Find all class definitions."""
        patterns = {
            'python': 'class $NAME($$$BASES):',
            'typescript': 'class $NAME { $$$BODY }',
            'rust': 'struct $NAME { $$$FIELDS }'
        }
        return self.search(patterns.get(lang, patterns['python']), lang)
    
    def find_imports(self, module: str, lang: str = "python") -> ASTSearchResult:
        """Find imports of a specific module."""
        patterns = {
            'python': f'from {module} import $$$NAMES',
            'typescript': f'import {{ $$$NAMES }} from "{module}"'
        }
        return self.search(patterns.get(lang, patterns['python']), lang)
    
    def find_api_calls(self, method: str, lang: str = "python") -> ASTSearchResult:
        """Find calls to a specific API method."""
        return self.search(f'$OBJ.{method}($$$ARGS)', lang)


# Usage in Claude Code CLI agent
searcher = AgentASTSearch("./src")

# Find all async functions
async_funcs = searcher.search("async def $NAME($$$ARGS):", "python")
print(f"Found {async_funcs.count} async functions")

# Find potentially unsafe patterns
unsafe_evals = searcher.search("eval($EXPR)", "python")
if unsafe_evals.count > 0:
    print(f"WARNING: Found {unsafe_evals.count} eval() calls")
    for loc in unsafe_evals.get_locations():
        print(f"  - {loc.file}:{loc.start_line}")
```

---

### 4. HTTP Client - httpx

Modern async HTTP client with HTTP/2 support and streaming capabilities.

#### Quick Start

```python
import httpx

# Synchronous usage
response = httpx.get('https://api.example.com/data')
data = response.json()

# Async usage
async with httpx.AsyncClient() as client:
    response = await client.get('https://api.example.com/data')
    data = response.json()
```

#### Configuration Options

```python
import httpx
from typing import Optional

# Client with full configuration
client = httpx.AsyncClient(
    base_url='https://api.example.com',
    headers={'Authorization': 'Bearer token'},
    timeout=httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=10.0,
        pool=5.0
    ),
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0
    ),
    http2=True,  # Enable HTTP/2
    follow_redirects=True
)
```

#### Streaming Example

```python
import httpx
from typing import AsyncIterator


async def stream_llm_response(
    url: str,
    prompt: str,
    headers: dict[str, str]
) -> AsyncIterator[str]:
    """Stream responses from an LLM API."""
    async with httpx.AsyncClient(http2=True) as client:
        async with client.stream(
            'POST',
            url,
            json={'prompt': prompt},
            headers=headers,
            timeout=60.0
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                yield chunk


async def stream_with_retry(
    url: str,
    payload: dict,
    max_retries: int = 3
) -> AsyncIterator[str]:
    """Stream with automatic retry on failure."""
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(http2=True) as client:
                async with client.stream(
                    'POST',
                    url,
                    json=payload,
                    timeout=httpx.Timeout(connect=10.0, read=120.0)
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield line
                    return  # Success, exit retry loop
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff


# Usage
async def main():
    async for chunk in stream_llm_response(
        'https://api.anthropic.com/v1/messages',
        'Explain async/await',
        {'x-api-key': 'key', 'anthropic-version': '2023-06-01'}
    ):
        print(chunk, end='', flush=True)
```

#### Agentic Usage Example

```python
import httpx
import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime


@dataclass
class APIResponse:
    status: int
    data: Any
    elapsed_ms: float
    timestamp: datetime


class AgentHTTPClient:
    """HTTP client optimized for Claude Code CLI agents."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = base_url
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            http2=True,
            follow_redirects=True
        )
    
    async def get(self, endpoint: str, params: Optional[dict] = None) -> APIResponse:
        """Execute GET request with timing."""
        start = datetime.now()
        response = await self.client.get(endpoint, params=params)
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        return APIResponse(
            status=response.status_code,
            data=response.json() if response.is_success else response.text,
            elapsed_ms=elapsed,
            timestamp=start
        )
    
    async def post(self, endpoint: str, data: dict) -> APIResponse:
        """Execute POST request with timing."""
        start = datetime.now()
        response = await self.client.post(endpoint, json=data)
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        return APIResponse(
            status=response.status_code,
            data=response.json() if response.is_success else response.text,
            elapsed_ms=elapsed,
            timestamp=start
        )
    
    async def batch_get(
        self,
        endpoints: list[str],
        concurrency: int = 5
    ) -> list[APIResponse]:
        """Execute multiple GET requests with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_get(endpoint: str) -> APIResponse:
            async with semaphore:
                return await self.get(endpoint)
        
        return await asyncio.gather(*[limited_get(ep) for ep in endpoints])
    
    async def close(self) -> None:
        await self.client.aclose()
    
    async def __aenter__(self) -> 'AgentHTTPClient':
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.close()


# Usage in Claude Code CLI agent
async def fetch_documentation_batch():
    async with AgentHTTPClient('https://api.github.com') as client:
        endpoints = [
            '/repos/anthropics/anthropic-sdk-python',
            '/repos/microsoft/TypeScript',
            '/repos/python/cpython'
        ]
        
        results = await client.batch_get(endpoints, concurrency=3)
        
        for result in results:
            if result.status == 200:
                print(f"Repo: {result.data['full_name']} ({result.elapsed_ms:.0f}ms)")
```

---

### 5. MCP Server - FastMCP

Build Model Context Protocol servers with minimal boilerplate.

#### Quick Start

```python
from fastmcp import FastMCP

mcp = FastMCP("agent-tools")


@mcp.tool()
async def search_code(pattern: str) -> list[str]:
    """Search codebase with pattern."""
    # Implementation
    return ["file1.py:10", "file2.py:25"]


@mcp.resource("file://{path}")
async def read_file(path: str) -> str:
    """Read file contents."""
    from pathlib import Path
    return Path(path).read_text()


# Run the server
if __name__ == "__main__":
    mcp.run()
```

#### Configuration Options

```python
from fastmcp import FastMCP
from fastmcp.server import Settings

settings = Settings(
    name="agent-tools",
    version="1.0.0",
    description="Tools for Claude Code CLI",
)

mcp = FastMCP(settings)
```

#### Tool Definition Patterns

```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from typing import Optional

mcp = FastMCP("code-analysis-tools")


# Simple tool with docstring description
@mcp.tool()
async def list_files(directory: str = ".") -> list[str]:
    """List all files in a directory recursively."""
    from pathlib import Path
    return [str(p) for p in Path(directory).rglob("*") if p.is_file()]


# Tool with Pydantic model for complex inputs
class SearchParams(BaseModel):
    pattern: str = Field(description="Search pattern (regex supported)")
    file_types: list[str] = Field(default=["py", "ts"], description="File extensions")
    max_results: int = Field(default=100, ge=1, le=1000)


@mcp.tool()
async def search_codebase(params: SearchParams) -> dict:
    """Search codebase with advanced options."""
    import re
    from pathlib import Path
    
    results = []
    regex = re.compile(params.pattern)
    
    for ext in params.file_types:
        for path in Path(".").rglob(f"*.{ext}"):
            try:
                content = path.read_text()
                matches = regex.findall(content)
                if matches:
                    results.append({
                        "file": str(path),
                        "matches": matches[:10]  # Limit matches per file
                    })
            except Exception:
                continue
            
            if len(results) >= params.max_results:
                break
    
    return {"total": len(results), "results": results}


# Tool with error handling
@mcp.tool()
async def execute_command(command: str, cwd: Optional[str] = None) -> dict:
    """Execute shell command safely."""
    import asyncio
    
    # Validate command
    dangerous = ['rm -rf', 'sudo', 'chmod 777', '> /dev']
    if any(d in command for d in dangerous):
        return {"error": "Command contains dangerous patterns", "allowed": False}
    
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        
        return {
            "returncode": proc.returncode,
            "stdout": stdout.decode()[:10000],  # Limit output size
            "stderr": stderr.decode()[:2000]
        }
    except asyncio.TimeoutError:
        return {"error": "Command timed out", "timeout": 30}
```

#### Resource Provider Example

```python
from fastmcp import FastMCP
from pathlib import Path
import json

mcp = FastMCP("workspace-resources")


@mcp.resource("file://{path}")
async def read_file(path: str) -> str:
    """Read any file from workspace."""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if file_path.suffix in ['.json', '.yaml', '.yml']:
        return file_path.read_text()
    elif file_path.suffix in ['.py', '.ts', '.js', '.md']:
        return file_path.read_text()
    else:
        return f"Binary file: {path} ({file_path.stat().st_size} bytes)"


@mcp.resource("project://structure")
async def get_project_structure() -> str:
    """Get project directory structure."""
    def build_tree(path: Path, prefix: str = "") -> list[str]:
        lines = []
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        
        for i, entry in enumerate(entries):
            if entry.name.startswith('.'):
                continue
            
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            
            if entry.is_dir() and entry.name not in ['node_modules', '__pycache__', 'dist']:
                extension = "    " if is_last else "│   "
                lines.extend(build_tree(entry, prefix + extension))
        
        return lines
    
    tree = build_tree(Path("."))
    return "\n".join(tree[:200])  # Limit output


@mcp.resource("config://{name}")
async def get_config(name: str) -> str:
    """Get configuration file contents."""
    config_files = {
        'package': 'package.json',
        'tsconfig': 'tsconfig.json',
        'pyproject': 'pyproject.toml',
        'env': '.env.example'
    }
    
    if name not in config_files:
        return json.dumps({"error": f"Unknown config: {name}", "available": list(config_files.keys())})
    
    path = Path(config_files[name])
    if path.exists():
        return path.read_text()
    return json.dumps({"error": f"Config file not found: {config_files[name]}"})


if __name__ == "__main__":
    mcp.run()
```

---

### 6. Memory Layer - mem0

Persistent memory for AI agents with semantic search.

#### Quick Start

```python
from mem0 import Memory

# Initialize memory
memory = Memory()

# Store a memory
memory.add("User prefers TypeScript over JavaScript", user_id="user1")

# Search memories
results = memory.search("programming preferences", user_id="user1")
for result in results:
    print(f"Memory: {result['text']} (score: {result['score']:.2f})")
```

#### Configuration Options

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "chroma",  # or "qdrant", "pinecone"
        "config": {
            "collection_name": "agent_memories",
            "path": "./.mem0/chroma"
        }
    },
    "llm": {
        "provider": "anthropic",
        "config": {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.0
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}

memory = Memory.from_config(config)
```

#### Store/Recall Patterns

```python
from mem0 import Memory
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MemoryEntry:
    text: str
    score: float
    metadata: dict


class AgentMemory:
    """Memory layer for Claude Code CLI agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory = Memory()
    
    def remember(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5
    ) -> str:
        """Store a new memory."""
        result = self.memory.add(
            content,
            user_id=self.agent_id,
            metadata={
                "category": category,
                "importance": importance,
                "timestamp": datetime.now().isoformat()
            }
        )
        return result.get("id", "")
    
    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5
    ) -> list[MemoryEntry]:
        """Search memories relevant to query."""
        results = self.memory.search(
            query,
            user_id=self.agent_id,
            limit=limit
        )
        
        entries = []
        for r in results:
            if category and r.get('metadata', {}).get('category') != category:
                continue
            entries.append(MemoryEntry(
                text=r['text'],
                score=r['score'],
                metadata=r.get('metadata', {})
            ))
        
        return entries
    
    def remember_error(self, error: str, context: str) -> str:
        """Store error for future avoidance."""
        return self.remember(
            f"ERROR: {error}\nCONTEXT: {context}",
            category="errors",
            importance=0.8
        )
    
    def remember_success(self, action: str, outcome: str) -> str:
        """Store successful action pattern."""
        return self.remember(
            f"SUCCESS: {action}\nOUTCOME: {outcome}",
            category="successes",
            importance=0.7
        )
    
    def get_relevant_context(self, task: str) -> str:
        """Get all relevant memories for a task."""
        memories = self.recall(task, limit=10)
        
        if not memories:
            return "No relevant past experiences found."
        
        context_parts = ["Relevant past experiences:"]
        for mem in memories:
            context_parts.append(f"- {mem.text} (relevance: {mem.score:.2f})")
        
        return "\n".join(context_parts)
    
    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            return True
        except Exception:
            return False
    
    def get_all_memories(self) -> list[dict]:
        """Retrieve all memories for this agent."""
        return self.memory.get_all(user_id=self.agent_id)


# Usage in Claude Code CLI agent
agent_memory = AgentMemory("code-assistant-v1")

# Store coding preferences
agent_memory.remember(
    "User prefers functional programming patterns over OOP",
    category="preferences",
    importance=0.9
)

agent_memory.remember(
    "Project uses pytest for testing with fixtures in conftest.py",
    category="project",
    importance=0.8
)

# Recall when needed
task = "Write tests for the authentication module"
context = agent_memory.get_relevant_context(task)
print(context)

# Learn from mistakes
agent_memory.remember_error(
    "ImportError: cannot import 'AsyncClient' from 'requests'",
    "Tried to use async with requests library instead of httpx"
)
```

---

### 7. Multi-Agent - claude-flow v3

Swarm orchestration for multi-agent Claude Code CLI workflows.

#### Quick Start

```bash
# Initialize in project
npx claude-flow init

# Start orchestrator
npx claude-flow start

# Run with specific topology
npx claude-flow start --topology mesh
```

#### Swarm Configuration (claude-flow.json)

```json
{
  "name": "code-review-swarm",
  "version": "1.0.0",
  "topology": "hierarchical",
  "agents": {
    "coordinator": {
      "role": "coordinator",
      "model": "claude-sonnet-4-20250514",
      "maxConcurrentTasks": 10,
      "systemPrompt": "You coordinate code review tasks among specialist agents."
    },
    "reviewer": {
      "role": "executor",
      "model": "claude-sonnet-4-20250514",
      "maxConcurrentTasks": 5,
      "capabilities": ["code-review", "best-practices"],
      "systemPrompt": "You are a senior code reviewer focusing on quality and maintainability."
    },
    "security": {
      "role": "executor",
      "model": "claude-sonnet-4-20250514",
      "maxConcurrentTasks": 3,
      "capabilities": ["security-audit", "vulnerability-detection"],
      "systemPrompt": "You audit code for security vulnerabilities and compliance."
    },
    "performance": {
      "role": "executor",
      "model": "claude-sonnet-4-20250514",
      "maxConcurrentTasks": 3,
      "capabilities": ["performance-analysis", "optimization"],
      "systemPrompt": "You analyze code for performance issues and optimization opportunities."
    }
  },
  "behaviors": {
    "workStealing": true,
    "loadBalancing": true,
    "autoScaling": {
      "enabled": true,
      "minAgents": 2,
      "maxAgents": 8,
      "scaleUpThreshold": 0.8,
      "scaleDownThreshold": 0.2
    }
  },
  "routing": {
    "strategy": "capability-based",
    "fallback": "round-robin"
  }
}
```

#### Programmatic Usage (TypeScript)

```typescript
import { 
  ClaudeFlow, 
  Topology, 
  AgentRole,
  TaskPriority 
} from 'claude-flow';

interface ReviewTask {
  file: string;
  type: 'security' | 'performance' | 'general';
  priority: TaskPriority;
}

async function runCodeReviewSwarm(files: string[]): Promise<void> {
  // Initialize swarm
  const flow = new ClaudeFlow({
    topology: Topology.HIERARCHICAL,
    behaviors: {
      workStealing: true,
      loadBalancing: true
    }
  });

  await flow.start();

  // Register agents
  flow.registerAgent({
    id: 'coordinator',
    role: AgentRole.COORDINATOR,
    capabilities: ['task-distribution', 'result-aggregation'],
    maxConcurrentTasks: 10
  });

  for (let i = 0; i < 3; i++) {
    flow.registerAgent({
      id: `reviewer-${i}`,
      role: AgentRole.EXECUTOR,
      capabilities: ['code-review', 'suggestions'],
      maxConcurrentTasks: 5
    });
  }

  // Submit review tasks
  const tasks: ReviewTask[] = files.map(file => ({
    file,
    type: file.includes('auth') ? 'security' : 'general',
    priority: TaskPriority.HIGH
  }));

  for (const task of tasks) {
    flow.submitTask({
      description: `Review ${task.file}`,
      priority: task.priority,
      requiredCapabilities: ['code-review'],
      metadata: { file: task.file, type: task.type }
    });
  }

  // Process all tasks
  await flow.processAll();

  // Get results
  const metrics = flow.getMetrics();
  console.log(`Completed ${metrics.completedTasks} reviews`);
  console.log(`Average time: ${metrics.averageTaskTime}ms`);

  await flow.stop();
}

// Execute
runCodeReviewSwarm([
  'src/auth/login.ts',
  'src/api/users.ts',
  'src/utils/helpers.ts'
]);
```

#### Agent Definition (YAML)

```yaml
# agents/code-reviewer.yaml
name: code-reviewer
role: executor
model: claude-sonnet-4-20250514

capabilities:
  - code-review
  - typescript
  - python
  - best-practices

maxConcurrentTasks: 5
timeout: 300000  # 5 minutes

systemPrompt: |
  You are a thorough code reviewer. For each file:
  1. Check for bugs and logic errors
  2. Verify error handling
  3. Assess code readability
  4. Suggest improvements
  
  Format your response as:
  ## Summary
  ## Issues Found
  ## Suggestions
  ## Rating (1-10)

tools:
  - name: read_file
    description: Read file contents
  - name: search_codebase
    description: Search for related code
  - name: get_git_history
    description: Get file change history

hooks:
  onTaskStart:
    - log_task_start
  onTaskComplete:
    - store_review_result
    - update_metrics
```

---

## Performance Tips

### Caching Strategies

```python
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any, Optional
import hashlib
import json


class TTLCache:
    """Simple TTL cache for API responses."""
    
    def __init__(self, default_ttl: int = 300):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self.default_ttl = default_ttl
    
    def _hash_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        hashed = self._hash_key(key)
        if hashed in self._cache:
            value, expiry = self._cache[hashed]
            if datetime.now() < expiry:
                return value
            del self._cache[hashed]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        hashed = self._hash_key(key)
        expiry = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self._cache[hashed] = (value, expiry)
    
    def invalidate(self, key: str) -> None:
        hashed = self._hash_key(key)
        self._cache.pop(hashed, None)


# Usage with API calls
cache = TTLCache(default_ttl=600)  # 10 minute cache

async def cached_api_call(endpoint: str, client: 'AgentHTTPClient') -> dict:
    cached = cache.get(endpoint)
    if cached:
        return cached
    
    response = await client.get(endpoint)
    if response.status == 200:
        cache.set(endpoint, response.data)
    
    return response.data
```

### Streaming Best Practices

```python
import asyncio
from collections.abc import AsyncIterator


async def process_stream_with_buffer(
    stream: AsyncIterator[str],
    buffer_size: int = 10
) -> AsyncIterator[list[str]]:
    """Buffer stream output for batch processing."""
    buffer: list[str] = []
    
    async for chunk in stream:
        buffer.append(chunk)
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []
    
    if buffer:  # Yield remaining items
        yield buffer


async def parallel_streams(
    streams: list[AsyncIterator[str]],
    max_concurrent: int = 5
) -> AsyncIterator[tuple[int, str]]:
    """Process multiple streams concurrently."""
    async def consume_stream(idx: int, stream: AsyncIterator[str]):
        async for chunk in stream:
            yield (idx, chunk)
    
    tasks = [consume_stream(i, s) for i, s in enumerate(streams[:max_concurrent])]
    
    # Use asyncio.as_completed for concurrent processing
    for coro in asyncio.as_completed([t.__anext__() for t in tasks]):
        try:
            result = await coro
            yield result
        except StopAsyncIteration:
            continue
```

### Batch Processing

```typescript
interface BatchConfig {
  batchSize: number;
  concurrency: number;
  delayBetweenBatches: number;
}

async function processBatches<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  config: BatchConfig = { batchSize: 10, concurrency: 5, delayBetweenBatches: 100 }
): Promise<R[]> {
  const results: R[] = [];
  
  // Split into batches
  const batches: T[][] = [];
  for (let i = 0; i < items.length; i += config.batchSize) {
    batches.push(items.slice(i, i + config.batchSize));
  }
  
  for (const batch of batches) {
    // Process batch with concurrency limit
    const semaphore = new Array(config.concurrency).fill(null);
    let index = 0;
    
    const batchResults = await Promise.all(
      batch.map(async (item) => {
        const slot = index++ % config.concurrency;
        await semaphore[slot];
        semaphore[slot] = processor(item);
        return semaphore[slot];
      })
    );
    
    results.push(...batchResults);
    
    // Delay between batches to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, config.delayBetweenBatches));
  }
  
  return results;
}

// Usage
const files = ['file1.ts', 'file2.ts', /* ... */];
const results = await processBatches(
  files,
  async (file) => analyzeFile(file),
  { batchSize: 10, concurrency: 5, delayBetweenBatches: 200 }
);
```

---

## Installation Script

Complete setup script to install all SDKs:

```bash
#!/bin/bash
# setup-claude-code-sdks.sh
# Complete SDK installation for Claude Code CLI

set -e

echo "=== Claude Code CLI SDK Installation ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 not found${NC}"
        return 1
    else
        echo -e "${GREEN}✓ $1 found${NC}"
        return 0
    fi
}

echo "Checking prerequisites..."
check_command node || { echo "Please install Node.js first"; exit 1; }
check_command npm || { echo "Please install npm first"; exit 1; }
check_command python3 || { echo "Please install Python 3 first"; exit 1; }
check_command pip || check_command pip3 || { echo "Please install pip first"; exit 1; }

# Optional: Rust for ast-grep
if check_command cargo; then
    HAS_RUST=true
else
    echo -e "${YELLOW}⚠ Rust not found - ast-grep will be skipped${NC}"
    HAS_RUST=false
fi

echo ""
echo "=== Installing TypeScript SDKs ==="

# Create package.json if not exists
if [ ! -f package.json ]; then
    echo '{"name": "claude-code-workspace", "type": "module"}' > package.json
fi

# TypeScript SDKs
echo "Installing chokidar..."
npm install chokidar

echo "Installing simple-git..."
npm install simple-git

echo "Installing vitest..."
npm install -D vitest

echo "Installing claude-flow..."
npm install claude-flow

echo ""
echo "=== Installing Python SDKs ==="

# Python SDKs
echo "Installing httpx with HTTP/2 support..."
pip install httpx[http2]

echo "Installing FastMCP..."
pip install fastmcp

echo "Installing mem0ai..."
pip install mem0ai

echo ""
echo "=== Installing CLI Tools ==="

# ast-grep (requires Rust)
if [ "$HAS_RUST" = true ]; then
    echo "Installing ast-grep..."
    cargo install ast-grep
else
    echo -e "${YELLOW}Skipping ast-grep (requires Rust)${NC}"
fi

echo ""
echo "=== Verifying Installation ==="

# Verify Node packages
node -e "require('chokidar')" && echo -e "${GREEN}✓ chokidar${NC}" || echo -e "${RED}✗ chokidar${NC}"
node -e "require('simple-git')" && echo -e "${GREEN}✓ simple-git${NC}" || echo -e "${RED}✗ simple-git${NC}"

# Verify Python packages
python3 -c "import httpx" && echo -e "${GREEN}✓ httpx${NC}" || echo -e "${RED}✗ httpx${NC}"
python3 -c "import fastmcp" && echo -e "${GREEN}✓ fastmcp${NC}" || echo -e "${RED}✗ fastmcp${NC}"
python3 -c "import mem0" && echo -e "${GREEN}✓ mem0${NC}" || echo -e "${RED}✗ mem0${NC}"

# Verify CLI tools
if [ "$HAS_RUST" = true ]; then
    command -v ast-grep && echo -e "${GREEN}✓ ast-grep${NC}" || echo -e "${RED}✗ ast-grep${NC}"
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Initialize claude-flow: npx claude-flow init"
echo "2. Configure API keys in .env"
echo "3. Run 'npx vitest' to verify testing setup"
echo ""
```

### Windows PowerShell Version

```powershell
# setup-claude-code-sdks.ps1
# Complete SDK installation for Claude Code CLI (Windows)

$ErrorActionPreference = "Stop"

Write-Host "=== Claude Code CLI SDK Installation ===" -ForegroundColor Cyan
Write-Host ""

function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check prerequisites
Write-Host "Checking prerequisites..."
if (-not (Test-Command "node")) { Write-Host "Please install Node.js first" -ForegroundColor Red; exit 1 }
if (-not (Test-Command "npm")) { Write-Host "Please install npm first" -ForegroundColor Red; exit 1 }
if (-not (Test-Command "python")) { Write-Host "Please install Python first" -ForegroundColor Red; exit 1 }
if (-not (Test-Command "pip")) { Write-Host "Please install pip first" -ForegroundColor Red; exit 1 }

$hasRust = Test-Command "cargo"
if (-not $hasRust) {
    Write-Host "Rust not found - ast-grep will be skipped" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Installing TypeScript SDKs ===" -ForegroundColor Cyan

# Create package.json if not exists
if (-not (Test-Path "package.json")) {
    '{"name": "claude-code-workspace", "type": "module"}' | Out-File -FilePath "package.json" -Encoding utf8
}

npm install chokidar
npm install simple-git
npm install -D vitest
npm install claude-flow

Write-Host ""
Write-Host "=== Installing Python SDKs ===" -ForegroundColor Cyan

pip install httpx[http2]
pip install fastmcp
pip install mem0ai

Write-Host ""
Write-Host "=== Installing CLI Tools ===" -ForegroundColor Cyan

if ($hasRust) {
    cargo install ast-grep
}

Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Initialize claude-flow: npx claude-flow init"
Write-Host "2. Configure API keys in .env"
Write-Host "3. Run 'npx vitest' to verify testing setup"
```

---

## Version Information

| SDK | Minimum Version | Tested Version |
|-----|-----------------|----------------|
| chokidar | 3.5.0 | 3.6.0 |
| simple-git | 3.20.0 | 3.27.0 |
| ast-grep | 0.20.0 | 0.31.0 |
| vitest | 1.0.0 | 2.1.0 |
| httpx | 0.25.0 | 0.28.0 |
| fastmcp | 0.1.0 | 0.4.0 |
| mem0ai | 0.1.0 | 0.1.29 |
| claude-flow | 3.0.0 | 3.2.0 |

---

*Last updated: 2026-01-24*
*Version: 4.0 - Claude Code CLI Edition*
