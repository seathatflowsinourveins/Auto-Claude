# Complete Creative/ML System Development Toolkit for Claude Code CLI

**Claude Code CLI transforms into a powerful creative AI development environment** with MCPs for real-time visual programming (TouchDesigner), generative AI image creation (ComfyUI), ML experiment tracking (MLflow, Weights & Biases), and 3D modeling (Blender). This toolkit enables the full development lifecycle for State of Witness and similar creative/ML systems.

The key distinction for State of Witness: **Claude CLI IS part of the architecture** — functioning as a generative brain with real-time MCP control to TouchDesigner.

---

## Tier 1: Real-Time Visual Programming MCPs

### 1. TouchDesigner MCP Server (8beeeaaat)
**Status**: Production-ready, actively maintained  
**Purpose**: AI-controlled visual programming for interactive installations

This is the primary TouchDesigner integration for Claude CLI, enabling real-time control of TouchDesigner projects.

```bash
# Installation via npm
npm install touchdesigner-mcp-server

# Claude Code setup
claude mcp add touchdesigner -- npx touchdesigner-mcp-server@latest
```

**Configuration**:
```json
{
  "mcpServers": {
    "touchdesigner": {
      "command": "npx",
      "args": ["touchdesigner-mcp-server@latest"],
      "env": {
        "TD_HOST": "127.0.0.1",
        "TD_PORT": "9981"
      }
    }
  }
}
```

**TouchDesigner Setup**:
1. Import `mcp_webserver_base.tox` into your project
2. Ensure WebServer DAT is running on port 9981
3. Restart Claude and TouchDesigner

**Key Tools**:
| Tool | Purpose |
|------|---------|
| `create_node` | Create new nodes in TD |
| `delete_node` | Remove existing nodes |
| `call_method` | Execute Python methods on nodes |
| `execute_script` | Run arbitrary Python in TD |
| `get_parameters` | Query node parameters |
| `update_parameters` | Modify node parameters |
| `check_errors` | Diagnose node errors |

**Example Prompts**:
```
"Create a particle system with gravity forces"
"Connect the noise TOP to the feedback loop"
"Execute a Python script to randomize all parameters"
"Check for errors on the render network"
```

---

### 2. TouchDesigner Documentation MCP (bottobot)
**Status**: Production-ready, 629 operators documented  
**Purpose**: Comprehensive TD documentation access for AI coding

```bash
# Clone and setup
git clone https://github.com/bottobot/touchdesigner-mcp-server
cd touchdesigner-mcp-server
npm install
```

**Key Features**:
- 629 TouchDesigner operators documented
- 14 interactive tutorials
- 69 Python API classes
- Search and query TD documentation

**Use Case**: When Claude needs to understand TD concepts before writing scripts, this MCP provides accurate, up-to-date documentation rather than relying on outdated training data.

---

## Tier 2: Generative AI Image MCPs

### 3. ComfyUI MCP Server (Multiple Implementations)

Multiple production-ready implementations exist for ComfyUI integration:

#### Option A: lalanikarim/comfy-mcp-server (Lightweight)
```bash
pip install comfy-mcp-server
```

**Configuration**:
```json
{
  "mcpServers": {
    "comfyui": {
      "command": "uvx",
      "args": ["comfy-mcp-server"],
      "env": {
        "COMFY_URL": "http://127.0.0.1:8188",
        "COMFY_WORKFLOW_JSON_FILE": "/path/to/workflow.json",
        "PROMPT_NODE_ID": "6",
        "OUTPUT_NODE_ID": "9",
        "OUTPUT_MODE": "file"
      }
    }
  }
}
```

#### Option B: SamuraiBuddha/mcp-comfyui (Full API Control)
```bash
git clone https://github.com/SamuraiBuddha/mcp-comfyui.git
cd mcp-comfyui
pip install -e .
```

**Configuration**:
```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python",
      "args": ["-m", "mcp_comfyui"],
      "env": {
        "COMFYUI_HOST": "localhost",
        "COMFYUI_PORT": "8188"
      }
    }
  }
}
```

**Key Tools**:
| Tool | Purpose |
|------|---------|
| `generate_image` | Generate images from prompts |
| `execute_workflow` | Run saved ComfyUI workflows |
| `list_models` | Get available checkpoints |
| `list_samplers` | Get sampling methods |
| `list_schedulers` | Get noise schedulers |
| `upload_image` | Upload for img2img |
| `get_queue` | Check generation queue |
| `cancel_generation` | Stop current generation |

**Example Prompts**:
```
"Generate a cyberpunk cityscape at 1024x1024 using SDXL"
"Run the logo_generator workflow with style=minimalist"
"List all available LoRA models"
"Upload this reference image and generate variations"
```

---

#### Option C: PurlieuStudios/comfyui-mcp (Game Dev Focus)
For Godot integration and game asset generation:

```bash
git clone https://github.com/purlieu-studios/comfyui-mcp.git
cd comfyui-mcp
pip install -e .
```

**Key Features**:
- Character portrait generation
- Item icon creation
- Environment art and tileable textures
- Procedural asset generation

---

## Tier 3: ML Experiment Tracking MCPs

### 4. MLflow MCP Server (Official)
**Status**: Production-ready (MLflow 3.5.1+)  
**Purpose**: ML experiment tracking and trace analysis

```bash
# Official MLflow MCP (requires MLflow 3.5.1+)
pip install mlflow>=3.5.1

# Start the MCP server
mlflow mcp
```

**Configuration**:
```json
{
  "mcpServers": {
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_EXPERIMENT_ID": "your-experiment-id"
      }
    }
  }
}
```

**Key Tools**:
| Tool | Purpose |
|------|---------|
| `search_traces` | Query traces with filters |
| `get_trace` | Get trace details |
| `log_feedback` | Log evaluation scores |
| `delete_traces` | Remove old traces |
| `get_trace_tag` | Get trace metadata |
| `set_trace_tag` | Set trace metadata |

**Example Prompts**:
```
"Find all failed traces in the last hour"
"Show me the slowest traces with execution times over 5 seconds"
"Log a relevance score of 0.85 for trace tr-abc123"
"Delete traces older than 30 days from experiment 1"
```

---

### 5. MLflow MCP Server (Community - kkruglik)
**Status**: Production-ready  
**Purpose**: Traditional ML lifecycle management

```bash
# Install via uvx
claude mcp add mlflow -- uvx mlflow-mcp
```

**Configuration**:
```json
{
  "mcpServers": {
    "mlflow": {
      "command": "uvx",
      "args": ["mlflow-mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    }
  }
}
```

**Key Capabilities**:
- Experiment management
- Run analysis and comparison
- Metrics and parameter queries
- Model registry exploration

---

### 6. Weights & Biases MCP Server (Official)
**Status**: Production-ready, hosted and local options  
**Purpose**: Experiment tracking, Weave traces, and reports

#### Option A: Hosted Server (Recommended)
```bash
# Claude Code setup
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server
uvx wandb login
```

**Configuration (Hosted)**:
```json
{
  "mcpServers": {
    "wandb": {
      "url": "https://mcp.withwandb.com/mcp",
      "apiKey": "YOUR_WANDB_API_KEY"
    }
  }
}
```

#### Option B: Local Server
```json
{
  "mcpServers": {
    "wandb": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/wandb/wandb-mcp-server", "wandb_mcp_server"],
      "env": {
        "WANDB_API_KEY": "${WANDB_API_KEY}"
      }
    }
  }
}
```

**Key Tools**:
| Tool | Purpose |
|------|---------|
| `query_wandb_gql` | GraphQL queries for experiments |
| `query_weave_traces` | Query LLM traces |
| `list_entities_projects` | List available W&B projects |
| `create_wandb_report` | Create shareable reports |
| `execute_sandbox_code` | Run Python analysis |
| `ask_wandbot` | Query W&B documentation |

**Example Prompts**:
```
"How many traces are in the wandb-applied-ai-team/mcp-tests project?"
"What was my best performing run in terms of F1 score?"
"Create a report comparing the top 5 runs"
"Plot the loss curves for the last 10 training runs"
```

---

## Tier 4: 3D Modeling & Asset Creation MCPs

### 7. Blender MCP Server
**Status**: Production-ready, 14.5k+ stars  
**Purpose**: AI-powered 3D modeling with Claude

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Blender addon
# 1. Download addon.py from the repository
# 2. Open Blender → Edit → Preferences → Add-ons
# 3. Click "Install..." and select addon.py
# 4. Enable "MCP Blender Bridge" addon
```

**Configuration**:
```json
{
  "mcpServers": {
    "blender": {
      "command": "uvx",
      "args": ["blender-mcp"]
    }
  }
}
```

**Key Capabilities**:
- Natural language 3D modeling
- Asset integration (Poly Haven HDRIs, textures)
- Python script execution in Blender
- Material and texture control
- Scene creation and manipulation

**Example Prompts**:
```
"Create a beach scene with palm trees and rocks"
"Add a sunset HDRI and adjust the lighting"
"Generate a low-poly character model"
"Apply ocean materials to the water plane"
```

---

## Complete Creative/ML Toolkit Configuration

```json
{
  "mcpServers": {
    "touchdesigner": {
      "command": "npx",
      "args": ["touchdesigner-mcp-server@latest"],
      "env": {
        "TD_HOST": "127.0.0.1",
        "TD_PORT": "9981"
      }
    },
    "comfyui": {
      "command": "python",
      "args": ["-m", "mcp_comfyui"],
      "env": {
        "COMFYUI_HOST": "localhost",
        "COMFYUI_PORT": "8188"
      }
    },
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:5000"
      }
    },
    "wandb": {
      "url": "https://mcp.withwandb.com/mcp",
      "apiKey": "${WANDB_API_KEY}"
    },
    "blender": {
      "command": "uvx",
      "args": ["blender-mcp"]
    }
  }
}
```

---

## Python SDKs for Creative/ML (No MCP, Use via CLI)

### 8. diffusers (Hugging Face)
```bash
pip install diffusers transformers accelerate
```

**Usage**: Direct Stable Diffusion access when ComfyUI MCP is unavailable.

### 9. MediaPipe
```bash
pip install mediapipe
```

**Usage**: Real-time ML pipelines for face detection, pose estimation, hand tracking.

### 10. PyTorch / TensorFlow
```bash
pip install torch torchvision torchaudio
pip install tensorflow
```

**Usage**: Custom ML model development and training.

### 11. Quality-Diversity Algorithms (pyribs)
```bash
pip install ribs
```

**Usage**: MAP-Elites and other QD algorithms for generative diversity.

---

## State of Witness Architecture Integration

For State of Witness, Claude CLI functions as the **generative brain** with this communication flow:

```
Claude CLI (Generative Brain)
    │
    ├── TouchDesigner MCP ──→ Real-time visuals
    │       ↓
    │   WebSocket (port 9981)
    │       ↓
    │   TouchDesigner (Visual Engine)
    │
    ├── ComfyUI MCP ──→ Generated textures/assets
    │       ↓
    │   HTTP API (port 8188)
    │       ↓
    │   ComfyUI (Diffusion Engine)
    │
    ├── MLflow/W&B MCP ──→ Experiment tracking
    │
    └── Blender MCP ──→ 3D asset generation
```

---

## Development Lifecycle Coverage

| Phase | Tools | Purpose |
|-------|-------|---------|
| **Design** | TouchDesigner MCP, Blender MCP | Visual architecture, 3D assets |
| **Research** | MLflow MCP, W&B MCP | Experiment tracking, model comparison |
| **Implement** | ComfyUI MCP, diffusers | Generative AI, image synthesis |
| **Debug** | MLflow traces, Sentry MCP | Error tracking, performance analysis |
| **Deploy** | TD Documentation MCP | Ensure correct TD patterns |

---

## Token Overhead Analysis

| MCP Server | Approximate Token Cost | Priority |
|------------|----------------------|----------|
| TouchDesigner | ~2,500 tokens | **Critical** |
| ComfyUI | ~2,000 tokens | High |
| MLflow | ~1,500 tokens | Medium |
| W&B | ~2,000 tokens | Medium |
| Blender | ~3,000 tokens | Low (as needed) |

**Recommendation**: Enable TouchDesigner + ComfyUI as primary MCPs, add MLflow/W&B for experiment sessions.

---

## Installation Script

```bash
#!/bin/bash
# creative_ml_toolkit_install.sh

echo "Installing Claude Code Creative/ML Toolkit..."

# TouchDesigner MCP
npm install -g touchdesigner-mcp-server

# ComfyUI MCP
pip install comfy-mcp-server

# MLflow (official)
pip install mlflow>=3.5.1

# W&B MCP
claude mcp add wandb -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server

# Blender MCP
# Note: Requires Blender addon installation separately

# Python SDKs
pip install diffusers transformers accelerate mediapipe ribs

echo "Creative/ML toolkit installation complete!"
```

---

## Key Takeaways

1. **TouchDesigner MCP is Critical**: For State of Witness, this is the primary integration point for Claude's generative brain
2. **ComfyUI for Asset Generation**: Use for textures, backgrounds, and visual elements
3. **MLflow + W&B for Tracking**: Essential for tracking generative experiments and model performance
4. **Blender for 3D Assets**: On-demand 3D asset creation through natural language
5. **Token Budget**: Keep 2-3 creative MCPs active per session

This toolkit transforms Claude CLI into a comprehensive creative AI development environment, enabling real-time control of visual systems, generative AI pipelines, and ML experiment tracking — all through natural language commands.
