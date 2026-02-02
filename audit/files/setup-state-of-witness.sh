#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# State of Witness - Creative/ML System Project Setup Script
# Claude CLI = INTEGRATED generative brain with real-time control
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }

PROJECT_DIR="${1:-$(pwd)}"

echo -e "${MAGENTA}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   State of Witness - Creative/ML System Setup                 ║"
echo "║   Claude CLI = INTEGRATED Generative Brain                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Create directory structure
log_info "Creating project structure in: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"/{.claude/{commands,agents,skills,hooks,rules},docs,src/{visuals,ml,generation},tests,experiments,assets}

# Create .mcp.json
log_info "Creating .mcp.json..."
cat > "$PROJECT_DIR/.mcp.json" << 'EOF'
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "touchdesigner": {
      "command": "npx",
      "args": ["touchdesigner-mcp-server@latest", "--stdio"],
      "env": {
        "TD_HOST": "${TD_HOST:-127.0.0.1}",
        "TD_PORT": "${TD_PORT:-9981}"
      }
    },
    "comfyui": {
      "command": "python",
      "args": ["-m", "mcp_comfyui"],
      "env": {
        "COMFYUI_HOST": "${COMFYUI_HOST:-localhost}",
        "COMFYUI_PORT": "${COMFYUI_PORT:-8188}"
      }
    },
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp"],
      "env": {
        "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI:-http://localhost:5000}"
      }
    },
    "wandb": {
      "url": "https://mcp.withwandb.com/mcp",
      "apiKey": "${WANDB_API_KEY}"
    },
    "context7": {
      "url": "https://mcp.context7.com/sse"
    }
  }
}
EOF
log_success "Created .mcp.json"

# Create .claude/settings.json
log_info "Creating .claude/settings.json..."
cat > "$PROJECT_DIR/.claude/settings.json" << 'EOF'
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allowedTools": [
      "Read",
      "Write",
      "Edit",
      "MultiEdit",
      "Glob",
      "Grep",
      "Bash(git *)",
      "Bash(python *)",
      "Bash(pip *)",
      "Bash(pytest *)",
      "Bash(mlflow *)",
      "mcp__touchdesigner_*",
      "mcp__comfyui_*",
      "mcp__mlflow_*",
      "mcp__wandb_*",
      "mcp__github_*",
      "mcp__context7_*"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "mcp__touchdesigner_*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"🎨 TouchDesigner operation complete\"}'",
            "timeout": 3
          }
        ]
      },
      {
        "matcher": "mcp__comfyui_*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"🖼️ ComfyUI generation complete\"}'",
            "timeout": 3
          }
        ]
      },
      {
        "matcher": "mcp__mlflow_*|mcp__wandb_*",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"feedback\": \"📊 Experiment tracked\"}'",
            "timeout": 3
          }
        ]
      },
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "command -v ruff &>/dev/null && ruff format \"$file\" --quiet && ruff check \"$file\" --fix --quiet || true",
            "timeout": 15
          }
        ]
      }
    ]
  }
}
EOF
log_success "Created .claude/settings.json"

# Create CLAUDE.md
log_info "Creating CLAUDE.md..."
cat > "$PROJECT_DIR/CLAUDE.md" << 'EOF'
# State of Witness - Computational Art & ML Generative System

## Project Overview
State of Witness is a computational art/ML generative system with real-time TouchDesigner control.
- Claude CLI IS integrated as **generative brain** with real-time MCP control capabilities
- Real-time visual generation and manipulation
- ML-powered creative processes

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                  State of Witness Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Claude CLI (Generative Brain)                                  │
│        │                                                         │
│        ├──► TouchDesigner MCP ──► Real-time Visuals             │
│        │         (port 9981)                                     │
│        │                                                         │
│        ├──► ComfyUI MCP ──► Generated Textures/Assets           │
│        │         (port 8188)                                     │
│        │                                                         │
│        ├──► MLflow MCP ──► Experiment Tracking                  │
│        │                                                         │
│        └──► W&B MCP ──► Model Performance                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

@docs/architecture.md (for detailed architecture)

## Development Commands
```bash
# Start services
mlflow ui                          # Start MLflow at localhost:5000
# Start TouchDesigner with WebServer DAT on port 9981
# Start ComfyUI server on port 8188

# Test
pytest tests/ -v                   # Run tests

# Experiments
python -m experiments.run          # Run experiments
```

## MCP Usage Guidelines
| Server | Purpose | Required Setup |
|--------|---------|----------------|
| touchdesigner | Real-time visuals | TD with WebServer DAT on 9981 |
| comfyui | Image generation | ComfyUI on port 8188 |
| mlflow | Experiment tracking | MLflow server on 5000 |
| wandb | Model tracking | WANDB_API_KEY set |
| github | Version control | GITHUB_TOKEN set |

## TouchDesigner Integration
Claude controls TD via MCP with these operations:
- `create_node` - Create operators
- `delete_node` - Remove operators
- `update_parameters` - Modify parameters
- `execute_script` - Run Python in TD
- `connect` - Wire operators together

### TD Setup Requirements
1. Import `mcp_webserver_base.tox` into project
2. Ensure WebServer DAT running on port 9981
3. Set execute permissions for Python scripts

## ComfyUI Integration
- Generate textures and visual assets
- Run custom workflows
- Batch generation for variations

## Code Standards
- **Python**: Type hints, Pydantic models
- **Experiments**: Always log to MLflow/W&B
- **Assets**: Organize in assets/ directory
- **Documentation**: Document generative processes

## Experiment Tracking
All experiments MUST be logged:
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"param": value})
    mlflow.log_metrics({"metric": value})
    mlflow.log_artifact("output.png")
```
EOF
log_success "Created CLAUDE.md"

# Create generate-visual command
log_info "Creating slash commands..."
cat > "$PROJECT_DIR/.claude/commands/generate-visual.md" << 'EOF'
Generate visuals for: $ARGUMENTS

## Generation Pipeline

1. **Concept Development**
   - Interpret the creative brief
   - Define visual parameters
   - Select appropriate techniques

2. **Asset Generation (ComfyUI)**
   - Generate base textures
   - Create variations
   - Apply style transfer if needed

3. **TouchDesigner Integration**
   - Create node network
   - Apply generated assets
   - Configure parameters

4. **Real-time Control**
   - Set up parameter controls
   - Configure audio reactivity if applicable
   - Test performance

5. **Experiment Logging**
   - Log parameters to MLflow
   - Save output artifacts
   - Document the process

## Output
- Generated visual assets
- TouchDesigner network
- Experiment logged with parameters
EOF
log_success "Created generate-visual.md"

# Create td-control command
cat > "$PROJECT_DIR/.claude/commands/td-control.md" << 'EOF'
Control TouchDesigner: $ARGUMENTS

## Available Operations

### Node Operations
- `create <type> <name>` - Create a new node
  - Types: TOP, CHOP, SOP, DAT, COMP, MAT
- `delete <name>` - Remove a node
- `connect <from> <to>` - Wire two nodes

### Parameter Control
- `param <node> <param> <value>` - Set a parameter
- `get_params <node>` - List all parameters
- `pulse <node> <param>` - Pulse a parameter

### Script Execution
- `execute <script>` - Run Python code in TD
- `expression <node> <param> <expr>` - Set parameter expression

### Network
- `list` - List all nodes
- `check_errors` - Check for errors

## Examples
```
/td-control create TOP noise myNoise
/td-control param myNoise roughness 0.5
/td-control connect myNoise feedback1
/td-control execute print(op('myNoise').par.roughness)
```

## Prerequisites
- TouchDesigner running
- WebServer DAT on port 9981
- mcp_webserver_base.tox imported
EOF
log_success "Created td-control.md"

# Create experiment-track command
cat > "$PROJECT_DIR/.claude/commands/experiment-track.md" << 'EOF'
Track experiment: $ARGUMENTS

## Experiment Tracking Protocol

1. **Initialize Experiment**
   - Create MLflow run
   - Set experiment name and tags
   - Log initial parameters

2. **During Experiment**
   - Log metrics at intervals
   - Capture intermediate outputs
   - Track resource usage

3. **Finalize**
   - Log final metrics
   - Save output artifacts
   - Generate summary

## MLflow Integration
```python
import mlflow

mlflow.set_experiment("experiment_name")

with mlflow.start_run(run_name="run_name"):
    # Log parameters
    mlflow.log_params({
        "param1": value1,
        "param2": value2
    })
    
    # Log metrics
    mlflow.log_metrics({
        "loss": loss_value,
        "accuracy": acc_value
    })
    
    # Log artifacts
    mlflow.log_artifact("output.png")
    mlflow.log_artifact("model.pth")
```

## W&B Integration
For model performance tracking and comparison across runs.

## Required Information
- Experiment name
- Parameters to track
- Metrics to log
- Artifacts to save
EOF
log_success "Created experiment-track.md"

# Create visual-style rule
cat > "$PROJECT_DIR/.claude/rules/visual-style.md" << 'EOF'
---
paths:
  - "**/visuals/**"
  - "**/generation/**"
  - "**/*.toe"
---

# Visual Style Guidelines

## Aesthetic Principles
- Computational art focus
- Real-time generative systems
- Quality-diversity in outputs

## TouchDesigner Best Practices
- Use TOPs for 2D processing
- Use CHOPs for audio/data
- Use SOPs for 3D geometry
- Use DATs for text/data
- Keep networks organized with containers

## ComfyUI Workflows
- Save reusable workflows
- Document node configurations
- Version control workflow JSON

## Performance Considerations
- Monitor GPU memory
- Optimize texture resolutions
- Profile frame rates
- Use caching where appropriate

## Output Standards
- Document all parameters
- Save seeds for reproducibility
- Export at multiple resolutions
- Archive successful experiments
EOF
log_success "Created visual-style.md rule"

# Create ml-best-practices rule
cat > "$PROJECT_DIR/.claude/rules/ml-best-practices.md" << 'EOF'
---
paths:
  - "**/ml/**"
  - "**/experiments/**"
  - "**/*model*.py"
---

# ML Best Practices

## Experiment Management
- ALWAYS log experiments to MLflow or W&B
- Use descriptive run names
- Tag experiments appropriately
- Never run experiments without logging

## Reproducibility
- Set and log random seeds
- Version control model configurations
- Log environment/dependency versions
- Save model checkpoints

## Model Development
- Start with simple baselines
- Iterate incrementally
- Use proper train/val/test splits
- Monitor for overfitting

## Quality-Diversity
- Use pyribs for QD algorithms
- Explore parameter spaces
- Document diversity metrics
- Archive diverse outputs

## Resource Management
- Profile GPU memory usage
- Use mixed precision where applicable
- Batch appropriately for hardware
- Clean up unused tensors
EOF
log_success "Created ml-best-practices.md rule"

# Create docs/architecture.md
cat > "$PROJECT_DIR/docs/architecture.md" << 'EOF'
# State of Witness Architecture

## System Overview

State of Witness is a computational art system where Claude CLI serves as the generative brain, controlling real-time visual systems through MCP protocols.

## Component Details

### Claude CLI (Generative Brain)
- Natural language creative direction
- Real-time parameter control
- Experiment management
- Asset generation orchestration

### TouchDesigner Integration
- WebSocket communication (port 9981)
- Node creation and manipulation
- Real-time parameter control
- Python script execution

### ComfyUI Integration
- HTTP API (port 8188)
- Stable Diffusion workflows
- Texture generation
- Style transfer

### MLflow/W&B Integration
- Experiment tracking
- Metric logging
- Artifact storage
- Model versioning

## Data Flow

```
                    ┌─────────────────┐
                    │   Claude CLI    │
                    │ (Generative     │
                    │    Brain)       │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │ TouchDesigner │ │    ComfyUI    │ │ MLflow/W&B    │
   │   (Visuals)   │ │  (Generation) │ │  (Tracking)   │
   └───────────────┘ └───────────────┘ └───────────────┘
           │                 │                 │
           │                 │                 │
           ▼                 ▼                 ▼
   ┌─────────────────────────────────────────────────┐
   │              Output / Performance               │
   └─────────────────────────────────────────────────┘
```

## Development Workflow

1. **Design Phase**
   - Claude CLI designs visual concepts
   - Define parameter spaces
   - Create initial node networks

2. **Implementation Phase**
   - Generate assets with ComfyUI
   - Build TD networks
   - Configure real-time controls

3. **Experiment Phase**
   - Track experiments with MLflow
   - Iterate on parameters
   - Document successful runs

4. **Performance Phase**
   - Real-time visual output
   - Live parameter manipulation
   - Audience interaction

## Dependencies

- Python 3.10+
- TouchDesigner 2024+
- ComfyUI (latest)
- MLflow 3.5.1+
- PyTorch / diffusers
- mediapipe
- pyribs
EOF
log_success "Created docs/architecture.md"

# Create .gitignore
cat > "$PROJECT_DIR/.gitignore" << 'EOF'
# Environment
.env
.env.*
.env.local

# Secrets
secrets/
*_key.json
*.pem
*.key

# Claude local settings
.claude/settings.local.json
CLAUDE.local.md

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# TouchDesigner
*.toe.bak
*.toe.autosave
Backup/

# Generated assets (large files)
assets/generated/
outputs/
*.png
*.jpg
*.mp4
!docs/images/*.png
!tests/fixtures/*.png

# MLflow
mlruns/
mlartifacts/

# ComfyUI
comfyui_outputs/

# Models (large files)
models/
*.pth
*.pt
*.ckpt
*.safetensors
EOF
log_success "Created .gitignore"

# Create experiments directory structure
mkdir -p "$PROJECT_DIR/experiments"
cat > "$PROJECT_DIR/experiments/README.md" << 'EOF'
# Experiments

All experiments should be tracked with MLflow or W&B.

## Structure

```
experiments/
├── README.md
├── configs/           # Experiment configurations
├── scripts/           # Experiment scripts
└── notebooks/         # Jupyter notebooks
```

## Running Experiments

1. Start MLflow server: `mlflow ui`
2. Run experiment: `python -m experiments.run --config configs/exp1.yaml`
3. View results: http://localhost:5000
EOF
log_success "Created experiments/README.md"

echo ""
echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║   State of Witness Project Setup Complete!                    ║${NC}"
echo -e "${MAGENTA}╠═══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${MAGENTA}║   Directory: $PROJECT_DIR${NC}"
echo -e "${MAGENTA}║                                                               ║${NC}"
echo -e "${MAGENTA}║   Prerequisites:                                              ║${NC}"
echo -e "${MAGENTA}║   • TouchDesigner with WebServer DAT on port 9981            ║${NC}"
echo -e "${MAGENTA}║   • ComfyUI server on port 8188                              ║${NC}"
echo -e "${MAGENTA}║   • MLflow: ${YELLOW}mlflow ui${MAGENTA}                                        ║${NC}"
echo -e "${MAGENTA}║                                                               ║${NC}"
echo -e "${MAGENTA}║   Start development:                                          ║${NC}"
echo -e "${MAGENTA}║   ${YELLOW}cd $PROJECT_DIR && claude${NC}${MAGENTA}                          ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════════╝${NC}"
