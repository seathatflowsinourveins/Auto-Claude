# Multi-Project Management Guide

## Available Projects

### 1. unleash (Primary Research Platform)
- **Path**: `Z:\insider\AUTO CLAUDE\unleash`
- **Purpose**: Autonomous AI development, Ralph Loop, SDK ecosystem
- **Languages**: Python 3.12+
- **Key Features**: Research engine, Claude-Flow v3, MCP integrations

### 2. Touchdesigner-createANDBE (State of Witness)
- **Path**: `Z:\insider\AUTO CLAUDE\Touchdesigner-createANDBE`
- **Purpose**: Creative AI visual generation via TouchDesigner
- **Languages**: Python 3.11+, GLSL 4.60
- **Key Features**: Pose tracking, archetype mapping, 2M particles

## Project Switching Commands

```python
# Switch to unleash
mcp__plugin_serena_serena__activate_project(project="unleash")

# Switch to TouchDesigner/State of Witness
mcp__plugin_serena_serena__activate_project(project="Touchdesigner-createANDBE")

# Check current config
mcp__plugin_serena_serena__get_current_config()

# List memories for current project
mcp__plugin_serena_serena__list_memories()
```

## Creating New Projects

To add a new project to Serena:

1. **Via Configuration**: Add to Serena's config file
2. **Via Path**: Use `activate_project(project="path/to/project")`
3. **Onboarding**: Run `onboarding()` after activation for new projects

## Memory Architecture Per Project

Each project maintains isolated memories:
- `project_overview` - Architecture, tech stack, conventions
- `suggested_commands` - Common shell commands
- Project-specific memories for domain knowledge

## Best Practices

1. **Always check active project** before operations
2. **Read project_overview** memory after switching
3. **Create memories** for reusable knowledge
4. **Isolate concerns** - keep project knowledge separate
5. **Use onboarding** for new projects to generate overview

## Session Workflow

```
1. Check get_current_config() at session start
2. Activate target project if different
3. Read relevant memories (project_overview, etc.)
4. Perform work
5. Write new memories for discoveries
6. Switch projects as needed
```

## Integration Points

| Project | MCP Servers | Ports |
|---------|-------------|-------|
| unleash | Qdrant, PostgreSQL | 6333, 5432 |
| State of Witness | TouchDesigner | 9981 |
| Both | Serena dashboard | 24282 |
