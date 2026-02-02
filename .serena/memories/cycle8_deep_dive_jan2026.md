# Cycle 8: Deep Dive Research - January 2026 Week 4

## Executive Summary

Deep research completed January 25, 2026 covering four key areas:
1. Agent Skills API - Marketplace, patterns, best practices
2. MCP Security Patches - CVE fixes, timeline, mitigation
3. CIMD/XAA - Modern OAuth for MCP
4. POPs Advanced - GPU particle techniques

---

## 1. Agent Skills API Deep Dive

### AgentSkills.best - First Marketplace

**URL**: https://agentskills.best
**Status**: Open source on GitHub (PRs welcome)

#### Official Anthropic Skills
| Skill | Purpose |
|-------|---------|
| **Artifacts Builder** | React + Tailwind + shadcn/ui interfaces |
| **MCP Builder** | MCP server creation guidance |
| **Skill Creator** | Meta-skill for building skills |

### SKILL.md Structure (Official)

```markdown
---
name: my-skill
description: What this skill does
author: Your Name
version: 1.0.0
---

# My Skill

## When to Use
- Condition 1
- Condition 2

## Instructions
Step-by-step guidance...

## Examples
```code examples```

## Resources
- scripts/helper.py
- templates/base.json
```

### Progressive Disclosure Pattern

```python
# Load context progressively, not all at once
class ProgressiveSkill:
    def __init__(self):
        self.core_instructions = self.load_core()  # Always load
        self.examples = None  # Load on demand
        self.scripts = None   # Load on demand
    
    def expand_context(self, need: str):
        if need == "examples" and self.examples is None:
            self.examples = self.load_examples()
        elif need == "scripts" and self.scripts is None:
            self.scripts = self.load_scripts()
```

### awesome-claude-skills Repository

**URL**: https://github.com/VoltAgent/awesome-claude-skills
**Stars**: 3.7k
**Content**: Curated collection of production-ready skills

### Key Best Practices (platform.claude.com/docs)

1. **Clear trigger conditions** - When should skill activate?
2. **Bundled resources** - Include scripts, templates, examples
3. **Version control** - Semantic versioning for skills
4. **Progressive disclosure** - Load context on demand
5. **Domain expertise** - Encode institutional knowledge

---

## 2. MCP Security Patches (CRITICAL UPDATE)

### Patched Vulnerabilities (as of Jan 21, 2026)

| CVE | Description | Severity | Fixed Version |
|-----|-------------|----------|---------------|
| CVE-2025-68145 | Path validation bypass | Critical | Dec 18, 2025 |
| CVE-2025-68143 | Unrestricted git_init | Critical | Dec 18, 2025 |
| CVE-2025-68144 | Argument injection git_diff | Critical | Dec 18, 2025 |
| CVE-2025-5277 | aws-mcp-server command injection | Critical | Jan 2026 |

### Timeline
- **Dec 8, 2025**: Vulnerabilities discovered by Cyata
- **Dec 18, 2025**: mcp-server-git patched
- **Jan 20, 2026**: Public disclosure (The Hacker News, Dark Reading)
- **Jan 21, 2026**: Anthropic official patch announcement (SC Media)

### Attack Chain (Pre-Patch)

```
Attacker crafts malicious prompt
    ↓
Git MCP processes prompt injection
    ↓
Path traversal (CVE-2025-68145) reads arbitrary files
    ↓
Combined with Filesystem MCP
    ↓
Remote Code Execution achieved
```

### Mitigation Configuration

```json
{
  "mcp_servers": {
    "git": {
      "version": ">=2025.12.18",  // MUST be patched version
      "sandbox": true,
      "allowed_repos": ["internal/*"],
      "block_operations": ["git_init_arbitrary"]
    },
    "filesystem": {
      "sandbox": true,
      "allowed_paths": ["./project"],
      "block_write": true  // Read-only for untrusted
    }
  }
}
```

### AWS MCP Server (CVE-2025-5277)

```python
# Vulnerable: Command injection via crafted prompt
# Fixed: Input sanitization required

class SecureAWSMCP:
    BLOCKED_CHARS = [';', '|', '&', '`', '$', '(', ')']
    
    def sanitize_input(self, user_input: str) -> str:
        for char in self.BLOCKED_CHARS:
            user_input = user_input.replace(char, '')
        return user_input
    
    def execute_aws_command(self, command: str):
        sanitized = self.sanitize_input(command)
        # Additional validation...
```

---

## 3. CIMD/XAA Implementation

### CIMD (Client ID Metadata Documents)

**Paradigm Shift**: client_id IS the URL, not a random string

```
OLD: client_id = "abc123xyz"  (requires DCR registration)
NEW: client_id = "https://myapp.com/.well-known/oauth-client.json"
```

### CIMD JSON Structure

```json
// Served at: https://myapp.com/.well-known/oauth-client.json
{
  "client_id": "https://myapp.com/.well-known/oauth-client.json",
  "client_name": "My MCP Client",
  "redirect_uris": [
    "https://myapp.com/callback",
    "http://localhost:8080/callback"
  ],
  "grant_types": ["authorization_code"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",  // Public client
  "scope": "read write",
  "logo_uri": "https://myapp.com/logo.png",
  "policy_uri": "https://myapp.com/privacy",
  "tos_uri": "https://myapp.com/terms"
}
```

### Python CIMD Client Implementation

```python
from dataclasses import dataclass
from urllib.parse import urljoin
import httpx

@dataclass
class CIMDClient:
    client_id_url: str  # The URL IS the client_id
    redirect_uri: str
    
    async def get_metadata(self) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.client_id_url)
            return response.json()
    
    def build_auth_url(self, auth_endpoint: str, state: str, code_verifier: str) -> str:
        import hashlib, base64
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        
        return f"{auth_endpoint}?" + "&".join([
            f"client_id={self.client_id_url}",  # URL as client_id
            f"redirect_uri={self.redirect_uri}",
            f"response_type=code",
            f"code_challenge={code_challenge}",
            f"code_challenge_method=S256",
            f"state={state}"
        ])
```

### XAA (Cross-App Access) / ID-JAG

**Purpose**: Enterprise IdP controls AI agent access to apps

```
Before XAA:
User → MCP Client → Direct OAuth → MCP Server
(IdP sees: "User logged into Asana")

After XAA:
User → MCP Client → IdP (policy check) → MCP Server
(IdP sees: "AI app accessing Asana as User, scope: read")
```

### XAA Flow

```python
class XAAFlow:
    def __init__(self, idp_url: str, client_cimd: str):
        self.idp = idp_url
        self.cimd = client_cimd
    
    async def request_access(self, target_app: str, scopes: list[str]):
        # 1. Request ID-JAG from IdP
        id_jag = await self.idp.issue_id_jag(
            client_id=self.cimd,
            target=target_app,
            scopes=scopes
        )
        
        # 2. Exchange ID-JAG for access token at target
        token = await self.target.exchange_id_jag(id_jag)
        
        return token
```

### Benefits for Projects

| Project | CIMD Benefit | XAA Benefit |
|---------|--------------|-------------|
| WITNESS | Simple auth for MCP servers | N/A (no enterprise) |
| TRADING | Verified client identity | Audit trail via IdP |
| UNLEASH | Scalable client management | Centralized policy |

---

## 4. POPs Advanced Techniques

### Official POPs Workshop

**URL**: https://derivative.ca/workshop/touchdesigner-pops-workshop/
**Status**: Available for purchase

### Key POPs Operators

| Operator | Purpose |
|----------|---------|
| **ParticlesPOP** | Core particle generation |
| **TrailPOP** | Trail generation (replaces Trail SOP) |
| **TopToPOP** | Convert TOP to point cloud |
| **ForcePOP** | Apply forces to particles |
| **NoisePOP** | Add noise to attributes |

### Video to Particles Pipeline

```python
# TouchDesigner POPs workflow
class VideoToParticles:
    def setup(self):
        # 1. Video input
        self.video = op('moviefilein1')
        
        # 2. TOP to POP conversion
        self.top_to_pop = op.create('topToPOP', 'topToPOP1')
        self.top_to_pop.par.top = self.video
        
        # 3. Add attributes (color from video)
        self.attrib_create = op.create('attribCreatePOP', 'attribCreate1')
        self.attrib_create.par.name = 'Cd'  # Color
        
        # 4. Apply forces
        self.force = op.create('forcePOP', 'force1')
        self.force.par.forcey = -9.8  # Gravity
        
        # 5. Add curl noise
        self.noise = op.create('noisePOP', 'noise1')
        self.noise.par.type = 'curl'
```

### Effectors Pattern (Function Store)

```glsl
// GLSL effector for custom particle control
vec3 customEffector(vec3 pos, vec3 vel, float age) {
    // Sample texture for force direction
    vec2 uv = pos.xy * 0.5 + 0.5;
    vec4 texForce = texture(sEffectorTex, uv);
    
    // Apply force based on texture
    vec3 force = texForce.rgb * 2.0 - 1.0;
    force *= texForce.a;  // Alpha as strength
    
    return force * uEffectorStrength;
}
```

### Interactive Webcam Particles

```python
# Webcam drives particle behavior
class WebcamParticles:
    def setup(self):
        # Webcam input
        self.webcam = op('videodevin1')
        
        # Motion detection (frame difference)
        self.diff = op.create('differenceTOP', 'diff1')
        self.diff.inputConnectors[0].connect(self.webcam)
        
        # Motion to force field
        self.force_field = self.diff  # High motion = high force
        
        # Apply to particles via effector
        self.particles.par.forceeffector = self.force_field
```

---

## Project Integration Priority

### WITNESS

1. **POPs Migration** (HIGH)
   - Replace particlesGpu with native POPs
   - Use effectors for archetype-driven forces
   - Video-to-particles for pose visualization

2. **Skills for Archetypes** (MEDIUM)
   ```markdown
   ---
   name: witness-archetype-skill
   ---
   # Archetype Visualization Skill
   Generate particle parameters for archetype...
   ```

### TRADING

1. **MCP Security Hardening** (CRITICAL)
   - Update to patched mcp-server-git
   - Block aws-mcp-server until CVE-2025-5277 patched
   - Implement CIMD for client verification

2. **XAA Integration** (HIGH)
   - Route MCP access through enterprise IdP
   - Centralized audit logging

### UNLEASH

1. **Agent Skills Framework** (HIGH)
   - Build skill marketplace structure
   - Implement progressive disclosure
   - Create meta-skills for self-improvement

2. **CIMD for All MCP Clients** (MEDIUM)
   - Standardize on URL-based client_id
   - Remove DCR complexity

---

## Security Update Required

### Immediate Actions

```bash
# Update mcp-server-git
pip install --upgrade mcp-server-git>=2025.12.18

# Verify version
pip show mcp-server-git | grep Version
```

### Configuration Updates

```json
{
  "mcp_security": {
    "git": {
      "require_version": ">=2025.12.18",
      "sandbox": true
    },
    "aws": {
      "blocked": true,  // Until CVE-2025-5277 patch verified
      "reason": "Command injection vulnerability"
    },
    "cimd": {
      "enabled": true,
      "verify_url": true
    }
  }
}
```

---

## Key Sources

### Official
- platform.claude.com/docs (Agent Skills)
- agentskills.best (Marketplace)
- derivative.ca (POPs Workshop)
- workos.com/blog (CIMD/XAA guides)

### Security
- SC Media (Anthropic patch announcement)
- The Hacker News (CVE disclosure)
- SentinelOne (CVE-2025-5277)
- Infosecurity Magazine (Cyata research)

### Community
- VoltAgent/awesome-claude-skills (GitHub)
- Function Store (POPs tutorials)
- Interactive & Immersive HQ (TD resources)

---

Last Updated: 2026-01-25
Cycle: 8
Enhancement Loop Status: ACTIVE
