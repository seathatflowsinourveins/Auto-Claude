# MCP Security Findings - January 2026

## Critical Vulnerabilities Discovered

### Git MCP Server Vulnerabilities (CRITICAL)

**CVE-2025-68145**: Remote Code Execution via malicious repo
- Attacker can exploit git commands to execute arbitrary code
- Affects all Git MCP server versions prior to patched release

**CVE-2025-68143**: Path Traversal Vulnerability
- Allows reading files outside designated repository
- Chain with filesystem MCP for full system access

**CVE-2025-68144**: Command Injection via Branch Names
- Malicious branch names can inject shell commands
- Requires interaction with untrusted repositories

### Chain Attack Pattern (CRITICAL)
```
Git MCP + Filesystem MCP = Remote Code Execution
```
1. Git MCP fetches malicious repo
2. Malicious content writes to accessible path
3. Filesystem MCP reads/executes the content

## Mitigation Strategies

### For All Projects
1. **Sandbox MCP servers**: Run in isolated containers
2. **Whitelist repos**: Only allow known-safe repositories
3. **Disable auto-fetch**: Require explicit user approval
4. **Audit MCP chains**: Review multi-server workflows

### Project-Specific Mitigations

#### WITNESS (Creative)
- TouchDesigner MCP: LOW RISK (local only)
- ComfyUI MCP: LOW RISK (local workflows)
- Qdrant MCP: MEDIUM RISK (sanitize vector metadata)

#### TRADING (AlphaForge)
- CRITICAL: No MCP in production trading path
- Git MCP: Whitelist only internal repos
- Financial data MCPs: Read-only, no write access
- Implement audit logging for all MCP calls

#### UNLEASH (Meta-Project)
- All MCPs enabled for research
- Use sandboxed Claude instances for untrusted sources
- Implement security-reviewer agent for MCP workflows

## MCP Security Best Practices (2026)

### 1. Least Privilege Principle
```json
{
  "mcp_servers": {
    "git": {
      "allowed_operations": ["clone", "status", "diff"],
      "denied_operations": ["push", "fetch --all"]
    }
  }
}
```

### 2. Input Sanitization
```python
def sanitize_mcp_input(input_data: dict) -> dict:
    """Sanitize before passing to MCP server."""
    # Remove shell metacharacters
    for key, value in input_data.items():
        if isinstance(value, str):
            input_data[key] = re.sub(r'[;&|`$]', '', value)
    return input_data
```

### 3. Output Validation
```python
def validate_mcp_output(output: dict, expected_schema: dict) -> bool:
    """Validate MCP output matches expected schema."""
    try:
        jsonschema.validate(output, expected_schema)
        return True
    except jsonschema.ValidationError:
        log.warning("MCP output failed validation", output=output)
        return False
```

### 4. Chain Isolation
- Never chain Git MCP with Filesystem MCP
- Use separate containers for high-risk MCPs
- Implement request signing between MCP servers

## MCP Ecosystem Stats (January 2026)
- 77,000+ stars on modelcontextprotocol/servers
- 200+ community MCP servers
- First major security audit completed
- "State of MCP Server Security 2025" report published

## Integration Requirements

### Pre-MCP Call Checklist
- [ ] Is the MCP server from a trusted source?
- [ ] Are inputs properly sanitized?
- [ ] Is output validated against expected schema?
- [ ] Are dangerous operation chains avoided?
- [ ] Is audit logging enabled?

### Post-Incident Response
1. Disable affected MCP server immediately
2. Review audit logs for exploitation
3. Update to patched version
4. Re-enable with additional restrictions

---
## NEW CVEs DISCOVERED (Cycle 8 Update - Jan 25, 2026)

### CVE-2025-5277: aws-mcp-server Command Injection
- **Severity**: Critical
- **Impact**: RCE via crafted prompts
- **Status**: Patch pending verification
- **Action**: BLOCK aws-mcp-server until patched

### Patch Status (Jan 25, 2026)
| CVE | Server | Status | Safe Version |
|-----|--------|--------|--------------|
| CVE-2025-68145 | mcp-server-git | ✅ PATCHED | >= Dec 18, 2025 |
| CVE-2025-68143 | mcp-server-git | ✅ PATCHED | >= Dec 18, 2025 |
| CVE-2025-68144 | mcp-server-git | ✅ PATCHED | >= Dec 18, 2025 |
| CVE-2025-5277 | aws-mcp-server | ⚠️ VERIFY | Check latest |

### Immediate Actions Required
1. `pip install --upgrade mcp-server-git>=2025.12.18`
2. Block aws-mcp-server in TRADING project
3. Implement CIMD for client verification
4. Enable XAA for enterprise IdP routing

---

Last Updated: 2026-01-25 (Cycle 8 Update)