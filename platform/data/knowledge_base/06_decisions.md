# Key Decisions (ADRs)

## ADR-001: Use Pydantic for Data Models
- **Status**: Accepted
- **Context**: Need structured data with validation
- **Decision**: Use Pydantic v2 for all data classes
- **Rationale**: Type safety, serialization, IDE support

## ADR-002: Async-First Design
- **Status**: Accepted
- **Context**: Many I/O operations (MCP, HTTP, DB)
- **Decision**: Use async/await throughout
- **Rationale**: Better resource utilization, concurrency

## ADR-003: Windows Compatibility
- **Status**: Accepted
- **Context**: Development on Windows 11
- **Decision**: ASCII-safe output, Path handling
- **Rationale**: Consistent cross-platform behavior
