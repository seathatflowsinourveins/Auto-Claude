# System Architecture

## Layer Structure
1. Presentation (CLI/API)
2. Application (Use Cases)
3. Domain (Business Logic)
4. Infrastructure (External Services)

## Patterns Used
- Repository pattern for data access
- Circuit breaker for resilience
- Event sourcing for audit
- CQRS for complex queries

## Integration Points
- MCP servers for external tools
- Letta for memory persistence
- Qdrant for vector search
- Neo4j for knowledge graph
