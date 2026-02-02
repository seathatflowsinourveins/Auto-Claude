# Testing Strategy

## Test Pyramid
- Unit tests: 70% (fast, isolated)
- Integration tests: 20% (component interaction)
- E2E tests: 10% (full workflows)

## Naming Convention
- test_<function>_<scenario>_<expected>
- Example: test_create_session_valid_input_returns_id

## Fixtures
- Use pytest fixtures for setup
- Scope: function (default), module, session
- Async fixtures with pytest-asyncio

## Mocking
- Mock external services
- Use responses for HTTP
- Use fakeredis for Redis
