# ADR 0003: Default-deny, declarative YAML scope policy

Status: accepted

## Context
Scopes must map to routes, HTTP methods, MCP tool names and LLM models. Authorization decisions must be auditable and reviewable in version control.

## Decision
Policy is YAML: `upstreams` (what exists) and `rules` (who may do what). A rule has `scopes_any` / `scopes_all` and typed grants (`http`, `mcp`, `llm`). Everything not granted is denied; deny rules win over allow rules; allow rules must name a scope. Matching is glob-based on a normalized path, tool name and model id. The engine is pure and immutable. MCP `tools/call` is authorized per tool name and `tools/list` is filtered. Clients receive one uniform 403; reasons go to the audit stream.

## Consequences
- No policy language (OPA/Cedar) in v1; revisit if conditions on claims/attributes are needed.
- Authorization needs the request body for MCP/LLM, so those bodies are buffered up to `max_body_bytes`.
- Config errors (unknown upstream in a rule) fail at startup, not at request time.
