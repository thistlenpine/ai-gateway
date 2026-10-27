"""Declarative scope policy engine. OWNER: policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ai_gateway.config import PolicyConfig, Rule
from ai_gateway.models import (
    DecisionReason,
    PolicyDecision,
    Principal,
    Resource,
    RouteMatch,
    UpstreamKind,
)
from ai_gateway.policy.matchers import name_match, path_match, scopes_satisfied

_MCP_CALL = "tools/call"


@dataclass(frozen=True, slots=True)
class _Grant:
    kind: UpstreamKind
    upstream: str
    methods: frozenset[str] = frozenset()  # HTTP: upper-cased
    patterns: tuple[str, ...] = ()  # HTTP path globs | MCP tool globs | LLM model globs
    mcp_methods: tuple[str, ...] = ()
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class _Rule:
    id: str
    deny: bool
    any_of: frozenset[str]
    all_of: frozenset[str]
    grants: tuple[_Grant, ...]


@dataclass(frozen=True, slots=True)
class _Route:
    upstream: str
    kind: UpstreamKind
    prefix: str
    segments: tuple[str, ...]


def _is_clean(path: str) -> bool:
    """Structural check that `path` looks normalized (we never decode here)."""
    if not path.startswith("/") or "\x00" in path:
        return False
    if path == "/":
        return True
    return all(s not in ("", ".", "..") for s in path[1:].split("/"))


class PolicyEngine:
    """Immutable after construction (reload = build a new engine and swap the reference).

    Semantics:
      * resolve_route: longest-prefix match on path segments over PolicyConfig.upstreams;
        '/billing' must not match '/billingx'. No match -> None.
      * authorize: DEFAULT DENY. Evaluate every rule whose scope condition the principal
        satisfies (deny rules with no scopes apply to all). Any matching deny grant ->
        EXPLICIT_DENY (wins). Else any matching allow grant -> ALLOWED (rule_id, matched
        scopes, LLM max_tokens cap). Else if some rule would match the resource but scopes
        are insufficient -> INSUFFICIENT_SCOPE with missing_scopes_hint; else NO_MATCHING_RULE.
      * Grant matching: HTTP: upstream equal, method in grant.methods (upper-case compare),
        any path glob matches. MCP: upstream equal; if resource.tool is set (tools/call)
        the tool must match grant.tools, otherwise resource.mcp_method must match grant.methods
        (globs); for GET/DELETE on the MCP endpoint (mcp_method None) require any MCP grant for
        the upstream. LLM: upstream equal and resource.model matches grant.models (a request
        without a model is denied).
      * A request with several Resources (MCP batch) is allowed only if every one is allowed
        (use authorize_all).

    Additional fail-closed behaviour: a resource whose upstream is unknown -> NO_ROUTE; whose
    kind differs from its upstream's, or whose path is not normalized, or that cannot be
    classified (MCP tool without tools/call, POST to MCP without a method, LLM without model)
    -> MALFORMED_REQUEST/NO_MATCHING_RULE. An allow rule without scopes never matches. When
    several allow grants match, the first rule (config order) is reported as rule_id and the
    LLM token cap is the most permissive of them (None = uncapped).
    """

    def __init__(self, policy: PolicyConfig) -> None:
        # Snapshot into immutable structures so later mutation of `policy` cannot alter decisions.
        self._kinds: dict[str, UpstreamKind] = {n: u.kind for n, u in policy.upstreams.items()}
        routes = [
            _Route(n, u.kind, u.prefix, () if u.prefix == "/" else tuple(u.prefix[1:].split("/")))
            for n, u in policy.upstreams.items()
        ]
        routes.sort(key=lambda r: (-len(r.segments), r.upstream))
        self._routes: tuple[_Route, ...] = tuple(routes)
        self._rules: tuple[_Rule, ...] = tuple(self._compile(r) for r in policy.rules)

    @staticmethod
    def _compile(rule: Rule) -> _Rule:
        grants: list[_Grant] = []
        for h in rule.http:
            grants.append(
                _Grant(
                    UpstreamKind.HTTP,
                    h.upstream,
                    methods=frozenset(m.upper() for m in h.methods),
                    patterns=tuple(h.paths),
                )
            )
        for m in rule.mcp:
            grants.append(
                _Grant(UpstreamKind.MCP, m.upstream, patterns=tuple(m.tools), mcp_methods=tuple(m.methods))
            )
        for ll in rule.llm:
            grants.append(
                _Grant(
                    UpstreamKind.LLM,
                    ll.upstream,
                    patterns=tuple(ll.models),
                    max_tokens=ll.max_tokens_per_request,
                )
            )
        return _Rule(
            rule.id,
            rule.effect == "deny",
            frozenset(rule.scopes_any),
            frozenset(rule.scopes_all),
            tuple(grants),
        )

    # ------------------------------------------------------------------ routing

    def resolve_route(self, path: str) -> RouteMatch | None:
        if not _is_clean(path):
            return None
        segs = [] if path == "/" else path[1:].split("/")
        for r in self._routes:  # sorted most specific first
            n = len(r.segments)
            if tuple(segs[:n]) == r.segments:
                rel = "/" + "/".join(segs[n:])
                return RouteMatch(upstream=r.upstream, kind=r.kind, prefix=r.prefix, relative_path=rel)
        return None

    # ------------------------------------------------------------------ authorization

    @staticmethod
    def _grant_matches(g: _Grant, res: Resource) -> bool:
        if g.kind is not res.kind or g.upstream != res.upstream:
            return False
        if g.kind is UpstreamKind.HTTP:
            return res.method.upper() in g.methods and any(path_match(p, res.path) for p in g.patterns)
        if g.kind is UpstreamKind.MCP:
            if res.tool is not None:
                return any(name_match(p, res.tool) for p in g.patterns)
            if res.mcp_method is not None:
                return any(name_match(p, res.mcp_method) for p in g.mcp_methods)
            return True  # GET/DELETE on the endpoint: any MCP grant for the upstream
        assert res.model is not None
        return any(name_match(p, res.model) for p in g.patterns)

    def _precheck(self, res: Resource) -> PolicyDecision | None:
        kind = self._kinds.get(res.upstream)
        if kind is None:
            return PolicyDecision(False, DecisionReason.NO_ROUTE, res)
        if kind is not res.kind:
            return PolicyDecision(False, DecisionReason.NO_MATCHING_RULE, res)
        if not _is_clean(res.path):
            return PolicyDecision(False, DecisionReason.MALFORMED_REQUEST, res)
        malformed = False
        if kind is UpstreamKind.MCP:
            if res.tool is not None:
                malformed = res.tool == "" or res.mcp_method != _MCP_CALL
            elif res.mcp_method is not None:
                malformed = res.mcp_method == "" or res.mcp_method == _MCP_CALL
            else:
                malformed = res.method.upper() not in ("GET", "DELETE")
        elif kind is UpstreamKind.LLM:
            malformed = not res.model
        return PolicyDecision(False, DecisionReason.MALFORMED_REQUEST, res) if malformed else None

    def authorize(self, principal: Principal, resource: Resource) -> PolicyDecision:
        early = self._precheck(resource)
        if early is not None:
            return early

        granted = principal.scopes
        allows: list[tuple[_Rule, _Grant]] = []
        insufficient: set[str] = set()
        saw_insufficient = False
        for rule in self._rules:
            hit = next((g for g in rule.grants if self._grant_matches(g, resource)), None)
            if hit is None:
                continue
            unscoped = not rule.any_of and not rule.all_of
            if rule.deny:
                if unscoped or scopes_satisfied(granted, any_of=rule.any_of, all_of=rule.all_of):
                    return PolicyDecision(
                        False,
                        DecisionReason.EXPLICIT_DENY,
                        resource,
                        rule_id=rule.id,
                        matched_scopes=(rule.any_of | rule.all_of) & granted,
                    )
                continue
            if scopes_satisfied(granted, any_of=rule.any_of, all_of=rule.all_of):
                allows.append((rule, hit))
            elif not unscoped:
                saw_insufficient = True
                insufficient |= rule.all_of - granted
                if rule.any_of and not (rule.any_of & granted):
                    insufficient |= rule.any_of

        if allows:
            first = allows[0][0]
            matched: set[str] = set()
            caps: list[int | None] = []
            for rule, grant in allows:
                matched |= (rule.any_of | rule.all_of) & granted
                if grant.kind is UpstreamKind.LLM:
                    caps.append(grant.max_tokens)
            cap = None if (not caps or None in caps) else max(c for c in caps if c is not None)
            return PolicyDecision(
                True,
                DecisionReason.ALLOWED,
                resource,
                rule_id=first.id,
                matched_scopes=frozenset(matched),
                max_tokens_per_request=cap,
            )
        if saw_insufficient:
            return PolicyDecision(
                False,
                DecisionReason.INSUFFICIENT_SCOPE,
                resource,
                missing_scopes_hint=tuple(sorted(insufficient)),
            )
        return PolicyDecision(False, DecisionReason.NO_MATCHING_RULE, resource)

    def has_any_grant(self, principal: Principal, upstream: str) -> bool:
        """Cheap scope-only test (no body needed): does some allow rule whose scope condition the
        principal satisfies carry a grant for `upstream`? False means every request to it is
        denied, so the pipeline can answer the uniform 403 before buffering/parsing a body."""
        for rule in self._rules:
            if rule.deny or not any(g.upstream == upstream for g in rule.grants):
                continue
            if scopes_satisfied(principal.scopes, any_of=rule.any_of, all_of=rule.all_of):
                return True
        return False

    def authorize_all(self, principal: Principal, resources: list[Resource]) -> PolicyDecision:
        """First denying decision, else the (last) allow decision. Empty list -> deny."""
        if not resources:
            return PolicyDecision(False, DecisionReason.MALFORMED_REQUEST)
        last: PolicyDecision | None = None
        for res in resources:
            last = self.authorize(principal, res)
            if not last.allowed:
                return last
        assert last is not None
        return last

    def tool_filter(self, principal: Principal, upstream: str) -> Callable[[str], bool]:
        """Predicate used to filter MCP `tools/list` responses: True if principal may call it."""

        def allowed(tool: str) -> bool:
            if not isinstance(tool, str) or not tool:
                return False
            res = Resource(UpstreamKind.MCP, upstream, "POST", "/", mcp_method=_MCP_CALL, tool=tool)
            return self.authorize(principal, res).allowed

        return allowed
