# AGENTS.md Template: Cosmic Python DDD/TDD Strategy

Use this template to guide coding agents toward a consistent Domain-Driven Design + Test-Driven Development approach inspired by Cosmic Python.

## 1) Mission

Build software around the domain model first, then layer application orchestration and infrastructure around it. Keep business logic testable, explicit, and independent of frameworks.

## 2) Non-negotiable architecture rules

1. Domain model is the center.
   - Domain code has no dependency on web, ORM, queue, or framework modules.
   - Domain objects enforce invariants and business rules.
2. Use ports and adapters.
   - Treat the system as hexagonal architecture: domain + service layer in the center, adapters on the edges.
   - Repositories, Unit of Work, message bus, and external gateways are ports.
   - DB/HTTP/framework code are adapters behind those ports.
   - Dependencies point inward only: adapters depend on ports/domain, never the reverse.
   - Ports define stable interfaces that the domain/service layer depends on.
   - Adapters implement ports and translate external data/protocols into domain-safe types.
   - Public adapter methods must be declared on the corresponding port contract.
   - Service layer type annotations should target ports, not concrete adapter classes.
3. Service layer owns use-case orchestration.
   - Service handlers coordinate repositories/UoW/events.
   - Controllers/CLI/API handlers stay thin.
4. Prefer events for side effects.
   - Domain emits events.
   - Handlers react; avoid cross-module direct coupling.
5. Keep consistency boundaries explicit.
   - Define aggregate roots and transaction boundaries clearly.

## 3) Hexagonal architecture blueprint

Use this directional model for all implementation decisions:

- Inside the hexagon:
  - `domain/` (entities, value objects, domain services, domain events)
  - `service_layer/` (use-case handlers, orchestration)
  - `ports` (repository/UoW/gateway interfaces)
- Outside the hexagon:
  - `adapters/` for persistence, API/CLI, messaging, external services
  - Framework wiring/bootstrap/composition root

Boundary rules:

- Adapters translate between transport/persistence formats and domain language.
- No business invariants in adapters.
- Adapter failures are handled at boundaries and mapped to domain-safe errors.
- Keep adapter-specific models (ORM/DTO schemas) out of domain entities.
- Do not leak adapter-only APIs into service/domain usage; if it is used publicly, promote it to the port.

## 4) Required discovery artifacts before major implementation

When starting a non-trivial feature/domain:

- Create or update `UBIQUITOUS_LANGUAGE.md`
  - Canonical terms, definitions, aliases to avoid, ambiguities.
- Create or update `DOMAIN_STRUCTURE.md`
  - Bounded contexts, candidate aggregates, invariants, commands, events, integration seams.

If these docs exist, refine them incrementally instead of replacing wholesale.

## 5) Preferred implementation sequence (outside-in)

1. Define use case and acceptance behavior.
2. Write a failing test at the highest useful abstraction level.
3. Model or refine domain behavior to satisfy business rule.
4. Add/update service-layer handler.
5. Define or refine ports for required dependencies.
6. Add/update repository/UoW/message bus ports as needed.
7. Implement/adjust adapters (ORM/API/events) last.
8. Wire dependencies in bootstrap/composition root.
9. Refactor with tests green.

Do vertical slices (one behavior end-to-end) instead of horizontal batches.

## 6) Testing strategy (high gear / low gear)

Balance tests intentionally:

- High gear (fast, isolated):
  - Pure domain tests (entities/value objects/domain services)
  - Service-layer tests with fake repository/UoW
  - Port-contract tests using in-memory fakes where useful
  - Goal: most business logic coverage here
- Low gear (slower, integration):
  - Adapter integration tests vs real infrastructure (DB, broker, external APIs)
  - API/entrypoint integration tests for critical paths
  - Message bus and bootstrap wiring tests
- End-to-end:
  - Keep minimal; only core user journeys

Rules:

- Test behavior via public interfaces, not private implementation details.
- Prefer fakes at architectural boundaries over brittle mocks.
- Add regression test first for bugs.
- One failing test, minimal code, green, then refactor.

## 7) Suggested test layout

```text
tests/
  unit/
    domain/
    service_layer/
    ports/
  integration/
    adapters/
      repository/
      api/
      messagebus/
      external_services/
    bootstrap/
  e2e/
```

## 8) Domain modeling checklist

Before merging a feature, ensure:

- Bounded context ownership is clear.
- Aggregate root and invariants are explicit.
- Commands and events use domain language.
- Transaction boundary is intentional.
- External dependency access goes through a port.
- Inbound and outbound adapters are identified for each use case.

## 9) Agent behavior rules

When generating or editing code:

- Do not put business rules in controllers/routes/ORM models unless they are true infrastructure concerns.
- Do not import infrastructure modules into domain layer.
- Keep domain types and use-case names aligned with `UBIQUITOUS_LANGUAGE.md`.
- If terminology conflicts appear, update the glossary first, then code.
- If architecture tradeoffs are unclear, document options and pick the simplest design that preserves boundaries.
- Place dependency wiring in a composition root; avoid hidden globals.
- Prefer this implementation order for boundaries: define/update port -> implement adapter -> wire adapter through service.
- If you add a new adapter capability, update the port interface in the same change.

## 10) Definition of done for each significant change

- Ubiquitous language updated if terminology changed.
- Domain structure updated if boundaries/aggregates changed.
- Unit tests pass for domain and service behavior.
- Integration tests pass for affected adapters.
- Lint/type/test checks pass per repository standards.
- Design remains framework-independent at domain core.
- New/changed adapters respect port contracts and inward dependencies.

## 11) PR summary expectations

Each PR or handoff should state:

- Domain concept changed (in domain terms)
- Invariants added/updated
- Boundaries touched (context/aggregate/UoW/port/adapter)
- Test coverage added (unit/integration/e2e)
- Risks and follow-up decisions

## 12) Python environment and dependency management

- Use `uv` as the only project and package manager for this repository.
- Use `ruff` as the linter and `mypy` for type checking, both configured via `pyproject.toml`.
- Prefer `uv run <command>` for all local execution (tests, lint, type checks, scripts).
- Prefer `uv sync` for environment synchronization and lockfile-aware dependency installation.
- When adding dependencies, use `uv add <package>` (and `uv add --dev <package>` for dev-only tools).
- Do not introduce `pip`/`requirements.txt`-based workflows, Poetry, or Pipenv for this repo.

## 13) Commenting and docstring standard

- Every function and method should include a short docstring explaining what it does in domain terms.
- Prefer one to two lines in plain language; avoid restating the function name.
- Focus on behavior and intent (`why/what`), not line-by-line implementation details (`how`).
- Add argument/return detail only when it is non-obvious from type hints and naming.
- Keep comments current: update docstrings whenever behavior changes.
