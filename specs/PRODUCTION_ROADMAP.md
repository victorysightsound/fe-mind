# FeMind Production Roadmap

## Purpose

This roadmap is the short-form companion to
[PRODUCTION_HARDENING.md](./PRODUCTION_HARDENING.md). It keeps the work split
into DIAL-friendly phases while preserving the higher-level intent: seamless
remote embedding/reranking when available, automatic fallback when a host or
network drops, and automatic recovery when the preferred host returns.

The detailed task breakdown, file targets, and acceptance criteria live in the
hardening spec. This roadmap keeps the phase order visible at a glance.

## Current State

Already in place:

- local MiniLM embedding and reranking backends
- remote embedding and reranking backends
- local fallback wrappers for both embedding and reranking
- a CPU-only release gate
- a validated real CUDA host path on Calvary AV
- practical eval, live-library eval, and memloft-slice eval suites
- documentation for release and validation flow

Still missing or incomplete:

- seamless reconnect behavior after a remote host returns
- explicit backend selection policy for preferred/secondary execution
- stronger telemetry and request-level observability
- wider hostile testing around outages, flapping, and auth failures
- a fuller CI matrix with docs and release surfaces
- optional production deployment tooling for install/update/start/stop
- stricter release preflight and publication automation

## Product Goal

FeMind should behave like this:

- prefer remote GPU or remote CPU inference when configured
- fall back to local CPU immediately when the remote path fails
- keep retrying the preferred remote host in the background
- switch back to the preferred host automatically once it is healthy again
- preserve the same embedding and reranking profile across CPU and GPU hosts
- fail fast when the configured profile or dimensions do not match

The default behavior should be seamless, not manually toggled.
Per-request overrides can exist, but the normal mode should be automatic.

## Behavior Contract

### Remote Preference

When a remote backend is configured and healthy:

- use the remote backend for requests
- keep the local backend idle as backup unless explicitly configured otherwise
- report the active execution mode clearly in `/status`

### Remote Failure

When the remote backend fails:

- switch to local fallback immediately if available
- log the failure reason once per transition window
- keep serving requests without waiting for manual intervention

### Recovery

When the remote backend becomes healthy again:

- probe it in the background
- promote it back to primary automatically after a successful verification
- avoid oscillation by using hysteresis or a short cooldown window

### Failure Policy

The system should distinguish:

- transient transport failures
- auth failures
- profile mismatches
- timeout failures
- real service crashes

Only transient failures should trigger reconnect attempts. Profile mismatch
must remain a hard error.

## Design Principles

- Local-first remains the default deployment model.
- Remote-first with fallback should be explicit, not implicit magic.
- Recovery should be automatic, but not noisy.
- Profile identity matters more than the transport or host.
- CPU-only hosts must still pass the release gate without CUDA installed.
- GPU hosts must still be verified on real hardware, not just with shims.

## Phase 1: Validation and Toolchain Hardening

Purpose: make the build and validation path stable everywhere.

Goals:

- canonical validation commands
- CPU-only and CUDA-capable workflow separation
- clear environment detection and failure messages
- CI parity with the release gate

Tasks:

1. Document the canonical validation commands and host expectations.
   - Canonical local gate:
     - `cargo fmt --check`
     - `cargo clippy --all-targets --all-features -- -D warnings`
     - `cargo test --features full`
     - `cargo doc --features full --no-deps`
   - CPU-only hosts should use `scripts/run-release-gate.sh` without a real GPU.
   - CUDA is required only for real device smoke tests and GPU host validation.
   - Likely files: `README.md`, `RELEASING.md`.
2. Remove machine-specific CUDA assumptions from the default workflow.
   - Identify anything that assumes a CUDA host, a local tunnel, or a local GPU.
   - Move those assumptions behind explicit feature flags or host-specific docs.
   - Keep the normal developer path usable on CPU-only machines.
   - Likely files: `Cargo.toml`, `README.md`, `scripts/*`.
3. Tighten environment detection and failure messages.
   - Detect missing CUDA/runtime prerequisites early.
   - Make the error text stable and actionable.
   - Ensure CPU-only hosts fail with a clear explanation only when they must.
   - Likely files: `src/*`, `scripts/*`.
4. Add a CI matrix for build validation.
   - Cover format, clippy, default tests, full tests, and docs.
   - Keep the CPU-only path green in hosted CI.
   - Add a separate CUDA-capable lane only where the runner exists.
   - Likely files: `.github/workflows/*`.

## Phase 2: Runtime and Service Hardening

Purpose: make the service safe and predictable to run in production.

Goals:

- request limits
- deterministic timeout handling
- deterministic startup and shutdown
- structured logs and stable error classes

Tasks:

1. Add request limits and early rejection for oversized bodies.
2. Add deterministic timeout behavior across embed and rerank.
3. Make startup and shutdown deterministic.
4. Add structured logs and stable status reporting.

## Phase 3: Backend Policy and Failover

Purpose: define the backend-selection model once and use it everywhere.

Goals:

- single backend-selection state machine
- shared embedding/reranking policy
- health-aware reconnect
- stable status reporting

Tasks:

1. Define the backend state machine.
2. Add health probes and cooldown logic.
3. Add an explicit backend manager abstraction.
4. Make embedding and reranking use the same policy shape.
5. Define status payloads for active/fallback/recovery state.

## Phase 4: Evaluation and Regression Hardening

Purpose: prove the engine behaves correctly on real scenarios.

Goals:

- practical eval failover coverage
- hostile tests for malformed input and flapping
- stable failure recording
- real-host smoke coverage

Tasks:

1. Add failover cases to practical eval.
2. Add regression checks for CPU-only and remote fallback.
3. Add hostile tests for flapping and malformed input.
4. Standardize failure recording.
5. Add smoke tests for the real hosts.

## Phase 5: CI and Release Gate Expansion

Purpose: make the shared validation surface match production expectations.

Goals:

- local release gate canonical
- CI parity with local checks
- isolated GPU validation lane
- CPU-only hosted CI stays green

Tasks:

1. Keep the local release gate canonical.
2. Expand CI coverage.
3. Add a GPU validation lane where available.
4. Keep CPU-only hosted CI green.

## Phase 6: Practical and Domain Coverage

Purpose: measure behavior on the workloads FeMind actually targets.

Goals:

- practical eval remains the primary release gate
- live-library and memloft-slice remain deeper follow-up layers
- domain coverage grows intentionally
- failure reasons remain stable

Tasks:

1. Add domain-relevant scenarios to practical eval.
2. Expand the live-library and memloft-slice follow-up layers.
3. Track pass rates by category.
4. Keep failure reasons stable.
5. Document the evaluation order.

## Phase 7: Release Automation and Packaging

Purpose: make shipping boring.

Goals:

- release preflight
- version bump and changelog updates
- publish path documentation
- clean package artifacts

Tasks:

1. Define release preflight.
2. Automate version bump and changelog updates.
3. Document the publish path.
4. Remove machine-local artifacts from the repo and package.

## Recommended DIAL Split

Suggested DIAL phases:

1. Validation and toolchain hardening
2. Runtime and service hardening
3. Backend policy and failover
4. Evaluation and regression hardening
5. CI and release gate expansion
6. Practical and domain coverage
7. Release automation and packaging

## Stop Conditions

Stop and split a new spec if any phase reveals:

- an incompatible service contract
- a new deployment class not covered here
- a major eval category gap
- a release problem that needs process changes rather than code
