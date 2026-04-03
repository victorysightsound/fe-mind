# FeMind Production Hardening Spec

For the higher-level roadmap, see [PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md).

## Goal

Make FeMind production-ready across CPU-only and CUDA hosts, with a repeatable
validation pipeline, explicit service boundaries, and release checks that fail
before users do.

## Current Baseline

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
- explicit backend selection policy for preferred and secondary execution
- stronger telemetry and request-level observability
- wider hostile testing around outages, flapping, malformed inputs, and auth failures
- a fuller CI matrix with docs and release surfaces
- optional production deployment tooling for install/update/start/stop
- stricter release preflight and publication automation

## Non-Goals

- New user-facing product features unrelated to hardening
- Benchmark chasing without release value
- Live LLM runs before the gate is explicitly passed
- Replacing the current architecture with a new storage or search model

## Release Criteria

FeMind is considered production-ready only when all of the following are true:

- `cargo fmt --check` passes
- `cargo clippy --all-targets --all-features -- -D warnings` passes
- `cargo test --features full` passes
- `cargo doc --features full --no-deps` passes
- CUDA-capable hosts can build and run the CUDA path
- CPU-only hosts can still lint, build, and test the crate without manual setup
- Practical evaluation scenarios are repeatable and documented
- The embed service has auth, limits, timeout behavior, and structured logging
- Docs, specs, and implementation agree on CUDA/runtime behavior
- Release packaging and publish steps are documented and reproducible

## DIAL Tasking Rules

- Keep each task small enough to finish, validate, and review on its own.
- Prefer one write set per task.
- Add exact file paths in the task notes when the task is clearly scoped.
- If a task exposes a new class of failure, split the work and stop the phase.
- Do not bundle release automation with runtime behavior changes.
- Do not bundle evaluation changes with core service refactors unless the eval
  depends on the refactor.

## Phase 1: Validation and Toolchain Hardening

Purpose: make the build and validation path stable everywhere.

Dependencies:

- existing CPU-only release gate
- existing CUDA-capable host validation
- current README and release docs

Tasks:

1. Define the canonical validation commands.
   - Document the exact commands for format, lint, test, and docs.
   - Separate the default CPU-only path from the full-feature path.
   - Make sure the docs say when CUDA is required and when it is not.
   - Likely files: `README.md`, `RELEASING.md`, `specs/PRODUCTION_ROADMAP.md`.

2. Remove machine-specific assumptions from the normal repo workflow.
   - Identify anything that assumes a CUDA host, a local tunnel, or a local GPU.
   - Move those assumptions behind explicit feature flags or host-specific docs.
   - Keep the normal developer path usable on CPU-only machines.
   - Likely files: `Cargo.toml`, `README.md`, `scripts/*`.

3. Add or tighten environment detection and failure messages.
   - Detect missing CUDA/runtime prerequisites early.
   - Make the error text stable and actionable.
   - Ensure CPU-only hosts fail with a clear explanation only when they must.
   - Likely files: `src/*`, `scripts/*`.

4. Add a CI matrix for build validation.
   - Cover format, clippy, default tests, full tests, and docs.
   - Keep the CPU-only path green in hosted CI.
   - Add a separate CUDA-capable lane only where the runner exists.
   - Likely files: `.github/workflows/*`.

Acceptance criteria:

- the validation commands are documented in one place
- CPU-only validation works without manual setup
- CUDA assumptions no longer leak into the default workflow
- CI covers the same release gate used locally

## Phase 2: Runtime and Service Hardening

Purpose: make the service safe and predictable to run in production.

Dependencies:

- service startup path
- request/response surface for embedding and reranking
- current logging and status code paths

Tasks:

1. Add explicit request limits.
   - Set maximum request body sizes for embedding and reranking.
   - Reject oversized batches before heavy work starts.
   - Keep the limit values documented and easy to change.
   - Likely files: `src/service/*`, `src/transport/*`.

2. Add deterministic timeout behavior.
   - Set one timeout policy for remote requests.
   - Make timeout handling consistent across embed and rerank.
   - Convert timeout failures into stable service errors.
   - Likely files: `src/service/*`, `src/backend/*`.

3. Make startup and shutdown deterministic.
   - Define what happens when model load fails.
   - Define what happens during in-flight request shutdown.
   - Ensure the service does not depend on implicit OS timing.
   - Likely files: `src/bin/*`, `src/service/*`.

4. Add structured logs for production debugging.
   - Log model load, batch start/end, fallback, timeout, and recovery.
   - Include request IDs or another stable correlation key.
   - Keep log fields stable enough to search in production.
   - Likely files: `src/logging/*`, `src/service/*`.

5. Tighten error types and status reporting.
   - Separate auth, timeout, transport, profile mismatch, and unavailable errors.
   - Make `/status` or equivalent output show the active backend mode.
   - Avoid mixing transient and permanent failures in the same class.
   - Likely files: `src/error.rs`, `src/status.rs`, `src/service/*`.

Acceptance criteria:

- oversized requests are rejected early and consistently
- timeout failures produce stable, searchable errors
- logs show the active execution path and fallback events
- shutdown behavior is deterministic and documented

## Phase 3: Backend Policy and Failover

Purpose: define the backend-selection model once and use it everywhere.

Dependencies:

- local fallback wrappers
- remote backend paths
- service-level status reporting

Tasks:

1. Define the backend state machine.
   - States: `remote-primary`, `local-fallback`, `remote-recovering`, `offline`
   - Transitions: healthy, failed, timed out, recovered, profile-mismatch
   - Decide which failures are transient and which are hard errors.
   - Likely files: `src/backend/*`, `src/service/*`, `specs/*`.

2. Add an explicit backend manager abstraction.
   - Own the primary/fallback selection policy in one place.
   - Expose the current mode and last failure time.
   - Share the same policy object across embedding and reranking.
   - Likely files: `src/backend/*`.

3. Add health probes and recovery logic.
   - Probe the remote backend in the background.
   - Use backoff and cooldown to avoid oscillation.
   - Promote the remote backend automatically after a verified recovery.
   - Likely files: `src/backend/*`, `src/service/*`.

4. Make profile mismatches hard failures.
   - Treat wrong model, wrong dimensions, and wrong profile as permanent.
   - Do not retry a mismatch as if it were a transport glitch.
   - Surface the mismatch clearly in errors and status.
   - Likely files: `src/backend/*`, `src/error.rs`.

5. Surface active mode in status output.
   - Show the current backend, fallback availability, and recovery state.
   - Report the last verified healthy remote probe when available.
   - Keep the status shape consistent across embed and rerank.
   - Likely files: `src/status.rs`, `src/bin/*`.

Acceptance criteria:

- remote loss triggers local fallback without manual intervention
- remote recovery is detected automatically
- profile mismatch still fails fast
- embed and rerank follow the same policy model

## Phase 4: Evaluation and Regression Hardening

Purpose: prove the engine behaves correctly on real scenarios.

Dependencies:

- practical eval harness
- eval scenario catalog
- failure recording format

Tasks:

1. Add failover cases to practical eval.
   - remote loss during retrieval
   - fallback to local CPU
   - remote recovery and re-promotion
   - likely files: `eval/practical/*`, `PRACTICAL_EVAL.md`.

2. Add regression checks for CPU-only and remote fallback.
   - Verify the local-only path stays healthy.
   - Verify the remote path can fail over without a restart.
   - Verify the fallback path recovers cleanly.
   - Likely files: `eval/practical/*`, `tests/*`.

3. Add hostile tests for flapping and malformed input.
   - repeated short outages
   - oscillation suppression
   - cooldown validation
   - invalid JSON
   - wrong model
   - wrong profile
   - oversized request bodies
   - Likely files: `tests/*`, `src/backend/*`.

4. Standardize failure recording.
   - Record the failure reason in a stable format.
   - Separate transport, auth, mismatch, and recall gaps.
   - Make sure failed scenarios are easy to triage later.
   - Likely files: `eval/*`, `results/*`, `specs/*`.

5. Add smoke tests for the real hosts.
   - Calvary AV GPU host
   - CPU-only fallback host
   - reconnect after restart
   - Likely files: `scripts/*`, `eval/*`.

Acceptance criteria:

- reconnect behavior is proven, not assumed
- flapping does not cause unstable oscillation
- regressions fail with clear reasons
- failover behavior is represented in eval

## Phase 5: CI and Release Gate Expansion

Purpose: make the shared validation surface match production expectations.

Dependencies:

- Phase 1 validation commands
- current GitHub Actions setup
- release gate docs

Tasks:

1. Keep the local release gate canonical.
   - `cargo fmt --check`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --features full`
   - `cargo doc --features full --no-deps`
   - Make the docs point to the same commands the CI uses.

2. Expand CI coverage.
   - default features
   - full features
   - docs
   - format check
   - release gate
   - Likely files: `.github/workflows/*`.

3. Add a GPU validation lane where available.
   - build
   - status
   - embed
   - rerank
   - Calvary AV smoke test if the runner can reach it
   - Likely files: `.github/workflows/*`, `scripts/*`.

4. Keep CPU-only hosted CI green.
   - Ensure the hosted path does not require CUDA tooling.
   - Confirm that the default test matrix stays usable on shared runners.
   - Likely files: `.github/workflows/*`, `Cargo.toml`.

Acceptance criteria:

- CI catches docs and format drift
- default feature set stays healthy
- release gate and CI use the same checks
- the GPU lane is isolated from CPU-only validation

## Phase 6: Practical and Domain Coverage

Purpose: measure behavior on the workloads FeMind actually targets.

Dependencies:

- practical eval scenarios
- live-library eval
- memloft-slice eval

Tasks:

1. Add domain-relevant scenarios to practical eval.
   - agent memory
   - project docs
   - maintenance and operations
   - provenance-sensitive questions
   - Likely files: `eval/practical/*`.

2. Expand the live-library and memloft-slice follow-up layers.
   - Keep them as deeper validation rather than the first gate.
   - Make sure the scenarios are still representative of real use.
   - Likely files: `eval/live-library/*`, `eval/memloft-slice/*`.

3. Track pass rates by category.
   - exact
   - ANN
   - graph depth
   - stable summary
   - safety / abstention
   - Likely files: `eval/*`, `results/*`.

4. Keep failure reasons stable.
   - transport
   - profile mismatch
   - recall gap
   - hallucination or unsupported answer
   - Likely files: `eval/*`, `specs/*`.

5. Document the evaluation order.
   - practical eval first
   - live-library second
   - memloft-slice third
   - live LLM testing only after approval

Acceptance criteria:

- domain coverage is intentional, not accidental
- evaluation failures are attributable to a specific category
- the release gate remains the primary quality filter

## Phase 7: Release Automation and Packaging

Purpose: make shipping boring.

Dependencies:

- CI and release gate
- current versioning and changelog files
- publish workflow expectations

Tasks:

1. Define release preflight.
   - clean tree
   - fmt
   - docs
   - tests
   - release gate
   - dry-run publish
   - Likely files: `RELEASING.md`, `CHANGELOG.md`, `Cargo.toml`.

2. Automate version bump and changelog updates.
   - Decide how the version is bumped.
   - Keep the changelog format stable.
   - Avoid release notes that describe internal process language.
   - Likely files: `CHANGELOG.md`, `Cargo.toml`, `scripts/*`.

3. Document the publish path.
   - Decide whether the release is manual or tagged.
   - Write the exact publish steps in order.
   - Include verification steps after publish.
   - Likely files: `RELEASING.md`, `README.md`.

4. Remove machine-local artifacts from the repo and package.
   - keep generated outputs and host-specific files out of source control
   - keep the release bundle reproducible from a clean checkout
   - Likely files: `.gitignore`, release scripts, packaging scripts.

Acceptance criteria:

- release docs match the real commands
- package dry-run passes
- release artifacts are reproducible
- the repo stays clean of machine-local development artifacts

## Recommended DIAL Split

Suggested task order:

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

