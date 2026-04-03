# Releasing

This repo is released as `femind`.

## Canonical Validation

Run these commands in order for the standard local gate:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --features full
cargo doc --features full --no-deps
```

`scripts/run-release-gate.sh` wraps the same gate and can be used on CPU-only
hosts. It bootstraps a temporary fake CUDA toolkit so the full-feature crate
checks do not require a real GPU during local or CI validation.

`scripts/run-release-preflight.sh` runs the release gate and then performs a
`cargo publish --dry-run` package check.

Use `scripts/run-cuda-smoke.sh` on a real CUDA host to validate the native
`cuda` feature path. That script requires a real toolkit and does not fall back
to a fake CUDA stub.

The GitHub Actions workflow mirrors the same split:

- hosted lanes cover the release gate, default tests, full tests, and docs
- the CUDA smoke lane is manual and targets a self-hosted CUDA runner

## Before A Release

- Run `cargo fmt --check`
- Run `scripts/run-release-gate.sh`
- Run `scripts/run-release-preflight.sh`
- Run `scripts/run-practical-eval.sh` if the practical live-validation pass is in scope
- Run `scripts/run-cuda-smoke.sh` on a real CUDA host when that validation is in scope
- Confirm `cargo test --features full`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo doc --features full --no-deps` are green
- Review the changelog and version bump together

## Publish Flow

1. Confirm the working tree is clean except for the intended release changes.
2. Run the release gate.
3. Bump the version and update `CHANGELOG.md`.
4. Run a dry-run package check:
   ```bash
   cargo publish --dry-run
   ```
5. Publish the crate when the package audit is clean:
   ```bash
   cargo publish
   ```

## Runtime Notes

- `scripts/run-release-gate.sh` bootstraps a local CUDA fallback when the host does not have a real toolkit installed.
- If a real CUDA host is available, the same release gate can run there without the fallback path.
- The embed service still supports `--device cuda`; CPU-only release validation only avoids making CUDA availability a hard requirement.
