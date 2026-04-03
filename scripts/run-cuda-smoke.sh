#!/bin/zsh
set -euo pipefail

require_real_cuda() {
  if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    return 0
  fi

  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi

  cat <<'EOF' >&2
femind CUDA smoke requires a real CUDA toolkit.
Set CUDA_PATH or put nvcc on PATH, then rerun on a CUDA-capable host.
EOF
  exit 1
}

require_real_cuda

cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --features full,cuda
cargo doc --features full,cuda --no-deps
