#!/bin/zsh
set -euo pipefail

bootstrap_fake_cuda() {
  if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    return 0
  fi

  local cuda_root="${CUDA_PATH:-${TMPDIR:-/tmp}/femind-cuda-fake}"
  mkdir -p "$cuda_root/bin" "$cuda_root/include" "$cuda_root/lib64"

  if [[ ! -f "$cuda_root/include/cuda.h" ]]; then
    cat >"$cuda_root/include/cuda.h" <<'EOF'
#pragma once
EOF
  fi

  if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
    cat >"$cuda_root/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  echo "Cuda compilation tools, release 12.4, V12.4.0"
else
  echo "fake nvcc"
fi
exit 0
EOF
    chmod +x "$cuda_root/bin/nvcc"
  fi

  if [[ ! -x "$cuda_root/bin/nvidia-smi" ]]; then
    cat >"$cuda_root/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "NVIDIA-SMI 000.00.00"
exit 0
EOF
    chmod +x "$cuda_root/bin/nvidia-smi"
  fi

  export CUDA_PATH="$cuda_root"
  export PATH="$cuda_root/bin:$PATH"
  export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-86}"
  export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12040}"
}

bootstrap_fake_cuda

scripts/run-release-gate.sh "$@"
cargo publish --dry-run --allow-dirty
