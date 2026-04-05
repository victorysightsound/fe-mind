#!/usr/bin/env bash
set -euo pipefail

bootstrap_fake_cuda() {
  if [[ -n "${CUDA_PATH:-}" && -x "${CUDA_PATH}/bin/nvcc" ]]; then
    return 0
  fi

  local cuda_root="${CUDA_PATH:-${TMPDIR:-/tmp}/femind-cuda-fake}"
  mkdir -p "$cuda_root/bin" "$cuda_root/include" "$cuda_root/lib64"
  mkdir -p "$cuda_root/lib64/stubs"

  if [[ ! -f "$cuda_root/include/cuda.h" ]]; then
    cat >"$cuda_root/include/cuda.h" <<'EOF'
#pragma once
EOF
  fi

  cat >"$cuda_root/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--version" ]]; then
  cat <<'OUT'
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jan_01_00:00:00_PST_1970
Cuda compilation tools, release 12.4, V12.4.0
Build cuda_12.4.r12.4/compiler.fake
OUT
  exit 0
fi

if [[ "${1:-}" == "--list-gpu-code" ]]; then
  cat <<'OUT'
sm_50
sm_60
sm_70
sm_75
sm_80
sm_86
sm_89
compute_50
compute_60
compute_70
compute_75
compute_80
compute_86
compute_89
OUT
  exit 0
fi

mode="noop"
out_file=""
out_dir=""
inputs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ptx)
      mode="ptx"
      shift
      ;;
    -c)
      mode="obj"
      shift
      ;;
    --lib)
      mode="lib"
      shift
      ;;
    -o)
      out_file="${2:-}"
      shift 2
      ;;
    --output-directory)
      out_dir="${2:-}"
      shift 2
      ;;
    *.cu|*.o)
      inputs+=("$1")
      shift
      ;;
    *)
      shift
      ;;
  esac
done

make_temp_c() {
  local path="${TMPDIR:-/tmp}/femind-fake-cuda-$$-$RANDOM.c"
  cat >"$path" <<'OUT'
void __femind_fake_cuda_symbol(void) {}
OUT
  printf '%s\n' "$path"
}

case "$mode" in
  ptx)
    for input in "${inputs[@]}"; do
      local_base="$(basename "${input%.cu}")"
      local_dir="${out_dir:-$(dirname "$input")}"
      mkdir -p "$local_dir"
      cat >"$local_dir/$local_base.ptx" <<OUT
.version 7.0
.target sm_${CUDA_COMPUTE_CAP:-86}
.address_size 64
.visible .entry ${local_base}() {
  ret;
}
OUT
    done
    ;;
  obj)
    if [[ -z "$out_file" ]]; then
      echo "fake nvcc missing -o for object build" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$out_file")"
    temp_c="$(make_temp_c)"
    cc -c "$temp_c" -o "$out_file"
    ;;
  lib)
    if [[ -z "$out_file" ]]; then
      echo "fake nvcc missing -o for archive build" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$out_file")"
    if [[ "${#inputs[@]}" -eq 0 ]]; then
      temp_c="$(make_temp_c)"
      temp_o="${out_file%.a}.o"
      cc -c "$temp_c" -o "$temp_o"
      inputs+=("$temp_o")
    fi
    ar rcs "$out_file" "${inputs[@]}"
    ;;
  *)
    echo "fake nvcc"
    ;;
esac
EOF
  chmod +x "$cuda_root/bin/nvcc"

  cat >"$cuda_root/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "--query-gpu=compute_cap" ]]; then
  cat <<'OUT'
compute_cap
8.6
OUT
else
  echo "NVIDIA-SMI 000.00.00"
fi
exit 0
EOF
  chmod +x "$cuda_root/bin/nvidia-smi"

  make_stub_cuda_lib() {
    local lib_path="$1"
    if [[ -f "$lib_path" ]]; then
      return 0
    fi

    local stub_source="${TMPDIR:-/tmp}/femind-fake-cuda-lib-$$-$RANDOM.c"
    cat >"$stub_source" <<'EOF'
void __femind_fake_cuda_library_symbol(void) {}
EOF
    cc -shared -fPIC "$stub_source" -o "$lib_path"
  }

  make_stub_cuda_lib "$cuda_root/lib64/libcudart.so"
  make_stub_cuda_lib "$cuda_root/lib64/libcuda.so"
  make_stub_cuda_lib "$cuda_root/lib64/stubs/libcudart.so"
  make_stub_cuda_lib "$cuda_root/lib64/stubs/libcuda.so"

  export CUDA_PATH="$cuda_root"
  export PATH="$cuda_root/bin:$PATH"
  export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-86}"
  export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12040}"
}

bootstrap_fake_cuda

scripts/run-release-gate.sh "$@"
cargo publish --dry-run --allow-dirty
