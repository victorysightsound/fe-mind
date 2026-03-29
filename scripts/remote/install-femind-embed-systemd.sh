#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "run as root" >&2
  exit 1
fi

REMOTE_USER="${REMOTE_USER:-remote}"
REMOTE_HOME="${REMOTE_HOME:-/home/${REMOTE_USER}}"
FEMIND_BUILD_DIR="${FEMIND_BUILD_DIR:-${REMOTE_HOME}/fe-mind-build}"
ENV_FILE="${ENV_FILE:-${REMOTE_HOME}/.femind-embedding-service.env}"
RUNNER_PATH="${RUNNER_PATH:-${REMOTE_HOME}/bin/femind-embed-service-runner.sh}"
SERVICE_NAME="${SERVICE_NAME:-femind-embed.service}"
SYSTEMCTL_BIN="${SYSTEMCTL_BIN:-$(command -v systemctl)}"
JOURNALCTL_BIN="${JOURNALCTL_BIN:-$(command -v journalctl)}"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
SUDOERS_PATH="${SUDOERS_PATH:-/etc/sudoers.d/${SERVICE_NAME%.service}}"
INSTALL_SUDOERS="${INSTALL_SUDOERS:-true}"
ENABLE_SERVICE="${ENABLE_SERVICE:-true}"
START_SERVICE="${START_SERVICE:-true}"
CUDA_BIN_DIR="${CUDA_BIN_DIR:-/usr/local/cuda/bin}"
WSL_GPU_LIB_DIR="${WSL_GPU_LIB_DIR:-/usr/lib/wsl/lib}"
ADDITIONAL_PATHS="${ADDITIONAL_PATHS:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
PREFIX="${PREFIX:-/embed}"
DEVICE="${DEVICE:-auto}"
CUDA_ORDINAL="${CUDA_ORDINAL:-0}"
AUTH_TOKEN_ENV_NAME="${AUTH_TOKEN_ENV_NAME:-FEMIND_EMBED_AUTH_TOKEN}"
REQUEST_TIMEOUT_SECS="${REQUEST_TIMEOUT_SECS:-60}"
MAX_BATCH_TEXTS="${MAX_BATCH_TEXTS:-32}"

if [[ -z "${SYSTEMCTL_BIN}" ]]; then
  echo "systemctl not found" >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing env file: ${ENV_FILE}" >&2
  exit 1
fi

if ! id -u "${REMOTE_USER}" >/dev/null 2>&1; then
  echo "missing user: ${REMOTE_USER}" >&2
  exit 1
fi

PATH_ENTRIES=()
maybe_add_path() {
  local candidate="${1:-}"
  if [[ -n "${candidate}" && -d "${candidate}" ]]; then
    PATH_ENTRIES+=("${candidate}")
  fi
}

maybe_add_path "${CUDA_BIN_DIR}"
maybe_add_path "${WSL_GPU_LIB_DIR}"
maybe_add_path "${REMOTE_HOME}/.cargo/bin"
IFS=':' read -r -a extra_paths <<<"${ADDITIONAL_PATHS}"
for extra_path in "${extra_paths[@]}"; do
  maybe_add_path "${extra_path}"
done
PATH_ENTRIES+=("/usr/local/sbin" "/usr/local/bin" "/usr/sbin" "/usr/bin" "/sbin" "/bin")
PATH_VALUE="$(IFS=:; printf '%s' "${PATH_ENTRIES[*]}")"

install -d -m 0755 -o "${REMOTE_USER}" -g "${REMOTE_USER}" \
  "${REMOTE_HOME}/bin"

cat >"${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH=${PATH_VALUE}
[ -f "${ENV_FILE}" ] && source "${ENV_FILE}"
exec "${FEMIND_BUILD_DIR}/target/release/femind-embed-service" \
  serve \
  --host "${HOST}" \
  --port "${PORT}" \
  --prefix "${PREFIX}" \
  --device "${DEVICE}" \
  --cuda-ordinal "${CUDA_ORDINAL}" \
  --request-timeout-secs "${REQUEST_TIMEOUT_SECS}" \
  --max-batch-texts "${MAX_BATCH_TEXTS}" \
  --auth-token-env "${AUTH_TOKEN_ENV_NAME}"
EOF
chown "${REMOTE_USER}:${REMOTE_USER}" "${RUNNER_PATH}"
chmod 0755 "${RUNNER_PATH}"

cat >"${SERVICE_PATH}" <<EOF
[Unit]
Description=FeMind embedding service
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${REMOTE_USER}
Group=${REMOTE_USER}
WorkingDirectory=${FEMIND_BUILD_DIR}
Environment=PATH=${PATH_VALUE}
ExecStart=${RUNNER_PATH}
Restart=always
RestartSec=2
TimeoutStartSec=120
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
chmod 0644 "${SERVICE_PATH}"

if [[ "${INSTALL_SUDOERS}" == "true" ]]; then
cat >"${SUDOERS_PATH}" <<EOF
${REMOTE_USER} ALL=(root) NOPASSWD: ${SYSTEMCTL_BIN} start ${SERVICE_NAME}, ${SYSTEMCTL_BIN} stop ${SERVICE_NAME}, ${SYSTEMCTL_BIN} restart ${SERVICE_NAME}, ${SYSTEMCTL_BIN} status ${SERVICE_NAME}, ${SYSTEMCTL_BIN} is-active ${SERVICE_NAME}, ${SYSTEMCTL_BIN} enable ${SERVICE_NAME}, ${SYSTEMCTL_BIN} disable ${SERVICE_NAME}, ${JOURNALCTL_BIN} -u ${SERVICE_NAME}
EOF
chmod 0440 "${SUDOERS_PATH}"
visudo -cf "${SUDOERS_PATH}" >/dev/null
fi

"${SYSTEMCTL_BIN}" daemon-reload
if [[ "${ENABLE_SERVICE}" == "true" ]]; then
  "${SYSTEMCTL_BIN}" enable "${SERVICE_NAME}" >/dev/null
fi
if [[ "${START_SERVICE}" == "true" ]]; then
  "${SYSTEMCTL_BIN}" restart "${SERVICE_NAME}"
fi
"${SYSTEMCTL_BIN}" --no-pager --full status "${SERVICE_NAME}" | sed -n '1,20p'
