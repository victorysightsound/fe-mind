# Remote MiniLM Embedding Backend

## Goal

Add a first-class remote MiniLM embedding backend to FeMind so production apps
can run `sentence-transformers/all-MiniLM-L6-v2` on a local-network GPU host
while keeping the same embedding profile as the existing local Candle backend.

This backend is not RecallBench-specific. RecallBench should consume it through
FeMind exactly the same way as production apps do.

## Why

FeMind and Memloft already center their local embedding story on the same MiniLM
family:

- model repo: `sentence-transformers/all-MiniLM-L6-v2`
- dimensions: `384`
- local-first storage

That makes remote execution a deployment concern, not a model-family change.

The target outcome is:

- local CPU MiniLM remains supported
- remote GPU MiniLM becomes a first-class option
- fallback to local remains available when the remote service is down
- Memloft can continue using existing vectors when the embedding profile is
  unchanged

The logical model label should stay stable across apps:

- canonical model label: `local-minilm`
- strict compatibility key: full `embedding_profile`

## Embedding Profile Contract

Compatibility is determined by profile identity, not just by the model label.

The remote backend must match the local MiniLM profile on:

- model repo
- dimensions
- tokenizer assets
- pooling strategy
- normalization behavior
- truncation policy
- preprocessing version

Hardware is not part of the profile identity. CPU and CUDA execution are the
same profile if they produce the same normalized vectors from the same model
assets and preprocessing rules.

Recommended profile key:

```text
local|sentence-transformers/all-MiniLM-L6-v2|384|v1|chars:none
```

FeMind should treat a remote GPU service as another execution mode for this
profile, not as a different embedding model.

## Memloft Compatibility Rule

Memloft and any future FeMind consumer should be able to keep using existing
stored vectors if the backend preserves the same MiniLM profile.

That requires:

- model identity match
- dimension match
- preprocessing/truncation match
- stable `model_name` / profile labeling between FeMind and Memloft

If the profile changes, vectors must be treated as a reindex target rather than
as cross-compatible data.

## Runtime Modes

FeMind should support these MiniLM runtime modes:

- `local-cpu`
  - use Candle on the local machine CPU
- `local-gpu`
  - use a GPU-capable local runtime on the same machine
- `remote-cpu`
  - use the remote embedding service hosted on another machine running CPU inference
- `remote-gpu`
  - use the remote embedding service hosted on another machine running GPU inference
- `remote-with-local-fallback`
  - prefer the remote service and fall back to local MiniLM when remote is unavailable
- `off`
  - disable vector generation and rely on FTS5-only retrieval

This mirrors the production behavior already proven in Librona without coupling
FeMind to Librona itself.

## Remote Service Shape

The remote service should stay narrow:

- `GET /health`
- `GET /status`
- `POST /embed`

The service only computes embeddings. It does not own the SQLite corpus, graph
logic, or retrieval stack.

The status payload should expose enough information to verify profile identity:

- model repo
- dimensions
- embedding profile
- execution mode
- local model asset fingerprint when available

The FeMind-hosted service should also support explicit host-device selection:

- `--device auto`
  - prefer CUDA when the build/runtime supports it, otherwise use CPU
- `--device cpu`
  - force CPU execution on the host
- `--device cuda`
  - require CUDA on the host and fail fast if unavailable
- `--cuda-ordinal <n>`
  - choose the CUDA device index when `--device cuda` is used

`--device cuda` requires a FeMind build that includes the `cuda` feature.
Without that feature, `auto` may still fall back to CPU, but explicit `cuda`
selection must fail fast instead of silently downgrading.

The FeMind-owned operator surface should expose:

- `femind-embed-service serve`
- `femind-embed-service status --config <path>`
- `femind-embed-service verify-remote --config <path>`

The host command should accept either direct flags or a TOML config file. The
client-side `status` and `verify-remote` commands should resolve:

- remote base URL
- auth token via environment or env file
- timeout
- local fallback preference
- profile verification preference

## Remote Host Pattern

The recommended deployment shape should mirror the proven Librona pattern:

- build `femind-embed-service` on the remote Linux, WSL, or native Windows host
- bind the service to `127.0.0.1`
- keep the process warm under `systemd`
- use a Windows scheduled task only to bring the WSL service up at logon or
  startup when the host is Windows-based
- native Windows hosts may also run `femind-embed-service.exe` directly through
  a scheduled task that prepares MSVC and CUDA environment variables first
- reach the service over SSH on top of ZeroTier instead of exposing the raw port
  directly on the network

Recommended defaults for lifecycle and operator behavior:

- run the host process warm under `systemd`
- use `Restart=always` and a short restart delay
- provide a Windows/WSL helper with explicit `off`, `status`, `logon`, and
  `startup` modes
- provide a native Windows helper with the same operator modes and explicit
  CUDA root configuration
- keep the service bound to loopback and authenticate with a bearer token
- keep idle CPU near zero; do not add background polling by default
- accept the warm-memory tradeoff so first-request latency stays low
- on native Windows CUDA hosts, keep toolkit and driver lines aligned
  (for example toolkit `12.9` with a `12.9` driver line)

This keeps idle CPU usage low while preserving fast first-request latency by
leaving the MiniLM model resident in memory.

FeMind should ship its own narrow service host for this contract so production
apps do not depend on benchmark-only tooling. The first shipped host may remain
CPU-backed, but the wire contract and runtime reporting should already support
later remote GPU execution on another machine.

## FeMind API Surface

New backend:

- `RemoteEmbeddingBackend`

New constructor pattern:

- `RemoteEmbeddingBackend::new(...)`
- `RemoteEmbeddingBackend::with_local_fallback(...)`

Expected behavior:

- verify remote profile on startup when enabled
- reject mismatched remote services with a clear error
- optionally degrade to local if configured
- report the same `model_name()` as the local MiniLM backend when the profile
  matches

Reference example configs:

- `examples/config/remote-embed-service.toml`
- `examples/config/remote-embedding-client.toml`
- `scripts/remote/configure-windows-native-autostart.ps1`

## Validation Requirements

Minimum validation before rollout:

1. Remote backend embeds successfully against a mock service
2. Profile mismatch is detected and surfaced clearly
3. Fallback-to-local works on transport failure
4. Remote backend reports `384` dimensions and the same model name/profile as
   the local backend
5. Existing vector-search tests still pass with the new backend present

## Non-Goals

- GPU-native Candle support inside FeMind itself
- LLM extraction offload
- benchmark-only adapters
- mixing vectors across different embedding profiles
