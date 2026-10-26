# VoxCPM2 local TTS server

`llama-tts-server` exposes VoxCPM2 as a small, OpenAI-compatible HTTP service.
It loads the BaseLM + Acoustic GGUF pair once at startup and keeps the runtime
warm for the process lifetime, so per-request latency is inference time only.

## Build

```bash
cmake -S . -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release --target llama-tts-server
```

The binary is `build/bin/Release/llama-tts-server(.exe)`.

## Run

```bash
llama-tts-server \
  --voxcpm2-base-lm   /path/to/VoxCPM2-BaseLM-Q8_0.gguf \
  --voxcpm2-acoustic  /path/to/VoxCPM2-Acoustic-F16.gguf \
  --host 127.0.0.1 --port 19080
```

Optional:

| Flag | Meaning |
| --- | --- |
| `--voxcpm2-n-gpu-layers N` | Layers to offload (`-1` = all) |
| `-t/--threads N` | CPU threads |
| `--api-key KEY` | Require `Authorization: Bearer KEY` on every route |

The service binds loopback by default and is meant to be supervised by the
owning agent runtime, not exposed directly.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health`, `/v1/health` | Liveness + `sample_rate` / `n_gpu_layers` |
| `POST` | `/v1/audio/speech` | OpenAI-compatible synthesis, returns WAV or raw PCM |
| `POST` | `/v1/audio/speech/stream` | Chunked WAV streaming |
| `GET` | `/v1/audio/speech/models` | Loaded model info |
| `POST` | `/v1/voxcpm2/init` | Swap models at runtime (optional) |

### `POST /v1/audio/speech`

```json
{
  "model": "voxcpm2",
  "input": "要合成的文本",
  "response_format": "wav",
  "seed": 42
}
```

`input` is required and capped at 4000 characters. `response_format` accepts
`wav` (default) or `pcm` (float32 little-endian, no header).

Voice cloning is available through the optional `reference_audio` field: a
base64-encoded WAV. Adding `prompt_text` switches to continuation mode, which
uses the reference clip as the preceding speech context.

## Concurrency

The VoxCPM2 runtime keeps decode state and is not thread-safe. Every synthesis
request (including streaming) is serialized behind one mutex, and
`/v1/voxcpm2/init` takes the same lock, so a model swap never races a request.
