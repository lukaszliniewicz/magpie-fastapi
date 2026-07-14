# Magpie FastAPI

An OpenAI-compatible FastAPI wrapper for NVIDIA Magpie TTS Multilingual, used by Pandrator.

The launcher manages a project-local Pixi environment on Windows and Linux. It binds to
`127.0.0.1:8030` by default.

```bash
# Linux
./run.sh --device cpu

# Windows
run.bat cpu

# Prepare dependencies without starting the service
python run.py --device cpu --prepare-only
```

Pass `--pixi-path /path/to/pixi` when using a portable or application-managed Pixi binary.
CUDA mode requires an NVIDIA GPU; when it is unavailable the launcher safely falls back to CPU.
