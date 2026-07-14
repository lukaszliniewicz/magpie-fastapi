#!/usr/bin/env python3
"""Cross-platform Pixi bootstrapper for the Magpie FastAPI service."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


TORCH_VERSION = "2.10.0"
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"

PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent
DEFAULT_PIXI = PARENT_DIR / "bin" / ("pixi.exe" if os.name == "nt" else "pixi")

DEFAULT_PORT = 8030
DEFAULT_HOST = "127.0.0.1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("magpie-run")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Magpie TTS FastAPI bootstrapper")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host interface (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port number (default: {DEFAULT_PORT})")
    parser.add_argument("--device", "--backend", dest="device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=None, help="CPU thread count")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Do not require nvidia-smi for CUDA")
    parser.add_argument("--pixi-path", default=None, help="Path to an existing Pixi executable")
    parser.add_argument("--prepare-only", "--install-only", dest="prepare_only", action="store_true")
    parser.add_argument("--inside-pixi", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def configure_portable_environment():
    cache_root = PARENT_DIR / "cache"
    pixi_cache = PARENT_DIR / ".pixi-cache"
    temp_dir = pixi_cache / "tmp"

    for directory in (cache_root, pixi_cache, temp_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PIXI_CACHE_DIR", str(pixi_cache))
    os.environ.setdefault("RATTLER_CACHE_DIR", str(pixi_cache / "rattler"))
    os.environ.setdefault("PIP_CACHE_DIR", str(pixi_cache / "pip"))
    os.environ.setdefault("UV_CACHE_DIR", str(pixi_cache / "uv-cache"))
    os.environ.setdefault("TMP", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMPDIR", str(temp_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))


def run_command(command, *, check=True, capture_output=False):
    command = [str(part) for part in command]
    log.info("Running: %s", " ".join(command))
    return subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=check,
        capture_output=capture_output,
        text=capture_output,
    )


def resolve_pixi(pixi_path=None):
    candidates = []
    if pixi_path:
        candidates.append(Path(pixi_path))
    candidates.extend(
        (
            DEFAULT_PIXI,
            Path.home() / ".pixi" / "bin" / ("pixi.exe" if os.name == "nt" else "pixi"),
        )
    )
    path_pixi = shutil.which("pixi.exe" if os.name == "nt" else "pixi")
    if path_pixi:
        candidates.append(Path(path_pixi))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError("Pixi was not found. Install Pixi, place it under ../bin, or pass --pixi-path.")


def in_project_pixi_environment():
    env_root = (PROJECT_DIR / ".pixi" / "envs" / "default").resolve()
    try:
        return os.path.commonpath((str(Path(sys.executable).resolve()), str(env_root))) == str(env_root)
    except (OSError, ValueError):
        return False


def ensure_running_inside_pixi(args, argv):
    if args.inside_pixi or in_project_pixi_environment():
        return

    pixi = resolve_pixi(args.pixi_path)
    run_command([pixi, "install"])
    command = [pixi, "run", "python", str(PROJECT_DIR / "run.py"), "--inside-pixi", *argv]
    raise SystemExit(subprocess.call([str(part) for part in command], cwd=PROJECT_DIR))


def torch_runtime_ready(device):
    try:
        import torch
        import torchaudio  # noqa: F401
    except Exception:
        return False

    if torch.__version__.split("+", 1)[0] != TORCH_VERSION:
        return False
    if device == "cuda":
        return bool(torch.cuda.is_available())
    return getattr(torch.version, "cuda", None) is None


def install_torch(device):
    index_url = TORCH_CUDA_INDEX_URL if device == "cuda" else TORCH_CPU_INDEX_URL
    log.info("Installing matched PyTorch/TorchAudio %s packages for %s.", TORCH_VERSION, device)
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            f"torch=={TORCH_VERSION}",
            f"torchaudio=={TORCH_VERSION}",
            "--index-url",
            index_url,
        ]
    )


def resolve_device(args):
    device = args.device
    if device == "cuda" and not args.skip_gpu_check and shutil.which("nvidia-smi") is None:
        log.warning("nvidia-smi was not detected; falling back to the CPU runtime.")
        device = "cpu"

    if not torch_runtime_ready(device):
        install_torch(device)

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                log.warning("CUDA is unavailable after setup; installing and using the CPU runtime.")
                device = "cpu"
                if not torch_runtime_ready(device):
                    install_torch(device)
        except Exception as exc:
            log.warning("CUDA validation failed (%s); using CPU.", exc)
            device = "cpu"
            if not torch_runtime_ready(device):
                install_torch(device)

    return device


def configure_cpu_threads(requested_threads):
    detected = os.cpu_count() or 1
    threads = requested_threads if requested_threads is not None else max(1, min(detected, 8))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)

    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception as exc:
        log.warning("Could not apply PyTorch thread limits: %s", exc)

    log.info("Configured the CPU runtime for %d threads.", threads)


def validate_magpie_import():
    from nemo.collections.tts.models import MagpieTTSModel  # noqa: F401


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    os.chdir(PROJECT_DIR)
    configure_portable_environment()
    ensure_running_inside_pixi(args, argv)

    device = resolve_device(args)
    if device == "cpu":
        configure_cpu_threads(args.threads)

    validate_magpie_import()
    os.environ["MAGPIE_DEVICE"] = device

    if args.prepare_only:
        log.info("Magpie runtime is ready.")
        return

    import uvicorn

    log.info("Starting Magpie API on %s:%d with %s.", args.host, args.port, device)
    uvicorn.run("main:app", host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
