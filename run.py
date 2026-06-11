import os
import sys
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("magpie-run")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PIXI_FILE = os.path.join(PROJECT_DIR, "pyproject.toml")
PIXI_LOCK = os.path.join(PROJECT_DIR, "pixi.lock")
ENV_MARKER = os.path.join(PROJECT_DIR, ".pixi", "envs", "default", "conda-meta")

DEFAULT_PORT = 8030
DEFAULT_HOST = "0.0.0.0"


def is_pixi_installed() -> bool:
    try:
        subprocess.run(
            ["pixi", "--version"],
            capture_output=True,
            check=True,
            shell=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_environment_ready() -> bool:
    if not os.path.exists(PIXI_LOCK):
        return False
    return os.path.isdir(ENV_MARKER)


def ensure_cuda_torch() -> bool:
    logger.info("Verifying CUDA support in PyTorch environment...")
    check_cmd = [
        "pixi", "run", "python", "-c",
        "import torch; print(torch.cuda.is_available())"
    ]
    try:
        res = subprocess.run(check_cmd, cwd=PROJECT_DIR, capture_output=True, text=True, shell=True)
        cuda_ok = res.stdout.strip() == "True"
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        logger.info("PyTorch GPU support not active. Installing CUDA-enabled PyTorch stack (matching cu126)...")
        install_cmd = [
            "pixi", "run", "pip", "install",
            "torch==2.10.0", "torchaudio==2.11.0",
            "--index-url", "https://download.pytorch.org/whl/cu126",
            "--force-reinstall", "--no-cache-dir"
        ]
        result = subprocess.run(install_cmd, cwd=PROJECT_DIR, shell=True)
        return result.returncode == 0
    else:
        logger.info("CUDA support is already active in PyTorch.")
        return True


def install_dependencies(device: str = "cuda"):
    if not os.path.exists(PIXI_FILE):
        logger.error("pyproject.toml not found at %s", PIXI_FILE)
        sys.exit(1)

    logger.info("Installing pixi dependencies...")
    result = subprocess.run(
        ["pixi", "install", "--frozen"],
        cwd=PROJECT_DIR,
        shell=True,
    )
    success = False
    if result.returncode == 0:
        logger.info("Dependencies installed successfully.")
        success = True
    else:
        logger.info("Frozen install failed, trying update...")
        result = subprocess.run(
            ["pixi", "install"],
            cwd=PROJECT_DIR,
            shell=True,
        )
        if result.returncode == 0:
            logger.info("Dependencies installed successfully.")
            success = True

    if success:
        if device == "cuda":
            return ensure_cuda_torch()
        return True

    logger.error("Failed to install pixi dependencies.")
    return False


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, device: str = "cuda"):
    os.environ["MAGPIE_DEVICE"] = device

    uvicorn_args = [
        "pixi", "run",
        "python", "-m", "uvicorn",
        "main:app",
        "--host", host,
        "--port", str(port),
        "--log-level", "info",
    ]

    logger.info("Starting Magpie TTS API server on %s:%d (device=%s)...", host, port, device)
    logger.info("Voice catalog: %d voices available", len(_get_voice_count()))

    subprocess.run(uvicorn_args, cwd=PROJECT_DIR, shell=True)


def _get_voice_count() -> list:
    try:
        from main import MAGPIE_VOICES
        return MAGPIE_VOICES
    except ImportError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Magpie TTS FastAPI Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for inference (default: cuda)")
    parser.add_argument("--install-only", action="store_true", help="Only install dependencies, don't start server")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency check/install")
    args = parser.parse_args()

    if not is_pixi_installed():
        logger.error("pixi is not installed. Please install pixi first: https://pixi.sh")
        sys.exit(1)

    if args.install_only:
        success = install_dependencies(device=args.device)
        sys.exit(0 if success else 1)

    if not args.skip_install:
        if not is_environment_ready():
            logger.info("Environment not ready. Installing dependencies...")
            if not install_dependencies(device=args.device):
                sys.exit(1)
        else:
            logger.info("Environment is ready.")
            if args.device == "cuda":
                ensure_cuda_torch()

    run_server(host=args.host, port=args.port, device=args.device)


if __name__ == "__main__":
    main()
