import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run


class LauncherTests(unittest.TestCase):
    def test_defaults_bind_to_loopback(self):
        args = run.parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.device, "cuda")

    def test_backend_alias_selects_cpu(self):
        self.assertEqual(run.parse_args(["--backend", "cpu"]).device, "cpu")

    def test_explicit_pixi_path_is_resolved(self):
        with TemporaryDirectory() as directory:
            executable = Path(directory) / ("pixi.exe" if os.name == "nt" else "pixi")
            executable.touch()
            self.assertEqual(run.resolve_pixi(executable), executable.resolve())

    def test_cuda_falls_back_to_cpu_without_nvidia(self):
        args = run.parse_args([])
        with patch("run.shutil.which", return_value=None), patch(
            "run.torch_runtime_ready", return_value=True
        ):
            self.assertEqual(run.resolve_device(args), "cpu")


if __name__ == "__main__":
    unittest.main()
