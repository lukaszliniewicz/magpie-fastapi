import torch
print(f"torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

import soundfile
print(f"soundfile {soundfile.__version__}")

import fastapi
print(f"fastapi {fastapi.__version__}")

from nemo.collections.tts.models import MagpieTTSModel
print("MagpieTTSModel: imported successfully")
print("All imports OK!")
