import os
import io
import sys
import threading
import logging
import re
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("magpie-api")

HAS_NEMO = False
try:
    from nemo.collections.tts.models import MagpieTTSModel
    import torch
    import soundfile as sf
    import numpy as np
    HAS_NEMO = True
except ImportError as e:
    logger.warning("NeMo or PyTorch not fully installed yet: %s. Run run.bat first.", e)

app = FastAPI(
    title="Magpie TTS API Wrapper",
    description="OpenAI-compatible TTS API wrapper for NVIDIA Magpie TTS Multilingual (via NeMo Framework)",
)

LOCALE_MAP = {
    "EN-US": "en",
    "ES-US": "es",
    "FR-FR": "fr",
    "DE-DE": "de",
    "VI-VN": "vi",
    "IT-IT": "it",
    "ZH-CN": "zh",
    "HI-IN": "hi",
    "JA-JP": "ja",
}

LOCALE_TO_LANGUAGE_CODE = {v: k for k, v in LOCALE_MAP.items()}

SPEAKER_NAMES = ["Sofia", "Aria", "Jason", "Leo", "John Van Stan"]

# model speaker_map from HF README: John=0, Sofia=1, Aria=2, Jason=3, Leo=4
SPEAKER_INDEX_MAP = {"John Van Stan": 0, "Sofia": 1, "Aria": 2, "Jason": 3, "Leo": 4}

EMOTIONS = ["Angry", "Calm", "Happy", "Neutral", "Sad", "Fearful"]

# Only EN-US has emotion variants for the 5 open-source speakers.
# ES-US/FR-FR/DE-DE emotions are for NIM-exclusive speakers (Diego, Pascal, etc.)
LOCALES_WITH_EMOTIONS = {"EN-US"}

def _build_voice_catalog() -> list[str]:
    voices = []
    for locale in LOCALE_MAP:
        for speaker in SPEAKER_NAMES:
            voices.append(f"Magpie-Multilingual.{locale}.{speaker}")
            if locale in LOCALES_WITH_EMOTIONS:
                for emotion in EMOTIONS:
                    voices.append(f"Magpie-Multilingual.{locale}.{speaker}.{emotion}")
    return voices

MAGPIE_VOICES = _build_voice_catalog()


def parse_magpie_voice(voice: str) -> tuple[str, str, str | None] | None:
    if not voice:
        return None
    parts = voice.split(".")
    if len(parts) < 3 or parts[0] != "Magpie-Multilingual":
        return None
    locale = parts[1].upper()
    speaker_raw = parts[2]
    emotion = parts[3] if len(parts) > 3 else None

    if locale not in LOCALE_MAP:
        return None

    speaker_key = None
    for name in SPEAKER_NAMES:
        if name.lower() == speaker_raw.lower():
            speaker_key = name
            break

    if speaker_key is None:
        return None

    return (locale, speaker_key, emotion)


def voice_to_language_code(voice: str) -> str:
    parsed = parse_magpie_voice(voice)
    if parsed is None:
        return "en"
    locale, _, _ = parsed
    return LOCALE_MAP.get(locale, "en")


def voice_to_speaker_index(voice: str) -> int:
    parsed = parse_magpie_voice(voice)
    if parsed is None:
        return 0
    _, speaker, _ = parsed
    return SPEAKER_INDEX_MAP.get(speaker, 1)


class ModelLoader:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def get_model(self, device: str = "cuda"):
        if not HAS_NEMO:
            raise HTTPException(status_code=500, detail="NeMo/PyTorch libraries are not installed. Run run.bat first.")

        with self._lock:
            if self._model is not None:
                return self._model

            target_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
            logger.info("Loading Magpie TTS model on device '%s'...", target_device)

            try:
                model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
                model.eval()
                if target_device == "cuda":
                    model.cuda()
                else:
                    model.cpu()
                self._model = model
                logger.info("Successfully loaded Magpie TTS model")
                return model
            except Exception as e:
                logger.error("Failed to load Magpie TTS model: %s", e, exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    def unload(self):
        with self._lock:
            if self._model is not None:
                self._model = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Unloaded Magpie TTS model")


model_loader = ModelLoader()


class SpeechRequest(BaseModel):
    model: str = "magpie-tts-multilingual"
    input: str
    voice: Optional[str] = "Magpie-Multilingual.EN-US.Aria"
    language: Optional[str] = None
    speed: Optional[float] = 1.0
    use_cfg: Optional[bool] = True
    apply_text_normalization: Optional[bool] = True
    response_format: Optional[str] = "wav"


@app.get("/health")
@app.get("/")
async def health_check():
    cuda_available = False
    cuda_device_name = ""
    if HAS_NEMO:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
    return {
        "status": "ok",
        "model": "nvidia/magpie_tts_multilingual_357m",
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
        "model_loaded": model_loader._model is not None,
        "voices_count": len(MAGPIE_VOICES),
    }


@app.get("/v1/models")
@app.get("/v1/audio/models")
async def list_models():
    return {
        "data": [
            {"id": "magpie-tts-multilingual", "object": "model", "owned_by": "nvidia"},
        ]
    }


@app.get("/v1/audio/voices")
@app.get("/v1/voices")
async def list_voices():
    voices_data = []
    for voice_id in MAGPIE_VOICES:
        parsed = parse_magpie_voice(voice_id)
        if parsed:
            locale, speaker, emotion = parsed
            entry = {"id": voice_id, "voice_id": voice_id, "name": voice_id, "locale": locale, "speaker": speaker}
            if emotion:
                entry["emotion"] = emotion
            voices_data.append(entry)
    return {"data": voices_data, "voices": MAGPIE_VOICES}


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def generate_speech(request: SpeechRequest):
    if not HAS_NEMO:
        raise HTTPException(status_code=500, detail="NeMo libraries not installed.")

    logger.info(
        "Speech request: model=%s, text_len=%d, voice=%s, lang=%s",
        request.model,
        len(request.input),
        request.voice,
        request.language,
    )

    voice_id = request.voice or "Magpie-Multilingual.EN-US.Aria"

    parsed = parse_magpie_voice(voice_id)
    if parsed is None:
        available = LOCALE_MAP.keys()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice format: '{voice_id}'. Expected format: Magpie-Multilingual.{{LOCALE}}.{{Speaker}} "
                   f"where LOCALE is one of {', '.join(sorted(available))} and Speaker is one of {', '.join(SPEAKER_NAMES)}",
        )

    locale, speaker, emotion = parsed
    language_code = LOCALE_MAP.get(locale, "en")
    speaker_idx = SPEAKER_INDEX_MAP.get(speaker, 0)

    if emotion:
        logger.info("Emotion '%s' specified for voice '%s' - model will use neutral style (emotion not controllable via open-source checkpoint)", emotion, voice_id)

    lang_param = (request.language or language_code).strip().lower()
    if lang_param not in LOCALE_MAP.values():
        logger.warning("Unsupported language '%s', falling back to '%s'", lang_param, language_code)
        lang_param = language_code

    backend = os.environ.get("MAGPIE_DEVICE", "cuda").lower()
    device = "cpu" if backend == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_loader.get_model(device=device)

    try:
        import inspect
        sig = inspect.signature(model.do_tts)
        params = sig.parameters

        kwargs = {
            "transcript": request.input,
            "language": lang_param,
            "apply_TN": request.apply_text_normalization,
            "speaker_index": speaker_idx,
        }

        if "use_cfg" in params and request.use_cfg is not None:
            kwargs["use_cfg"] = request.use_cfg

        logger.info("Calling do_tts with language=%s, speaker=%s (idx=%d), apply_TN=%s", lang_param, speaker, speaker_idx, request.apply_text_normalization)

        audio, audio_len = model.do_tts(**kwargs)

        sr = getattr(model, "sr", 22050)

        audio_numpy = audio[0].cpu().numpy()

        if request.speed and request.speed != 1.0:
            try:
                import tempfile
                import subprocess

                speed_val = float(request.speed)
                if speed_val < 0.01:
                    speed_val = 0.01

                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = os.path.join(tmpdir, "input.wav")
                    output_path = os.path.join(tmpdir, "output.wav")
                    sf.write(input_path, audio_numpy, sr, format="wav")

                    # atempo filter supports 0.5 to 2.0. Chain if we are outside this range.
                    filter_parts = []
                    temp_speed = speed_val
                    while temp_speed < 0.5:
                        filter_parts.append("atempo=0.5")
                        temp_speed /= 0.5
                    while temp_speed > 2.0:
                        filter_parts.append("atempo=2.0")
                        temp_speed /= 2.0
                    filter_parts.append(f"atempo={temp_speed}")
                    filter_str = ",".join(filter_parts)

                    cmd = [
                        "ffmpeg", "-y", "-i", input_path,
                        "-filter:a", filter_str,
                        output_path
                    ]
                    
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                    if result.returncode == 0:
                        out_data, out_sr = sf.read(output_path)
                        output_buf = io.BytesIO()
                        sf.write(output_buf, out_data, out_sr, format="wav")
                        output_buf.seek(0)
                        logger.info("Applied robust FFmpeg speed adjustment: %fx", speed_val)
                        return StreamingResponse(output_buf, media_type="audio/wav")
                    else:
                        raise RuntimeError(f"FFmpeg failed: {result.stderr.decode('utf-8', errors='ignore')}")
            except Exception as speed_err:
                logger.warning("Speed adjustment failed, returning original: %s", speed_err)

        output_buf = io.BytesIO()
        sf.write(output_buf, audio_numpy, sr, format="wav")
        output_buf.seek(0)

        return StreamingResponse(output_buf, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("TTS generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.on_event("shutdown")
async def shutdown():
    model_loader.unload()
