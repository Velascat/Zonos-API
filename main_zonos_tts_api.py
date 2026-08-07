# //////////////////////////////////////////////////////////////////////////////////////
# ///// Setup Logger ///////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

from pathlib import Path

from logger import set_logger_name

set_logger_name(logger_name=Path(__file__).stem)

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Imports ////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

import os
import sys
import json
import uuid
import time
import torch
import traceback
import torchaudio
import multiprocessing
import base64
import queue
import threading
from io import BytesIO
from typing import Dict, Optional, List, Union

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, conlist, confloat

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict, supported_language_codes
from logger import logger

# //////////////////////////////////////////////////////////////////////////////////////
# ///// Initialization /////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

multiprocessing.set_start_method("spawn", force=True)

app = FastAPI(title="Zonos API", description="OpenAI-compatible TTS API for Zonos")

TRANSFORMER_REPO_ID = "Zyphra/Zonos-v0.1-transformer"
HYBRID_REPO_ID = "Zyphra/Zonos-v0.1-hybrid"
MODEL_SLOT_BY_ID = {
    TRANSFORMER_REPO_ID: "transformer",
    HYBRID_REPO_ID: "hybrid",
}
DEFAULT_MODEL_ID = TRANSFORMER_REPO_ID

MODELS = {"transformer": None, "hybrid": None}
HYBRID_SKIP_REASON: Optional[str] = None


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


HYBRID_REQUIRED = _bool_env("ZONOS_REQUIRE_HYBRID", False)
ZONOS_SKIP_MAMBA = _bool_env("ZONOS_SKIP_MAMBA", False)
VOICE_STORAGE_DIR = os.environ.get("VOICE_STORAGE_DIR", "data/voice_storage")
VOICE_METADATA_FILE = os.path.join(VOICE_STORAGE_DIR, "voice_metadata.json")
VOICE_CACHE: Dict[str, torch.Tensor] = {}


def _cuda_is_available() -> bool:
    cuda_attr = getattr(torch, "cuda", None)
    is_available = getattr(cuda_attr, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


DEVICE = "cuda" if _cuda_is_available() else "cpu"

os.makedirs(VOICE_STORAGE_DIR, exist_ok=True)


def log_backend_versions() -> bool:
    """Log torch/mamba versions and report whether the hybrid stack is usable."""
    global HYBRID_SKIP_REASON

    HYBRID_SKIP_REASON = None
    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or "cpu-only"
    cuda_available = _cuda_is_available()
    logger.info(
        "Torch %s (CUDA %s, available=%s)",
        torch_version,
        cuda_version,
        cuda_available,
    )

    try:
        torch_path = getattr(torch, "__file__", "<unknown>")
        logger.info("Python prefix=%s", sys.prefix)
        logger.info("Torch path=%s", torch_path)
    except Exception:
        logger.warning("Could not log env/torch path", exc_info=True)

    if ZONOS_SKIP_MAMBA or not cuda_available:
        if ZONOS_SKIP_MAMBA:
            HYBRID_SKIP_REASON = "mamba-ssm import skipped by configuration"
        else:
            HYBRID_SKIP_REASON = "cuda unavailable"
        logger.warning(
            "Skipping mamba-ssm import (%s).",
            HYBRID_SKIP_REASON,
        )
        return False

    try:
        import mamba_ssm  # type: ignore

        logger.info(
            "mamba-ssm %s import OK", getattr(mamba_ssm, "__version__", "unknown")
        )
        return True
    except Exception as exc:
        HYBRID_SKIP_REASON = "mamba-ssm import failed"
        logger.warning(
            "mamba-ssm import failed; the hybrid checkpoint will be skipped (%s).",
            exc,
        )
        return False


# //////////////////////////////////////////////////////////////////////////////////////
# ///// Helpers ////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////


def load_models(skip_hybrid: bool = False):
    global HYBRID_SKIP_REASON
    logger.info("Loading Zonos models into VRAM...")
    device = DEVICE

    try:
        MODELS["transformer"] = (
            Zonos.from_pretrained(TRANSFORMER_REPO_ID, device=device, backbone="torch")
            .eval()
            .requires_grad_(False)
        )
        logger.info("Transformer checkpoint loaded on %s.", device)
    except Exception:
        logger.error(
            "Failed to load transformer checkpoint:\n%s", traceback.format_exc()
        )
        return

    if skip_hybrid:
        if HYBRID_SKIP_REASON is None:
            if HYBRID_REQUIRED:
                HYBRID_SKIP_REASON = "hybrid required but unavailable"
            else:
                HYBRID_SKIP_REASON = "skipped by configuration"
        logger.warning(
            "Skipping hybrid checkpoint load because mamba-ssm is unavailable (%s).",
            HYBRID_SKIP_REASON,
        )
        MODELS["hybrid"] = None
        return

    try:
        MODELS["hybrid"] = (
            Zonos.from_pretrained(HYBRID_REPO_ID, device=device, backbone="mamba_ssm")
            .eval()
            .requires_grad_(False)
        )
        logger.info("Hybrid checkpoint loaded on %s.", device)
    except Exception:
        MODELS["hybrid"] = None
        HYBRID_SKIP_REASON = "hybrid checkpoint load failure"
        logger.warning(
            "Hybrid checkpoint unavailable; requests will fall back to the transformer model.\n%s",
            traceback.format_exc(),
        )


def load_voice_metadata():
    try:
        if os.path.exists(VOICE_METADATA_FILE):
            with open(VOICE_METADATA_FILE, "r") as f:
                return json.load(f)
    except Exception:
        logger.error("Error loading voice metadata:\n%s", traceback.format_exc())
    return {}


def save_voice_metadata(metadata):
    try:
        with open(VOICE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        logger.error("Error saving voice metadata:\n%s", traceback.format_exc())


def load_voice_embeddings():
    metadata = load_voice_metadata()
    for voice_id, info in metadata.items():
        tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
        try:
            if os.path.exists(tensor_path):
                VOICE_CACHE[voice_id] = torch.load(tensor_path, map_location=DEVICE)
                logger.info(f"Loaded voice: {info.get('name', voice_id)}")
        except Exception:
            logger.error(
                "Error loading voice embedding %s:\n%s", voice_id, traceback.format_exc()
            )


def save_voice_embedding(voice_id, embedding):
    try:
        torch.save(embedding, os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt"))
        return True
    except Exception:
        logger.error(
            "Error saving voice embedding %s:\n%s", voice_id, traceback.format_exc()
        )
        return False


def get_voice_embedding(voice_identifier):
    metadata = load_voice_metadata()
    if voice_identifier in VOICE_CACHE:
        return VOICE_CACHE[voice_identifier]

    for voice_id, info in metadata.items():
        if info.get("name") == voice_identifier:
            tensor_path = os.path.join(VOICE_STORAGE_DIR, f"{voice_id}.pt")
            try:
                if os.path.exists(tensor_path):
                    embedding = torch.load(tensor_path, map_location=DEVICE)
                    VOICE_CACHE[voice_id] = embedding
                    return embedding
            except Exception:
                logger.error(
                    "Error loading voice %s:\n%s", voice_id, traceback.format_exc()
                )
    return None


def _required_model_slots() -> list[str]:
    slots = list(MODELS.keys())
    if not slots:
        return []
    if not HYBRID_REQUIRED and "hybrid" in slots:
        slots = [slot for slot in slots if slot != "hybrid"]
    return slots


def _readiness_snapshot() -> Dict[str, object]:
    required_slots = _required_model_slots()
    total = len(required_slots)
    ready = sum(1 for slot in required_slots if MODELS.get(slot) is not None)
    detail = None

    if total < 1:
        detail = "no models configured"
    elif ready < total:
        if "transformer" in required_slots and MODELS.get("transformer") is None:
            detail = "transformer model not loaded"
        elif (
            HYBRID_REQUIRED
            and "hybrid" in required_slots
            and MODELS.get("hybrid") is None
        ):
            detail = HYBRID_SKIP_REASON or "hybrid model not loaded"
        else:
            detail = "models still loading"

    status = "ready" if total > 0 and ready == total else "not_ready"
    return {
        "status": status,
        "models": {"total": total, "ready": ready},
        "required": required_slots,
        "detail": detail,
    }


@app.get("/healthz")
async def healthz():
    snapshot = _readiness_snapshot()
    status_code = 200 if snapshot["status"] == "ready" else 503
    return JSONResponse(snapshot, status_code=status_code)


@app.get("/livez")
async def livez():
    return JSONResponse({"status": "alive"}, status_code=200)


# //////////////////////////////////////////////////////////////////////////////////////
# ///// Request Models /////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////


class SpeechRequest(BaseModel):
    model: str = Field(DEFAULT_MODEL_ID)
    input: str = Field(..., max_length=500)
    voice: Optional[str] = None
    speed: float = Field(1.0, ge=0.5, le=2.0)
    language: str = Field("en-us")
    emotion: Optional[Dict[str, float]] = None
    response_format: str = Field("mp3")
    prefix_audio: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(0.15, ge=0.0, le=1.0)
    pitch_std: Optional[float] = Field(None, ge=0.0, le=400.0)
    speaking_rate: Optional[float] = Field(None, ge=0.0, le=40.0)
    fmax: Optional[float] = Field(None, ge=0.0, le=24000.0)
    vqscore_8: Optional[
        conlist(confloat(ge=0.5, le=0.8), min_length=8, max_length=8)
    ] = None
    dnsmos_ovrl: Optional[float] = Field(None, ge=1.0, le=5.0)
    speaker_noised: Optional[bool] = None


class VoiceResponse(BaseModel):
    voice_id: str
    name: Optional[str]
    created: int


class VoiceListResponse(BaseModel):
    voices: List[Dict[str, Union[str, int, None]]]


# //////////////////////////////////////////////////////////////////////////////////////
# ///// Endpoints //////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    try:
        requested_key = MODEL_SLOT_BY_ID.get(request.model)
        if requested_key is None:
            requested_key = "transformer" if "transformer" in request.model else "hybrid"
            if requested_key == "hybrid":
                logger.warning(
                    "Unknown model id '%s'; defaulting to hybrid slot", request.model
                )
            else:
                logger.warning(
                    "Unknown model id '%s'; defaulting to transformer weights",
                    request.model,
                )
        model = MODELS.get(requested_key)

        if model is None and requested_key == "hybrid":
            reason = HYBRID_SKIP_REASON or "hybrid weights not loaded"
            logger.warning(
                "Hybrid model requested but not available (%s); falling back to transformer weights.",
                reason,
            )
            model = MODELS.get("transformer")

        if model is None:
            raise HTTPException(
                status_code=503, detail="Model weights are not loaded yet."
            )
        speaking_rate = 15.0 * request.speed

        emotion_tensor = None
        if request.emotion:
            emotion_tensor = torch.tensor(
                [
                    request.emotion.get("happiness", 1.0),
                    request.emotion.get("sadness", 0.05),
                    request.emotion.get("disgust", 0.05),
                    request.emotion.get("fear", 0.05),
                    request.emotion.get("surprise", 0.05),
                    request.emotion.get("anger", 0.05),
                    request.emotion.get("other", 0.1),
                    request.emotion.get("neutral", 0.2),
                ],
                device=DEVICE,
            ).unsqueeze(0)

        speaker_embedding = get_voice_embedding(request.voice) if request.voice else None
        if request.voice and speaker_embedding is None:
            raise HTTPException(
                status_code=404, detail=f"Voice '{request.voice}' not found."
            )

        cond_kwargs = {
            "text": request.input,
            "language": request.language,
            "speaker": speaker_embedding,
            "emotion": emotion_tensor,
            "speaking_rate": request.speaking_rate
            if request.speaking_rate is not None
            else speaking_rate,
            "device": DEVICE,
            "unconditional_keys": [] if request.emotion else ["emotion"],
        }
        if request.pitch_std is not None:
            cond_kwargs["pitch_std"] = request.pitch_std
        if request.fmax is not None:
            cond_kwargs["fmax"] = request.fmax
        if request.vqscore_8 is not None:
            cond_kwargs["vqscore_8"] = request.vqscore_8
        if request.dnsmos_ovrl is not None:
            cond_kwargs["dnsmos_ovrl"] = request.dnsmos_ovrl
        if request.speaker_noised is not None:
            cond_kwargs["speaker_noised"] = request.speaker_noised

        cond_dict = make_cond_dict(**cond_kwargs)

        conditioning = model.prepare_conditioning(cond_dict)

        sampling_params = {
            k: v
            for k, v in {
                "top_k": request.top_k,
                "top_p": request.top_p,
                "min_p": request.min_p,
            }.items()
            if v is not None
        } or {"min_p": 0.15}

        start_time = time.time()
        heartbeat_interval = float(os.environ.get("ZONOS_HEARTBEAT_INTERVAL", "30"))
        last_heartbeat = start_time

        def _heartbeat(_frame: torch.Tensor, step: int, total_steps: int) -> bool:
            nonlocal last_heartbeat
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                logger.info(
                    "zonos_synth_heartbeat",
                    extra={
                        "voice": request.voice or "default",
                        "elapsed_s": round(now - start_time, 2),
                        "step": step,
                        "total_steps": total_steps,
                        "text_len": len(request.input or ""),
                    },
                )
            return True

        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 30,
            cfg_scale=2.0,
            batch_size=1,
            sampling_params=sampling_params,
            callback=_heartbeat,
        )

        wav_out = model.autoencoder.decode(codes).cpu().detach()
        sr_out = model.autoencoder.sampling_rate

        if wav_out.dim() > 2:
            wav_out = wav_out.squeeze()
        if wav_out.dim() == 1:
            wav_out = wav_out.unsqueeze(0)

        buffer = BytesIO()
        torchaudio.save(buffer, wav_out, sr_out, format=request.response_format)
        buffer.seek(0)

        logger.info(
            "zonos_synth_complete",
            extra={
                "voice": request.voice or "default",
                "elapsed_s": round(time.time() - start_time, 2),
                "text_len": len(request.input or ""),
            },
        )

        return StreamingResponse(buffer, media_type=f"audio/{request.response_format}")

    except Exception:
        logger.exception("Error during speech synthesis")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate speech",
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/v1/audio/speech_stream")
async def create_speech_stream(request: SpeechRequest):
    """
    Stream heartbeats during synthesis and return base64 audio when complete.
    """
    try:
        requested_key = MODEL_SLOT_BY_ID.get(request.model)
        if requested_key is None:
            requested_key = "transformer" if "transformer" in request.model else "hybrid"
        model = MODELS.get(requested_key)

        if model is None and requested_key == "hybrid":
            reason = HYBRID_SKIP_REASON or "hybrid weights not loaded"
            logger.warning(
                "Hybrid model requested but not available (%s); falling back to transformer weights.",
                reason,
            )
            model = MODELS.get("transformer")

        if model is None:
            raise HTTPException(
                status_code=503, detail="Model weights are not loaded yet."
            )

        if model.autoencoder is None:
            raise HTTPException(
                status_code=503,
                detail="Autoencoder not initialized on the server.",
            )

        speaking_rate = 15.0 * request.speed

        emotion_tensor = None
        if request.emotion:
            emotion_tensor = torch.tensor(
                [
                    request.emotion.get("happiness", 1.0),
                    request.emotion.get("sadness", 0.05),
                    request.emotion.get("disgust", 0.05),
                    request.emotion.get("fear", 0.05),
                    request.emotion.get("surprise", 0.05),
                    request.emotion.get("anger", 0.05),
                    request.emotion.get("other", 0.1),
                    request.emotion.get("neutral", 0.2),
                ],
                device=DEVICE,
            ).unsqueeze(0)

        speaker_embedding = get_voice_embedding(request.voice) if request.voice else None
        if request.voice and speaker_embedding is None:
            raise HTTPException(
                status_code=404, detail=f"Voice '{request.voice}' not found."
            )

        cond_kwargs = {
            "text": request.input,
            "language": request.language,
            "speaker": speaker_embedding,
            "emotion": emotion_tensor,
            "speaking_rate": (
                request.speaking_rate
                if request.speaking_rate is not None
                else speaking_rate
            ),
            "device": DEVICE,
            "unconditional_keys": [] if request.emotion else ["emotion"],
        }
        if request.pitch_std is not None:
            cond_kwargs["pitch_std"] = request.pitch_std
        if request.fmax is not None:
            cond_kwargs["fmax"] = request.fmax
        if request.vqscore_8 is not None:
            cond_kwargs["vqscore_8"] = request.vqscore_8
        if request.dnsmos_ovrl is not None:
            cond_kwargs["dnsmos_ovrl"] = request.dnsmos_ovrl
        if request.speaker_noised is not None:
            cond_kwargs["speaker_noised"] = request.speaker_noised

        cond_dict = make_cond_dict(**cond_kwargs)
        conditioning = model.prepare_conditioning(cond_dict)

        sampling_params = {
            k: v
            for k, v in {
                "top_k": request.top_k,
                "top_p": request.top_p,
                "min_p": request.min_p,
            }.items()
            if v is not None
        } or {"min_p": 0.15}

        heartbeat_interval = float(os.environ.get("ZONOS_HEARTBEAT_INTERVAL", "30"))
        progress_queue: queue.Queue[dict] = queue.Queue()

        def _heartbeat(_frame: torch.Tensor, step: int, total_steps: int) -> bool:
            progress_queue.put(
                {
                    "event": "progress",
                    "step": step,
                    "total_steps": total_steps,
                    "text_len": len(request.input or ""),
                    "voice": request.voice or "default",
                    "timestamp": time.time(),
                }
            )
            return True

        def _worker():
            start_time = time.time()
            heartbeat_state = {"last": start_time}
            try:
                # Kick off first heartbeat immediately.
                progress_queue.put(
                    {
                        "event": "start",
                        "voice": request.voice or "default",
                        "text_len": len(request.input or ""),
                        "timestamp": start_time,
                    }
                )

                codes = model.generate(
                    prefix_conditioning=conditioning,
                    max_new_tokens=86 * 30,
                    cfg_scale=2.0,
                    batch_size=1,
                    sampling_params=sampling_params,
                    callback=lambda frame, step, total: (
                        lambda now: (
                            (
                                heartbeat_state.__setitem__("last", now),
                                progress_queue.put(
                                    {
                                        "event": "progress",
                                        "step": step,
                                        "total_steps": total,
                                        "text_len": len(request.input or ""),
                                        "voice": request.voice or "default",
                                        "timestamp": now,
                                    }
                                ),
                            )
                        )
                        if (now - heartbeat_state["last"]) >= heartbeat_interval
                        else None
                    )((time.time()))
                    or True,
                )

                wav_out = model.autoencoder.decode(codes).cpu().detach()
                sr_out = model.autoencoder.sampling_rate

                if wav_out.dim() > 2:
                    wav_out = wav_out.squeeze()
                if wav_out.dim() == 1:
                    wav_out = wav_out.unsqueeze(0)

                buffer = BytesIO()
                torchaudio.save(buffer, wav_out, sr_out, format=request.response_format)
                buffer.seek(0)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

                progress_queue.put(
                    {
                        "event": "done",
                        "voice": request.voice or "default",
                        "elapsed_s": round(time.time() - start_time, 2),
                        "text_len": len(request.input or ""),
                        "audio_b64": encoded,
                        "response_format": request.response_format,
                    }
                )
            except Exception:
                progress_queue.put(
                    {
                        "event": "error",
                        "detail": "Failed to generate speech",
                        "traceback": traceback.format_exc(),
                    }
                )

        worker = threading.Thread(
            target=_worker, daemon=True, name="zonos-speech-stream"
        )
        worker.start()

        def _iter():
            while True:
                item = progress_queue.get()
                yield (json.dumps(item) + "\n").encode("utf-8")
                if item.get("event") in {"done", "error"}:
                    break

        return StreamingResponse(_iter(), media_type="application/json")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error during speech synthesis (stream)")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate speech",
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/v1/audio/voice")
async def create_voice(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    try:
        wav, sr = torchaudio.load(BytesIO(await file.read()))
        speaker_embedding = MODELS["transformer"].make_speaker_embedding(wav, sr)

        timestamp = int(time.time())
        voice_id = f"voice_{timestamp}_{uuid.uuid4().hex[:8]}"
        VOICE_CACHE[voice_id] = speaker_embedding.to(DEVICE)

        if not save_voice_embedding(voice_id, speaker_embedding.to(DEVICE)):
            raise HTTPException(status_code=500, detail="Failed to save voice embedding")

        metadata = load_voice_metadata()
        metadata[voice_id] = {"created": timestamp, "name": name}
        save_voice_metadata(metadata)

        return VoiceResponse(voice_id=voice_id, name=name, created=timestamp)

    except Exception:
        logger.exception("Error during voice creation")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to process uploaded voice",
                "traceback": traceback.format_exc(),
            },
        )


@app.get("/v1/audio/voices")
async def list_voices():
    metadata = load_voice_metadata()
    return VoiceListResponse(
        voices=[
            {"voice_id": vid, "name": info.get("name"), "created": info.get("created")}
            for vid, info in metadata.items()
        ]
    )


@app.get("/v1/audio/models")
async def list_models():
    models = []
    for repo_id, slot in MODEL_SLOT_BY_ID.items():
        ready = MODELS.get(slot) is not None
        entry = {
            "id": repo_id,
            "created": 1234567890,
            "object": "model",
            "owned_by": "zyphra",
            "ready": ready,
        }
        if slot == "hybrid" and not ready and HYBRID_SKIP_REASON:
            entry["note"] = HYBRID_SKIP_REASON
        models.append(entry)

    return {"models": models}


@app.on_event("startup")
async def startup_event():
    # Clear download wreckage BEFORE any load can resume it: a corrupt
    # .incomplete partial wedges hf_hub_download's retries indefinitely
    # (the 2026-08-03/04 outage needed a human to delete it). Age-guarded,
    # never raises — see zonos.integrity.sweep_stale_partials.
    from zonos.integrity import sweep_stale_partials

    swept = sweep_stale_partials()
    if swept:
        logger.warning(
            "Removed %d stale download partial(s) at startup: %s",
            len(swept),
            ", ".join(swept),
        )
    hybrid_ready = log_backend_versions()
    load_models(skip_hybrid=not hybrid_ready)
    load_voice_embeddings()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
