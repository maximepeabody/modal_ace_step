import io
from pathlib import Path
from typing import Optional
from typing_extensions import Literal

import modal
import torch
from fastapi import File, Form, UploadFile
from pydantic import BaseModel

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("git+https://github.com/ace-step/ACE-Step.git")
)

app = modal.App("ace-step", image=image)

# Shared volume for model weights
volume = modal.Volume.from_name("ace-step-model-cache", create_if_missing=True)
CACHE_PATH = "/cache"

@app.cls(gpu="A10G", container_idle_timeout=300, volumes={CACHE_PATH: volume}, enable_memory_snapshot=True, max_containers=1)
class Model:
    @modal.enter(snap=True)
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        print(f"Loading model from {CACHE_PATH}/checkpoints/ for snapshotting.")
        # When snap=True, no GPU is available, so the model will load on CPU.
        self.checkpoint_dir = f"{CACHE_PATH}/checkpoints/"
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=self.checkpoint_dir,
            dtype="bfloat16",
            torch_compile=False, # torch.compile doesn't work well with Modal yet
        )
        self.pipeline.device = torch.device("cpu")
        self.pipeline.load_checkpoint(self.checkpoint_dir)
        print(f"Model loaded on CPU")

    @modal.enter(snap=False)
    def to_gpu(self):
        self.pipeline.device = torch.device("cuda")
        self.pipeline.load_checkpoint(self.checkpoint_dir)
        print(f"Model moved to GPU")

    def _get_wav_bytes(self, audio_path):
        with open(audio_path, "rb") as f:
            return f.read()

    @modal.method()
    def text_to_audio(self, prompt: str,
                      duration: int = 10, 
                      lyrics: str = "",
                      guidance_scale: float = 15.0,
                      seed: int = -1):
        """
        Generates audio from a text prompt.
        """
        import uuid
        
        print(f"Generating audio for prompt: {prompt}")

        save_path = f"/tmp/{uuid.uuid4()}.wav"
        
        if lyrics == "":
            lyrics = "[inst]"
        
        pipeline_params = {
            "prompt": prompt,
            "audio_duration": duration + 1,
            "lyrics": lyrics,
            "save_path": save_path,
            "format": "wav",
            "use_erg_lyric": lyrics != "" and lyrics is not None,
            "guidance_scale": guidance_scale,
        }

        if seed != -1:
            pipeline_params["manual_seeds"] = [seed]

        output_paths = self.pipeline(**pipeline_params)

        if output_paths:
            print(f"Saved audio to {output_paths[0]}")
            return self._get_wav_bytes(output_paths[0])
        return None

    @modal.method()
    def audio_to_audio(self, 
                       audio_bytes: bytes, 
                       prompt: str,
                       lyrics: str = "", 
                       ref_audio_strength: float = 0.5, 
                       mode: Literal['edit', 'audio2audio'] = 'audio2audio',
                       original_prompt: Optional[str] = None,
                       edit_n_min: float = 0.2,
                       edit_n_max: float = 0.8,
                       edit_n_avg: int = 3):
        """
        Generates audio from a reference audio and a text prompt.
        """
        import uuid
        import torchaudio
        import io
        import torch

        temp_dir = Path("/tmp/audiogen")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        input_path = temp_dir / f"{uuid.uuid4()}.wav"
        output_path = temp_dir / f"{uuid.uuid4()}.wav"

        # Decode audio from bytes, process, and save as a proper WAV file.
        audio_tensor, sr = torchaudio.load(io.BytesIO(audio_bytes))

        # Convert mono to stereo if needed
        if audio_tensor.shape[0] == 1:
            audio_tensor = torch.cat([audio_tensor, audio_tensor], dim=0)

        # Take first two channels if more than stereo
        audio_tensor = audio_tensor[:2]
        
        # Resample to 48kHz, which is what the model expects.
        target_sr = 48000
        if sr != target_sr:
            audio_tensor = torchaudio.transforms.Resample(sr, target_sr)(audio_tensor)
        
        torchaudio.save(str(input_path), audio_tensor, target_sr)

        # Calculate audio duration to pass to the pipeline
        duration_in_seconds = audio_tensor.shape[1] / target_sr

        if not lyrics or lyrics == "":
            lyrics = "[inst]"
        pipeline_params = {
            "lyrics": lyrics,
            "save_path": str(output_path),
            "format": "wav",
            "use_erg_tag": True,
            "use_erg_lyric": True, # bool(lyrics) and lyrics != "",
            "guidance_scale_text": 5,
            "guidance_scale_lyric": 0 if (lyrics == "" or lyrics is None) else 10,
            "guidance_interval": 0.5,
            "guidance_interval_decay": 0,
        }

        if mode == 'edit':
            # For edit mode, we need a source and target prompt.
            # If no original_prompt is provided, we use the target prompt as a fallback.
            source_prompt = original_prompt if original_prompt is not None else prompt
            pipeline_params.update({
                "prompt": source_prompt,
                "edit_target_prompt": prompt,
                "task": "edit",
                "src_audio_path": str(input_path),
                "edit_target_lyrics": lyrics,
                "edit_n_min": edit_n_min,
                "edit_n_max": edit_n_max,
                "edit_n_avg": edit_n_avg,
            })
        else:  # audio2audio
            print(f"Audio2audio mode")
            pipeline_params.update({
                "prompt": prompt,
                "task": "audio2audio",
                "audio_duration": duration_in_seconds + 1,
                "audio2audio_enable": True,
                "ref_audio_input": str(input_path),
                "ref_audio_strength": ref_audio_strength,
            })

        output_paths = self.pipeline(**pipeline_params)
        
        if output_paths:
            return self._get_wav_bytes(output_paths[0])
        return None


class TextToAudioRequest(BaseModel):
    prompt: str
    duration: int = 10
    lyrics: str = ""
    guidance_scale: float = 15.0
    seed: int = -1


@app.function()
@modal.web_endpoint(method="POST")
def text_to_audio_endpoint(req: TextToAudioRequest):
    """
    Web endpoint for text-to-audio generation.
    request body: {"prompt": "...", "duration": 10, "lyrics": "..."}
    """
    from fastapi.responses import Response

    model = Model()
    audio_bytes = model.text_to_audio.remote(
        req.prompt, req.duration, req.lyrics, req.guidance_scale, req.seed
    )
    
    if audio_bytes:
        return Response(content=audio_bytes, media_type="audio/wav")
    
    return Response(content="Failed to generate audio", status_code=500)

@app.function()
@modal.web_endpoint(method="POST")
async def audio_to_audio_endpoint(
    prompt: str = Form(),
    audio: UploadFile = File(),
    lyrics: Optional[str] = Form(""),
    ref_audio_strength: float = Form(0.5),
    mode: Literal['edit', 'audio2audio'] = Form('audio2audio'),
    original_prompt: Optional[str] = Form(None),
    edit_n_min: float = Form(0.0),
    edit_n_max: float = Form(1.0),
    edit_n_avg: int = Form(1)
):
    """
    Web endpoint for audio-to-audio generation.
    Request should be a POST with multipart/form-data.
    It should contain an 'audio' file and a 'prompt' field.
    An optional 'lyrics' field can also be included.
    """
    from fastapi.responses import Response

    model = Model()
    audio_bytes = await audio.read()
    output_audio_bytes = model.audio_to_audio.remote(
        audio_bytes,
        prompt,
        lyrics or "", 
        ref_audio_strength, 
        mode,
        original_prompt,
        edit_n_min, 
        edit_n_max,
        edit_n_avg
    )

    if output_audio_bytes:
        return Response(content=output_audio_bytes, media_type="audio/wav")

    return Response(content="Failed to generate audio", status_code=500)

@app.local_entrypoint()
def main(
    prompt: str = "smooth relaxing jazz",
    duration: int = 30,
):
    print(f"Generating audio for prompt: {prompt}")
    model = Model()
    audio_bytes = model.text_to_audio.remote(prompt, duration)
    
    if audio_bytes:
        # create output directory if it doesn't exist
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{prompt}.wav"
        print(f"Saving to {output_path}")
        output_path.write_bytes(audio_bytes) 

    else:
        print("Failed to generate audio")
