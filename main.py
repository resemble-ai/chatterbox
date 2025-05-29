import io
import random
import base64 # Added
import re # Added
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
# import torchaudio # No longer explicitly used in predict
import uvicorn
from fastapi import FastAPI, HTTPException # HTTPException Added
from fastapi.responses import JSONResponse # Added
from httpx import AsyncClient, RequestError, HTTPStatusError # RequestError, HTTPStatusError Added
from litserve import LitAPI
from pydantic import BaseModel, Field

from chatterbox.tts import ChatterboxTTS


class TTSRequest(BaseModel):
    text: str = Field(..., example="Text for speech translation, 300-500 words approx")
    ref_file_url: str = Field(..., example="URL to the voice file to be cloned")
    cfg: float = Field(default=0.5, example=0.5) # Used as cfg_weight
    exaggeration: float = Field(default=0.5, example=0.5)
    random_seed: int = Field(default=0)
    temperature: float = Field(default=0.8, example=0.8)


class ChatterboxLitAPI(LitAPI):
    def setup(self): # Removed device argument
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        # Ensure model's sample rate is accessible, e.g., self.model.sr
        # If ChatterboxTTS doesn't expose it directly, we might need to assume or retrieve it.
        # For now, assuming self.model.sr exists and is 24000 as per Chatterbox's S3GEN_SR.
        print(f"ChatterboxTTS model loaded on {self.device}. Sample rate: {self.model.sr}")

    async def decode_request(self, request: TTSRequest, client: AsyncClient):
        temp_ref_path = Path("temp_ref.wav")
        try:
            response = await client.get(request.ref_file_url)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            with open(temp_ref_path, "wb") as f:
                f.write(response.content)
        except HTTPStatusError as e:
            # Log the error e.response.status_code, e.request.url
            raise HTTPException(status_code=e.response.status_code, detail=f"Error fetching or accessing reference file URL: {e.request.url}. Server responded with {e.response.status_code}")
        except RequestError as e:
            # Log the error e.request.url
            raise HTTPException(status_code=503, detail=f"Network error while trying to download reference file: {e.request.url}. Please check the URL and network connectivity.")
        except IOError as e:
            # Log the error
            raise HTTPException(status_code=500, detail=f"Failed to write temporary reference audio file: {e}")
        except Exception as e:
            # Catch any other unexpected errors during download/file write
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while preparing reference audio: {str(e)}")


        return {
            "text": request.text,
            "ref_audio_path": str(temp_ref_path),
            "cfg": request.cfg, # This is cfg_weight for model.generate
            "exaggeration": request.exaggeration,
            "random_seed": request.random_seed,
            "temperature": request.temperature,
        }

    async def predict(self, inputs: dict):
        text = inputs["text"]
        ref_audio_path_str = inputs["ref_audio_path"] # Keep as string for Path operations
        ref_audio_path = Path(ref_audio_path_str) # Convert to Path object for unlink
        cfg = inputs["cfg"] # This will be cfg_weight
        exaggeration = inputs["exaggeration"]
        random_seed = inputs["random_seed"]
        temperature = inputs["temperature"]

        try:
            # Set random seed for reproducibility
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

            try:
                # Prepare conditionals from reference audio
                self.model.prepare_conditionals(ref_audio_path_str, exaggeration=exaggeration)
            except FileNotFoundError:
                 raise HTTPException(status_code=400, detail=f"Reference audio file not found at path derived from URL. Path: {ref_audio_path_str}")
            except Exception as e: # Catch errors from librosa or other audio processing issues
                # Log the original exception e
                raise HTTPException(status_code=400, detail=f"Error processing reference audio: {str(e)}")

            # Text Chunking
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            
            if not sentences:
                raise HTTPException(status_code=400, detail="Input text is empty or contains no processable sentences after filtering.")

            list_of_wav_tensors = []
            try:
                for chunk_text in sentences:
                    wav_chunk_tensor = self.model.generate(
                        chunk_text, 
                        cfg_weight=cfg, 
                        temperature=temperature
                    )
                    list_of_wav_tensors.append(wav_chunk_tensor)
            except Exception as e: # Catch errors during model.generate
                # Log the original exception e
                raise HTTPException(status_code=500, detail=f"Speech generation failed for a chunk: {str(e)}")
            
            if not list_of_wav_tensors: # Should be caught by empty sentences check, but as a safeguard
                 raise HTTPException(status_code=500, detail="No audio tensors generated despite processable sentences.")

            final_wav_tensor = torch.cat(list_of_wav_tensors, dim=1)

            # Calculate Audio Duration
            num_samples = final_wav_tensor.shape[1]
            sample_rate = self.model.sr
            duration_ms = (num_samples / sample_rate) * 1000

            # Save to in-memory buffer
            buffer = io.BytesIO()
            sf.write(buffer, final_wav_tensor.squeeze(0).cpu().numpy(), sample_rate, format="WAV")
            buffer.seek(0)

            return {"audio_buffer": buffer, "duration_ms": duration_ms}

        except HTTPException: # Re-raise HTTPExceptions directly
            raise
        except Exception as e:
            # Log the original exception e
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")
        finally:
            # Clean up temporary file
            if ref_audio_path.exists(): # Check if path exists before trying to unlink
                ref_audio_path.unlink(missing_ok=True)


    async def encode_response(self, output_data: dict):
        audio_buffer = output_data["audio_buffer"]
        duration_ms = output_data["duration_ms"]
        
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        
        return JSONResponse(content={
            "audio_base64": audio_base64,
            "duration_ms": duration_ms,
            "media_type": "audio/wav" # Hint for client
        })


if __name__ == "__main__":
    api = ChatterboxLitAPI()
    app = FastAPI()

    @app.post("/predict")
    async def predict_endpoint(request: TTSRequest):
        # This local test endpoint needs to simulate the LitServe flow, including error handling
        try:
            if not hasattr(api, "model") or api.model is None:
                api.setup()

            async with AsyncClient() as client:
                decoded_input = await api.decode_request(request, client)
            
            prediction_output = await api.predict(decoded_input)
            
            response = await api.encode_response(prediction_output)
            return response
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except Exception as e: # Catch any other unexpected errors for the test endpoint
            return JSONResponse(status_code=500, content={"detail": f"An unexpected error occurred in test endpoint: {str(e)}"})


    # To run this FastAPI app directly (for testing purposes):
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    # print("FastAPI server running on port 8000. POST to /predict.")
    # print("This direct FastAPI setup is for basic testing only.")
    # print("For production, use LitServe: `litestar run main:api` after installing chatterbox and its deps.")
    
    # The primary way to run with LitServe (as per LitServe docs):
    # 1. Ensure chatterbox is installed: pip install git+https://github.com/anotherjesse/chatterbox.git
    # 2. Run with: litestar run main:api --port 8000
    # (Assuming main.py contains `api = ChatterboxLitAPI()`)
    # LitServe will handle the server lifecycle. The __main__ block below is more for conceptual understanding
    # or if you were to integrate LitAPI into a raw FastAPI app manually.

    print("To run this LitAPI with LitServe, use the command:")
    print("`litestar run main:api --port 8000`")
    print("Ensure you have installed chatterbox and its dependencies first.")
    print("For example: pip install git+https://github.com/anotherjesse/chatterbox.git")
    print("You may also need: pip install librosa soundfile")
