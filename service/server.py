import asyncio
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.parse import quote

import torch
import torchaudio as ta
import uvicorn
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from .utils import setup_logger


class ListVoiceNameResponse(BaseModel):
    """Response model for listing available voice names.

    Contains a dictionary mapping voice keys to their display names.
    """
    voice_names: dict[str, str]


class GenerateAudioRequest(BaseModel):
    """Request model for generating audio from text.

    Contains the text to synthesize and the voice key to use for synthesis.
    """
    text: str
    voice_key: str


class FastAPIServer:
    """FastAPI server for text-to-speech audio generation.

    Provides HTTP endpoints for listing available voices and generating
    audio from text using the ChatterboxMultilingualTTS model.
    """

    def __init__(
        self,
        audio_prompts_dir: str,
        checkpoint_dir: str | None = None,
        device: str | None = None,
        enable_cors: bool = False,
        host: str = '0.0.0.0',
        port: int = 80,
        startup_event_listener: None | list = None,
        shutdown_event_listener: None | list = None,
        logger_cfg: None | dict = None,
    ) -> None:
        """Initialize the FastAPI server.

        Sets up the FastAPI application, configures CORS if enabled,
        registers event listeners, and initializes the TTS model.

        Args:
            audio_prompts_dir (str):
                Directory path containing audio prompt files.
                Files should be named as '{voice_key}_{language_id}.wav'.
            checkpoint_dir (str | None, optional):
                Directory path to load TTS model checkpoint from.
                If None, loads pretrained model from HuggingFace.
                Defaults to None.
            device (str | None, optional):
                Device to run the model on ('cuda', 'mps', or 'cpu').
                If None, automatically selects based on availability.
                Defaults to None.
            enable_cors (bool, optional):
                Whether to enable CORS middleware. Defaults to False.
            host (str, optional):
                Host address to bind the server to. Defaults to '0.0.0.0'.
            port (int, optional):
                Port number to bind the server to. Defaults to 80.
            startup_event_listener (None | list, optional):
                List of startup event listener functions.
                Defaults to None.
            shutdown_event_listener (None | list, optional):
                List of shutdown event listener functions.
                Defaults to None.
            logger_cfg (None | dict, optional):
                Logger configuration, see `setup_logger` for detailed
                description. Logger name will use the class name.
                Defaults to None.
        """
        logger_name = self.__class__.__name__
        if logger_cfg is None:
            logger_cfg = dict(logger_name=logger_name)
        else:
            logger_cfg = logger_cfg.copy()
            logger_cfg["logger_name"] = logger_name
        self.logger_cfg = logger_cfg
        self.logger = setup_logger(**logger_cfg)
        self.audio_prompts_dir = audio_prompts_dir
        self._load_audio_prompts()
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        # for fastapi
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.enable_cors = enable_cors
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=['*'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*'],
            )
        if startup_event_listener is not None:
            for listener in startup_event_listener:
                self.app.add_event_handler('startup', listener)
        if shutdown_event_listener is not None:
            for listener in shutdown_event_listener:
                self.app.add_event_handler('shutdown', listener)
        self._build_tts_model()
        self.last_audio_prompt_key: str | None = None
        self.model_lock = Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _load_audio_prompts(self) -> None:
        """Load audio prompt files from the configured directory.

        Scans the audio prompts directory for WAV files following the naming
        convention '{voice_key}_{language_id}.wav'. Valid files are registered
        in the audio_prompts dictionary for use in voice synthesis.
        """
        self.audio_prompts = dict()
        for file in os.listdir(self.audio_prompts_dir):
            if file.endswith('.wav'):
                file_name = file.split('.')[0]
                splits = file_name.split('_')
                if len(splits) != 2:
                    self.logger.warning(f"Invalid file name: {file_name}, skipping")
                    continue
                voice_key, language_id = splits
                self.audio_prompts[file_name] = dict(
                    name=voice_key,
                    language_id=language_id,
                    path=os.path.join(self.audio_prompts_dir, file)
                )

    def _build_tts_model(self) -> None:
        """Initialize and load the TTS model.

        Determines the appropriate device (CUDA, MPS, or CPU) based on
        availability if device is not specified. Loads the TTS model from
        local checkpoint directory if provided, otherwise loads the
        pretrained model from HuggingFace.
        """
        if self.device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            self.logger.info(f"Device not specified, using device automatically: {device}")
        else:
            device = self.device
        if self.checkpoint_dir is not None and os.path.exists(self.checkpoint_dir):
            self.tts_model = ChatterboxMultilingualTTS.from_local(self.checkpoint_dir, device)
        else:
            msg = "Checkpoint directory not specified or does not exist, using pretrained model"
            self.logger.info(msg)
            self.tts_model = ChatterboxMultilingualTTS.from_pretrained(device)

    def _generate_audio(self, voice_key: str, text: str) -> io.BytesIO:
        """Generate audio from text using the specified voice.

        This method handles the core audio generation logic, including voice
        prompt preparation, text-to-speech synthesis, and audio format conversion.
        It uses thread-safe locking to ensure model access is serialized.

        Args:
            voice_key (str):
                Voice key identifier matching an audio prompt file name
                (without extension).
            text (str):
                Text content to synthesize into speech.

        Returns:
            io.BytesIO:
                BytesIO buffer containing the generated WAV audio file
                in PCM format (16-bit signed integer).
        """
        with self.model_lock:
            if self.last_audio_prompt_key is None or \
                    self.last_audio_prompt_key != voice_key:
                audio_prompt_path = self.audio_prompts[voice_key]['path']
                prepare_start_time = time.time()
                self.tts_model.prepare_conditionals(audio_prompt_path)
                prepare_end_time = time.time()
                self.logger.info(f"Prepare time: {prepare_end_time - prepare_start_time:.2f} seconds for {voice_key}")
                self.last_audio_prompt_key = voice_key
            language_id = self.audio_prompts[voice_key]['language_id']
            generate_start_time = time.time()
            tensor_wav = self.tts_model.generate(text, language_id=language_id)
            generate_end_time = time.time()
            # Calculate duration directly from tensor shape
            duration = tensor_wav.shape[-1] / self.tts_model.sr
            wav_io = io.BytesIO()
            # Save as PCM format WAV (16-bit signed integer) for compatibility
            ta.save(
                uri=wav_io,
                src=tensor_wav,
                sample_rate=self.tts_model.sr,
                channels_first=True,
                format='wav',
                encoding='PCM_S',
                bits_per_sample=16
            )
        self.logger.info(
            f"Generate time: {generate_end_time - generate_start_time:.2f} seconds " +
            f"for {voice_key}, duration: {duration:.2f} seconds")
        return wav_io

    def _add_api_routes(self, router: APIRouter) -> None:
        """Add API routes to the router.

        This method registers all HTTP endpoints with the provided FastAPI router,
        including voice listing, audio generation, health checks, and root redirect.

        Args:
            router (APIRouter):
                FastAPI router to add routes to.
        """
        # GET routes
        router.add_api_route(
            "/api/v1/list_voice_names",
            self.list_voice_names,
            methods=["GET"],
            response_model=ListVoiceNameResponse,
        )
        router.add_api_route(
            "/api/v1/generate_audio",
            self.generate_audio,
            methods=["POST"]
        )
        router.add_api_route(
            "/",
            self.root,
            methods=["GET"],
        )
        router.add_api_route(
            '/health',
            endpoint=self.health,
            status_code=200,
            methods=['GET'],
        )

    def run(self) -> None:
        """Run this FastAPI service according to configuration.

        Registers all API routes and starts the uvicorn server with the
        configured host, port, and application settings. This method
        blocks until the server is stopped.
        """
        router = APIRouter()
        self._add_api_routes(router)
        self.app.include_router(router)
        uvicorn.run(self.app, host=self.host, port=self.port)

    def root(self) -> RedirectResponse:
        """Redirect to API documentation.

        Returns:
            RedirectResponse:
                Redirect response to /docs endpoint.
        """
        return RedirectResponse(url="/docs")

    async def health(self) -> JSONResponse:
        """Health check endpoint for service monitoring.

        This endpoint provides a simple health check that returns
        an 'OK' status to indicate the service is running properly.
        Used by load balancers and monitoring systems.

        Returns:
            JSONResponse:
                JSON response containing 'OK' status string.
        """
        resp = JSONResponse(content='OK')
        return resp

    async def list_voice_names(self) -> ListVoiceNameResponse:
        """List all available voice names.

        Scans the audio prompts directory and returns all available
        voice configurations that can be used for text-to-speech synthesis.

        Returns:
            ListVoiceNameResponse:
                Response containing a dictionary mapping voice keys
                to their display names.
        """
        voice_names = dict()
        for key in self.audio_prompts:
            voice_names[key] = self.audio_prompts[key]['name']
        resp = ListVoiceNameResponse(voice_names=voice_names)
        return resp

    async def generate_audio(self, request: GenerateAudioRequest) -> Response:
        """Generate audio from text using the specified voice.

        Synthesizes speech from the input text using the TTS model with
        the specified voice. If the voice prompt needs to be loaded,
        it will be prepared before generation. The generated audio is
        returned as a WAV file with appropriate download headers.

        Args:
            request (GenerateAudioRequest):
                Request containing the text to synthesize and the voice
                key to use.

        Returns:
            Response:
                HTTP response containing the generated audio as a WAV
                file with appropriate headers for download.
        """
        if request.voice_key not in self.audio_prompts:
            msg = f"Voice key {request.voice_key} not found"
            self.logger.error(msg)
            raise HTTPException(status_code=404, detail=msg)
        loop = asyncio.get_event_loop()
        wav_io = await loop.run_in_executor(
            self.thread_pool, self._generate_audio, request.voice_key, request.text)
        wav_io.seek(0)
        resp = Response(
            content=wav_io.getvalue(),
            media_type="audio/wav")
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f'{request.voice_key}_{timestamp_str}.wav'
        # Use RFC 5987 standard encoding for filename to support non-ASCII characters
        encoded_filename = quote(filename, safe='')
        resp.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{encoded_filename}"
        return resp
