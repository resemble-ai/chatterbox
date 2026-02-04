import logging
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

# Speech Recognition logic updated for Malayalam (ml-IN) - By Ahmed Shajahan
class SpeechRecognizer:
    def __init__(self, model_id="openai/whisper-tiny", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        logger.info(f"Loading speech recognizer ({model_id}) on {device}...")

        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=device
            )
        except Exception as e:
            logger.error(f"Failed to load speech recognizer: {e}")
            self.pipe = None

    def transcribe(self, audio_path, language_id=None):
        """
        Transcribe audio file to text.
        Args:
            audio_path: Path to the audio file.
            language_id: Optional language code (e.g., 'en', 'ml').
        """
        if self.pipe is None:
            return "Error: Speech recognition model not loaded."

        generate_kwargs = {}

        # Handle specific language codes
        if language_id:
            # Speech Recognition logic updated for Malayalam (ml-IN) - By Ahmed Shajahan
            if language_id == "ml":
                language_id = "malayalam"
                # Note: Whisper uses 'malayalam' or 'ml'. 'ml-IN' is not supported by the model directly.
                logger.info(f"Processing Malayalam audio (code: {language_id})")

            # Whisper pipeline mostly auto-detects, but we can force language if supported
            generate_kwargs["language"] = language_id
            pass

        try:
            result = self.pipe(audio_path, generate_kwargs=generate_kwargs)
            return result["text"]
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return f"Error during transcription: {e}"
