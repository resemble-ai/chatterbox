
import unittest
from unittest.mock import MagicMock, patch
from chatterbox.asr import SpeechRecognizer

class TestMalayalamSupport(unittest.TestCase):
    def test_transcribe_passes_ml_in(self):
        print("Testing: 'ml' language code -> 'ml-IN' parameter passing")
        
        # Mock the pipeline so we don't load the actual heavy model
        with patch("chatterbox.asr.pipeline") as mock_pipeline:
            # Setup mock pipe instance
            mock_pipe_instance = MagicMock()
            mock_pipe_instance.return_value = {"text": "dummy text"}
            mock_pipeline.return_value = mock_pipe_instance
            
            # Initialize
            recognizer = SpeechRecognizer(device="cpu")
            dummy_audio = "dummy.wav"
            
            # Action: Call transcribe with 'ml'
            recognizer.transcribe(dummy_audio, language_id="ml")
            
            # Assert: Check if pipe was called with 'ml-IN'
            # The pipe is called as: pipe(audio_path, generate_kwargs=...)
            call_args = mock_pipe_instance.call_args
            self.assertIsNotNone(call_args, "Pipeline was not called")
            
            _, kwargs = call_args
            gen_kwargs = kwargs.get("generate_kwargs", {})
            
            print(f"Call Arguments: {gen_kwargs}")
            self.assertEqual(gen_kwargs.get("language"), "ml-IN", 
                             "Failed! Language parameter was not converted to 'ml-IN'")
            print("Success! 'ml-IN' was passed to the recognizer.")

if __name__ == "__main__":
    unittest.main()
