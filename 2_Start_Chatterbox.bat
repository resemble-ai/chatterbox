@echo off

echo""
echo Starting Chatterbox Gadio Client...:

call ".venv/scripts/activate.bat"

start "Chatterbox" /high python gradio_tts_app.py

call ".venv\Scripts\deactivate.bat"