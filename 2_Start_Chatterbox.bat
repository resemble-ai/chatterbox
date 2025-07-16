@echo off

echo Starting Chatterbox Gadio Client...:

ipconfig | find /i "IPv4"

call ".venv/scripts/activate.bat"

python gradio_tts_app.py

call ".venv\Scripts\deactivate.bat"