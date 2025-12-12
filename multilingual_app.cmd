@echo off
echo Starting Chatterbox TTS Gradio App...
echo.

call .venv\Scripts\activate.bat
start http://127.0.0.1:7860
python multilingual_app.py

if errorlevel 1 pause