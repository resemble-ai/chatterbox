@echo off
setlocal enabledelayedexpansion

:: Chatterbox Audiobook Studio - Complete Setup and Launch
:: This script creates a virtual environment, installs dependencies, downloads models, and launches the web UI

title Chatterbox Audiobook Studio Setup

echo.
echo   ██████╗██╗  ██╗ █████╗ ████████╗ █████╗ ██████╗ ██╗     ███████╗
echo  ██╔════╝██║  ██║██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██║     ██╔════╝
echo  ██║     ███████║███████║   ██║   ███████║██████╔╝██║     █████╗  
echo  ██║     ██╔══██║██╔══██║   ██║   ██╔══██║██╔══██╗██║     ██╔══╝  
echo  ╚██████╗██║  ██║██║  ██║   ██║   ██║  ██║██████╔╝███████╗███████╗
echo   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝
echo.
echo  🎧 Enhanced Audiobook Creation with Voice Library & Multi-Voice Support

:: Set variables
set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "MODELS_DIR=%PROJECT_DIR%models"
set "VOICE_LIBRARY_DIR=%PROJECT_DIR%voice_library"
set "PYTHON_VERSION=3.10"

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python %PYTHON_VERSION% or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo 📋 Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ❌ Python %PYTHON_VERSION% or higher is required
    pause
    exit /b 1
)

echo ✅ Python version check passed

:: Create necessary directories
echo.
echo 📁 Creating project directories...
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%VOICE_LIBRARY_DIR%" mkdir "%VOICE_LIBRARY_DIR%"
if not exist "audiobook_projects" mkdir "audiobook_projects"
echo ✅ Directories created

:: Check if virtual environment exists
echo.
echo 🔧 Setting up virtual environment...
if exist "%VENV_DIR%" (
    echo ✅ Virtual environment already exists
) else (
    echo 📦 Creating new virtual environment...
    python -m venv "%VENV_DIR%"
    python -m gradio --server-name "0.0.0.0" --server-port 7860 --share --show-error --inbrowser --quiet
    echo ✅ Virtual environment created successfully
)

:: Activate virtual environment
echo.
echo 🚀 Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo.
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo.
echo 📦 Installing dependencies...
if exist "requirements.txt" (
    echo 📋 Installing from requirements.txt...
    pip install -r requirements.txt
) else (
    echo 📋 Installing core dependencies...
    pip install torch>=2.4.0 torchaudio>=2.4.0 torchvision>=0.19.0
    pip install numpy>=1.24.0 librosa>=0.10.0 gradio>=5.44.0
    pip install transformers>=4.46.0 diffusers>=0.29.0
    pip install soundfile scipy requests tqdm
)

:: Install chatterbox package in development mode
echo.
echo 📦 Installing Chatterbox TTS package...
pip install -e .

:: Check for CUDA availability
echo.
echo 🔍 Checking CUDA availability...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ⚠️ CUDA not available, will use CPU mode
) else (
    echo ✅ CUDA available - GPU acceleration enabled
)

:: Download models if not exists
echo.
echo 📥 Checking and downloading models...
python -c "
import os
import sys
from pathlib import Path

print('🔍 Checking model availability...')

# Check if models need to be downloaded
try:
    from src.chatterbox.tts import ChatterboxTTS
    from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
    
    print('📥 Downloading English TTS model...')
    try:
        model = ChatterboxTTS.from_pretrained(device='cpu')
        print('✅ English TTS model downloaded successfully')
    except Exception as e:
        print(f'⚠️ Error downloading English model: {e}')
    
    print('📥 Downloading Multilingual TTS model...')
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
        print('✅ Multilingual TTS model downloaded successfully')
    except Exception as e:
        print(f'⚠️ Error downloading multilingual model: {e}')
        
except ImportError as e:
    print(f'⚠️ Import error: {e}')
    print('💡 Models will be downloaded on first use')
"

:: Create voice library structure
echo.
echo 📚 Setting up voice library...
if not exist "%VOICE_LIBRARY_DIR%\example_narrator" mkdir "%VOICE_LIBRARY_DIR%\example_narrator"

:: Create example voice profile
echo.
echo 📝 Creating example voice profile...
python -c "
import json
import os
from pathlib import Path

voice_dir = Path('voice_library/example_narrator')
config = {
    'voice_name': 'example_narrator',
    'display_name': 'Example Narrator',
    'description': 'A sample narrator voice profile for testing',
    'exaggeration': 0.5,
    'cfg_weight': 0.5,
    'temperature': 0.8,
    'created_date': __import__('time').time(),
    'version': '1.0'
}

with open(voice_dir / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('✅ Example voice profile created')
"

:: Final setup check
echo.
echo ✅ Setup complete! Summary:
echo.
echo 📁 Project Directory: %PROJECT_DIR%
echo 🐍 Virtual Environment: %VENV_DIR%
echo 📦 Models Directory: %MODELS_DIR%
echo 📚 Voice Library: %VOICE_LIBRARY_DIR%
echo.

:: Launch the web UI
echo.
echo 🚀 Launching Chatterbox Audiobook Studio...
echo.
echo 🌐 Opening web interface at http://localhost:7860
echo 🛑 Press Ctrl+C to stop the server
echo.

python gradio_audiobook_app.py --share

:: Deactivate virtual environment when done
deactivate

echo.
echo ✅ Web UI stopped. Thank you for using Chatterbox Audiobook Studio!
pause
