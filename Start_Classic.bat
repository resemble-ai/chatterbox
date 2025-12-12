@echo off
setlocal EnableExtensions
set "ENV_NAME=chatterbox"
cd /d "%~dp0"

REM --- Load conda if not on PATH ---
where conda >nul 2>&1
if errorlevel 1 (
  if exist "%UserProfile%\miniconda3\condabin\conda.bat" call "%UserProfile%\miniconda3\condabin\conda.bat" activate
  if exist "C:\ProgramData\miniconda3\condabin\conda.bat" call "C:\ProgramData\miniconda3\condabin\conda.bat" activate
  if exist "%UserProfile%\anaconda3\condabin\conda.bat" call "%UserProfile%\anaconda3\condabin\conda.bat" activate
  if exist "C:\ProgramData\anaconda3\condabin\conda.bat" call "C:\ProgramData\anaconda3\condabin\conda.bat" activate
)

call conda --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] conda not found. Run: conda init cmd.exe then reopen terminal.
  exit /b 1
)

REM --- Block on RTX 50xx because cu121 is not compatible ---
python -c "import torch,sys; sys.exit(1 if (torch.cuda.is_available() and 'RTX 50' in torch.cuda.get_device_name(0)) else 0)" 1>nul 2>nul
if errorlevel 1 (
  echo [ERROR] This script cannot run on RTX 50xx GPUs. Use Start_5080.bat instead.
  pause
  exit /b 1
)

REM --- Ensure env and activate ---
call conda env list | findstr /i " %ENV_NAME% " >nul
if errorlevel 1 (
  echo [INFO] Creating env %ENV_NAME% - Python 3.11
  call conda create -y -n %ENV_NAME% python=3.11
  if errorlevel 1 (
    echo [ERROR] conda create failed.
    exit /b 1
  )
) else (
  echo [INFO] Env %ENV_NAME% already exists.
)

call conda activate %ENV_NAME%
if errorlevel 1 (
  echo [ERROR] conda activate failed.
  exit /b 1
)

python -m pip install -U pip

REM --- Check if torch cu121 via conda is OK (do not reinstall if OK) ---
python -c "import sys,pkgutil; sys.exit(0 if pkgutil.find_loader('torch') else 1)"
if errorlevel 1 goto install_torch

python -c "import sys,torch; ok=torch.cuda.is_available() and (('12.1' in (getattr(torch.version,'cuda','') or '')) or ('+cu121' in getattr(torch,'__version__',''))); print('torch=',getattr(torch,'__version__','?')); print('cuda=',getattr(torch.version,'cuda',None)); print('cuda_available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu'); sys.exit(0 if ok else 1)"
if errorlevel 1 goto install_torch
goto torch_ok

:install_torch
echo [INFO] Installing torch stable cu121 via conda
pip uninstall -y torch torchvision torchaudio >nul 2>&1
call conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
if errorlevel 1 (
  echo [ERROR] conda pytorch-cuda=12.1 failed.
  exit /b 1
)

:torch_ok
python -c "import torch;print('torch=',torch.__version__);print('cuda=',getattr(torch.version,'cuda',None));print('cuda_available=',torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

REM --- Pin numeric stack via conda only if not already pinned ---
python -c "import sys; import numpy,scipy,sklearn; sys.exit(0 if (numpy.__version__.startswith('1.25.') and scipy.__version__.startswith('1.11.') and sklearn.__version__.startswith('1.3.')) else 1)" 1>nul 2>nul
if errorlevel 1 (
  echo [INFO] Ensuring numeric stack via conda
  call conda install -y numpy=1.25.* scipy=1.11.* scikit-learn=1.3.*
)

REM --- Multilingual extras (do not touch torch) ---
python -c "import sys,pkgutil; sys.exit(0 if pkgutil.find_loader('pkuseg') else 1)" 1>nul 2>nul
if errorlevel 1 pip install pkuseg

python -c "import sys,pkgutil; sys.exit(0 if pkgutil.find_loader('pykakasi') else 1)" 1>nul 2>nul
if errorlevel 1 pip install pykakasi

python -c "import sys,pkgutil; sys.exit(0 if pkgutil.find_loader('charset_normalizer') else 1)" 1>nul 2>nul
if errorlevel 1 pip install charset-normalizer

python -c "import sys,pkgutil; sys.exit(0 if pkgutil.find_loader('mcp') else 1)" 1>nul 2>nul
if errorlevel 1 pip install "gradio[mcp]"

REM --- Install repo editable WITHOUT deps (avoid touching torch) ---
python -c "import sys,pkgutil; sys.exit(0 if pkgutil.find_loader('chatterbox') else 1)" 1>nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing repo editable (no deps)
  pip install -e "%cd%" --no-deps
  if errorlevel 1 (
    echo [ERROR] editable install failed.
    exit /b 1
  )
) else (
  echo [INFO] chatterbox already importable, skipping editable install
)

REM --- Non-torch pinned deps, no-deps to avoid numpy changes ---
pip install --no-deps diffusers==0.29.0 transformers==4.46.3 librosa==0.11.0 s3tokenizer==0.2.0 conformer==0.3.2 resemble-perth==1.0.1 safetensors==0.5.3

REM --- Sanity check ---
python -c "import torch,numpy,scipy,librosa; print('torch',torch.__version__,'cuda',getattr(torch.version,'cuda',None),'cuda_ok',torch.cuda.is_available()); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('librosa',librosa.__version__)"

REM --- App menu ---
:menu
echo.
echo ================================================
echo   Select app to run   [PyTorch: conda cu121]
echo   1) gradio_tts_app.py
echo   2) gradio_vc_app.py
echo   3) multilingual_app.py
echo   Q) Quit
echo ================================================
set "CHOICE="
set /p "CHOICE=Enter choice [1/2/3/Q]: "

if /i "%CHOICE%"=="1" goto run_tts
if /i "%CHOICE%"=="2" goto run_vc
if /i "%CHOICE%"=="3" goto run_multi
if /i "%CHOICE%"=="Q" goto end
if /i "%CHOICE%"=="q" goto end
echo Invalid choice.
goto menu

:run_tts
python gradio_tts_app.py
goto menu

:run_vc
python gradio_vc_app.py
goto menu

:run_multi
python multilingual_app.py
goto menu

:end
endlocal
