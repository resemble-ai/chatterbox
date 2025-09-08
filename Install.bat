@echo off
setlocal EnableExtensions

REM Detect and load conda if not on PATH
where conda >nul 2>&1
if errorlevel 1 (
  for %%A in (
    "%UserProfile%\anaconda3\condabin\conda.bat"
    "C:\ProgramData\anaconda3\condabin\conda.bat"
    "%UserProfile%\miniconda3\condabin\conda.bat"
    "C:\ProgramData\miniconda3\condabin\conda.bat"
  ) do (
    if exist %%~A call "%%~A" activate
  )
)

call conda --version >nul 2>&1 || (
  color 0C
  echo.
  echo ================================================================
  echo   CONDA NOT FOUND
  echo   Install Anaconda, then reopen your terminal:
  echo     https://www.anaconda.com/download
  echo ================================================================
  color 07
  start "" "https://www.anaconda.com/download"
  pause
  exit /b 1
)

set "ENV_NAME=chatterbox"
call conda env list | findstr /i " %ENV_NAME% " >nul
if errorlevel 1 (
  echo Creating env %ENV_NAME% with Python 3.11...
  call conda create -y -n %ENV_NAME% python=3.11 || exit /b 1
) else (
  echo Env %ENV_NAME% already exists.
)

call conda activate %ENV_NAME% || exit /b 1

where git >nul 2>&1 || (echo [ERROR] git not found in PATH.& exit /b 1)

if not exist ".git" (
  echo [ERROR] This script must be run inside an existing chatterbox repo.
  echo Run: git clone https://github.com/resemble-ai/chatterbox.git
  exit /b 1
) else (
  echo Repo already present. Pulling latest...
  git pull
)

python -m pip install -U pip || exit /b 1
pip install numpy || exit /b 1
pip install -e "%cd%" || exit /b 1

echo [OK] Chatterbox installed in conda env '%ENV_NAME%'.
endlocal
