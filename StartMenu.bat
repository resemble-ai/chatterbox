@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM ================================================
REM Try to make conda available without requiring "conda init"
REM ================================================
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

call conda --version >nul 2>&1
if errorlevel 1 goto no_conda

REM ================================================
REM Main menu
REM ================================================
:menu
echo.
echo ================================================
echo   Select action
echo   0) Install (Only First time !)
echo   1) Launch for Nvidia RTX 5080 (nightly CUDA 12.8 - cu128)
echo   2) Launch for Nvidia Classic (CUDA 12.1 - cu121)
echo   Q) Quit
echo ================================================
set "CHOICE="
set /p "CHOICE=Enter choice [0/1/2/Q]: "

if /i "%CHOICE%"=="0" goto run_install
if /i "%CHOICE%"=="1" goto go5080
if /i "%CHOICE%"=="2" goto goclassic
if /i "%CHOICE%"=="Q" goto end
if /i "%CHOICE%"=="q" goto end

echo [WARN] Invalid choice.
goto menu

:run_install
if exist "%~dp0Install.bat" (
  echo [INFO] Running Install.bat...
  call "%~dp0Install.bat"
) else (
  echo [ERROR] Install.bat not found next to this script.
  echo         Put Install.bat in the same folder then retry.
  pause
)
goto menu

:go5080
if exist "%~dp0Start_5080.bat" (
  call "%~dp0Start_5080.bat"
) else (
  echo [ERROR] Start_5080.bat not found next to this script.
  pause
)
goto menu

:goclassic
if exist "%~dp0Start_Classic.bat" (
  call "%~dp0Start_Classic.bat"
) else (
  echo [ERROR] Start_Classic.bat not found next to this script.
  pause
)
goto menu

:end
endlocal
exit /b 0

:no_conda
color 0C
echo.
echo ================================================================
echo   CONDA NOT FOUND
echo   This project requires Anaconda or Miniconda on Windows.
echo.
echo   Get Anaconda (recommended):
echo     https://www.anaconda.com/download
echo.
echo   Alternative (Miniconda):
echo     https://docs.conda.io/en/latest/miniconda.html
echo.
echo   After installation, just reopen this StartMenu.bat.
echo   You do NOT need to run "conda init".
echo ================================================================
color 07
start "" "https://www.anaconda.com/download"
REM Optional: also open Miniconda page
REM start "" "https://docs.conda.io/en/latest/miniconda.html"
pause
endlocal
exit /b 1
