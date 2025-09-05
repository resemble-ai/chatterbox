<#
PowerShell installation script converted from 1_Install.bat
Targets Windows 10+. 
create a Python 3.12 venv, activate it, install pip requirements, and exit with non-zero on failure.

Notes:
- Requires winget (App Installer) on Windows 10; if winget is not present the script will skip package installs and warn.
- Uses Start-Process with -Wait for UI installers where necessary.
#>

function Print-Banner {
    $banner = @'
  ad88888ba   88                                 88                      888b      88                       
 d8"     "8b  88                                 ""                      8888b     88                ,d     
 Y8,          88                                                         88 `8b    88                88     
 `Y8aaaaa,    88   ,d8  8b       d8  8b,dPPYba,  88  88,dPYba,,adPYba,   88  `8b   88   ,adPPYba,  MM88MMM  
   `""""""8b,  88 ,a8"   `8b     d8'  88P'   "Y8  88  88P'   "88"    "8a  88   `8b  88  a8P_____88    88     
         `8b  8888[      `8b   d8'   88          88  88      88      88  88    `8b 88  8PP"""""""    88     
 Y8a     a8P  88`"Yba,    `8b,d8'    88          88  88      88      88  88     `8888  "8b,   ,aa    88,    
  "Y88888P"   88   `Y8a     Y88'     88          88  88      88      88  88      `888   `"Ybbd8"'    "Y888  
                            d8'                                                  
                           d8'      ChatterBox                                            

'@
    Write-Host $banner
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Header($text) {
    Write-Host "`n=== $text ===" -ForegroundColor Cyan
}

# Temporary file path
$tempFile = Join-Path $env:TEMP "winget_output.txt"

function Test-WingetAvailable {
    try {
        Get-Command winget -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

Clear-Host
Print-Banner

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

$wingetPresent = Test-WingetAvailable
if (-not $wingetPresent) {
    Write-Warning "winget not found. Skipping winget-based installs. On Windows 10 you may need to install App Installer from the Microsoft Store."
}

Write-Header "Checking if Python 3.12 is installed"
if ($wingetPresent) {
    winget list --id Python.Python.3.12 --accept-source-agreements > $tempFile 2>&1
    $found = Select-String -Path $tempFile -Pattern 'Python.Python.3.12' -SimpleMatch -Quiet
    if (-not $found) {
        Write-Host "Python.Python.3.12 is NOT installed. Installing via winget..."
        Start-Process -FilePath winget -ArgumentList 'install','--id=Python.Python.3.12','-e','--silent','--accept-package-agreements','--accept-source-agreements' -NoNewWindow -Wait
	    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Host "Python.3.12 is already installed."
    }
    Remove-Item -Path $tempFile -ErrorAction SilentlyContinue
}

Write-Header "Creating Python 3.12 virtual environment"
# Prefer the py launcher with the -3.12 argument, fall back to python if not available
$pyExe = 'py'
$pyArgs = '-3.12'
$usePyLauncher = $false
try {
    # Call py with the -3.12 argument correctly (separate args)
    & $pyExe $pyArgs -V | Out-Null
    $usePyLauncher = $true
} catch {
    Write-Warning "Python 3.12 launcher not found as 'py -3.12'. Trying 'python'."
    $pyExe = 'python'
    $pyArgs = ''
}

# Create venv in .venv
if (Test-Path -Path '.venv') {
    Write-Host "Removing existing .venv to recreate..."
    Remove-Item -Recurse -Force -Path '.venv'
}

$createVenvArgs = @()
if ($pyArgs -ne '') { $createVenvArgs += $pyArgs }
$createVenvArgs += '-m'; $createVenvArgs += 'venv'; $createVenvArgs += '.venv'; $createVenvArgs += '--clear'
$proc = Start-Process -FilePath $pyExe -ArgumentList $createVenvArgs -NoNewWindow -Wait -PassThru
if ($proc.ExitCode -ne 0) {
    Write-Error "Failed to create virtual environment using $pyCmd. ExitCode=$($proc.ExitCode)"
    exit 1
}

Write-Host "Activating virtual environment and installing requirements..."
# Activation for PowerShell uses Scripts/Activate.ps1
$activateScript = Join-Path -Path (Get-Location) -ChildPath ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Activation script not found at $activateScript"
    exit 1
}

# Dot-source the Activate.ps1 so the current session uses the venv
. $activateScript

try {
    Write-Host "Upgrading pip and installing packages from requirements.txt (be patient)"
    python -m pip install --quiet --upgrade pip
    pip install -r requirements.txt
} catch {
    Write-Error "Package installation failed: $_"
    # Deactivate if possible and exit non-zero
    if (Get-Command -ErrorAction SilentlyContinue Deactivate) { Deactivate }
    exit 1
}

# Deactivate virtualenv if present
if (Get-Command -ErrorAction SilentlyContinue Deactivate) { Deactivate }

Write-Header "Done"
Write-Host "If all succeeded you can run 2_Start.bat or 2_Start_ML.bat" -ForegroundColor Green

exit 0
