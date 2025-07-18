@echo off
echo "Please have Python 3.12"
echo "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"

echo ""
echo "Checking for eSpeak NG"
winget install --id=eSpeak-NG.eSpeak-NG  -e --silent --accept-package-agreements --accept-source-agreements

echo ""
echo "Installing BuildTools"
winget install --id=Microsoft.VisualStudio.2022.BuildTools  -e

echo "Making venv"
py -3.12 -m venv --clear --upgrade-deps .venv
if %errorlevel% neq 0 (
    echo "Error installing requirements. Please check the output above."
    goto :EOF
)
call ".venv/scripts/activate.bat"

echo "Installing. Please Wait...."
python -m pip install --upgrade pip
pip --disable-pip-version-check install --no-clean -r requirements.txt

if %errorlevel% neq 0 (
    echo "Error installing requirements. Please check the output above."
    goto :EOF
)
 

echo " Installation complete."
echo If all worked ok then run:
echo 2_Start_Zonos.bat

:EOF
call ".venv\Scripts\deactivate.bat"
