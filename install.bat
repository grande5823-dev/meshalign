@echo off
REM Set up the meshalign webapp environment (Miniconda/Anaconda on Windows).
REM
REM Usage:
REM   install.bat                    (creates env "meshalign", installs deps)
REM   install.bat myenv              (uses a custom env name)
REM
REM After this runs once, start the webapp with: start.bat

setlocal enabledelayedexpansion

set ENVNAME=%1
if "%ENVNAME%"=="" set ENVNAME=meshalign

REM --- find conda ---------------------------------------------------------
set CONDA=
for %%c in ("%USERPROFILE%\miniconda3\Scripts\conda.exe" ^
            "%USERPROFILE%\anaconda3\Scripts\conda.exe" ^
            "C:\ProgramData\miniconda3\Scripts\conda.exe" ^
            "C:\ProgramData\anaconda3\Scripts\conda.exe" ^
            "C:\miniconda3\Scripts\conda.exe" ^
            "C:\anaconda3\Scripts\conda.exe") do (
    if exist %%c set CONDA=%%~c
)

if "!CONDA!"=="" (
    where conda >nul 2>&1
    if !errorlevel!==0 (
        set CONDA=conda
    )
)

if "!CONDA!"=="" (
    echo Could not find conda. Install Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    echo Then re-run this script.
    pause
    exit /b 1
)

echo Found conda: !CONDA!
echo.

REM --- create or update the environment ------------------------------------
echo Creating/updating environment "%ENVNAME%"...

!CONDA! env list | findstr /c:"%ENVNAME% " >nul 2>&1
if !errorlevel!==0 (
    echo Environment "%ENVNAME%" already exists — updating packages...
) else (
    echo Creating new environment "%ENVNAME%"...
    !CONDA! create -n %ENVNAME% python=3.11 -y
    if !errorlevel! neq 0 (
        echo Failed to create environment.
        pause
        exit /b 1
    )
)

REM --- install packages -----------------------------------------------------
echo.
echo Installing packages (this may take a few minutes)...
echo.

!CONDA! run -n %ENVNAME% pip install python-multipart -e "%~dp0." fastapi "uvicorn[standard]" open3d numpy matplotlib scipy pandas

if !errorlevel! neq 0 (
    echo.
    echo Package installation failed. Try manually:
    echo   conda activate %ENVNAME%
    echo   pip install fastapi "uvicorn[standard]" open3d numpy matplotlib scipy pandas
    pause
    exit /b 1
)

REM --- verify ----------------------------------------------------------------
echo.
echo Verifying installation...
!CONDA! run -n %ENVNAME% python -c "import fastapi, uvicorn, open3d, numpy; print('All packages OK')"

if !errorlevel! neq 0 (
    echo.
    echo Verification failed. See the error above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To start the webapp:
echo    conda activate %ENVNAME%
echo    start.bat
echo ============================================================
echo.
pause
