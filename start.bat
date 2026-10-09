@echo off
REM Start the meshalign alignment console webapp.
REM
REM Usage:
REM   start.bat                          (localhost only, port 8766)
REM   start.bat 9000                     (custom port, localhost only)
REM   start.bat --network                (accessible from the network)
REM   start.bat --network 9000           (network + custom port)
REM
REM Finds the "meshalign" conda env automatically — no need to activate it.
REM Alignments are saved to webapp\runs\ and survive restarts.

setlocal enabledelayedexpansion

set PORT=8766
set HOST=127.0.0.1

REM --- parse args ---------------------------------------------------------
:parse
if "%~1"=="" goto :done_parse
if /i "%~1"=="--network" (
    set HOST=0.0.0.0
    shift
    goto :parse
)
if /i "%~1"=="--help" goto :usage
set PORT=%~1
shift
goto :parse
:done_parse

if "%HOST%"=="0.0.0.0" (
    set URL=http://0.0.0.0:%PORT%
) else (
    set URL=http://localhost:%PORT%
)

REM --- find the conda env's Python --------------------------------------
set PY=
set ENVNAME=meshalign

where conda >nul 2>&1
if !errorlevel!==0 (
    for /f "tokens=*" %%i in ('conda env list ^| findstr /c:"%ENVNAME% "') do (
        for /f "tokens=1" %%j in ("%%i") do (
            if "%%j"=="%ENVNAME%" (
                for /f "tokens=*" %%k in ("%%i") do (
                    set ENVDIR=%%k
                    set ENVDIR=!ENVDIR:*%ENVNAME% =!
                )
            )
        )
    )
)

if "!PY!"=="" (
    if not defined ENVDIR (
        for %%d in (
            "%USERPROFILE%\miniconda3\envs\%ENVNAME%"
            "%USERPROFILE%\anaconda3\envs\%ENVNAME%"
            "C:\ProgramData\miniconda3\envs\%ENVNAME%"
            "C:\ProgramData\anaconda3\envs\%ENVNAME%"
            "C:\miniconda3\envs\%ENVNAME%"
            "C:\anaconda3\envs\%ENVNAME%"
        ) do (
            if exist "%%~d\python.exe" (
                set PY=%%~d\python.exe
            )
        )
    )
)

if "!PY!"=="" (
    if defined ENVDIR (
        if exist "!ENVDIR!\python.exe" (
            set PY=!ENVDIR!\python.exe
        )
    )
)

if "!PY!"=="" (
    for %%p in (python py python3) do (
        if "!PY!"=="" (
            %%p -c "import fastapi, uvicorn, open3d" >nul 2>&1
            if !errorlevel!==0 set PY=%%p
        )
    )
)

if "!PY!"=="" (
    echo Could not find the "%ENVNAME%" conda environment.
    echo.
    echo Make sure you ran install.bat first, or try:
    echo   conda activate %ENVNAME%
    echo   start.bat
    pause
    exit /b 1
)

echo Starting meshalign console on %URL%
echo   Python: !PY!
echo   Alignments saved to: %~dp0webapp\runs\
echo.

cd /d "%~dp0webapp"
"!PY!" -m uvicorn app:app --host %HOST% --port %PORT%

pause
goto :eof

:usage
echo Usage: start.bat [--network] [port]
echo.
echo   --network    expose to the local network (default: localhost only)
echo   port         port number (default: 8766)
echo.
echo Examples:
echo   start.bat                  localhost:8766
echo   start.bat 9000              localhost:9000
echo   start.bat --network         network:8766
echo   start.bat --network 9000    network:9000
