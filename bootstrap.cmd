@echo off
REM ==============================================================================
REM Project : AG News Text Classification
REM Team    : Aimer PAM
REM Author  : Vo Hai Dung
REM License : MIT
REM ==============================================================================
REM One-step launcher for a Windows machine with no developer tooling.
REM
REM Batch is used instead of PowerShell because .cmd files are not subject to the
REM PowerShell execution policy, which blocks unsigned .ps1 files that arrive
REM inside a downloaded ZIP. The uv installer is invoked through a child
REM PowerShell carrying -ExecutionPolicy Bypass, so nothing has to be changed
REM machine-wide and no Administrator prompt appears. PATH is extended for this
REM process only, which removes the usual "restart your terminal" step.
REM
REM Usage:  bootstrap.cmd  [any run_demo.py flags]

setlocal
cd /d "%~dp0"

REM A double-clicked window closes the moment the script ends, taking any error
REM message with it, so failures wait for the reader in that case only. stderr
REM is discarded alongside stdout here: when a Git Bash entry sits earlier on
REM PATH its Unix find shadows the Windows one and complains about /i. Losing
REM the probe only costs the wait, so failing quietly is correct.
set "LAUNCHED_BY_CLICK="
echo %cmdcmdline% | find /i "%~nx0" >nul 2>nul && set "LAUNCHED_BY_CLICK=1"

if not exist "run_demo.py" (
    echo run_demo.py was not found next to this file.
    echo Run bootstrap.cmd from inside the project folder, not from a copy of
    echo the script on its own.
    goto fail
)

set "UV_BIN=%USERPROFILE%\.local\bin"

where uv >nul 2>nul
if %errorlevel%==0 goto run

if exist "%UV_BIN%\uv.exe" (
    set "PATH=%UV_BIN%;%PATH%"
    goto run
)

echo Installing uv into %UV_BIN% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

if not exist "%UV_BIN%\uv.exe" (
    echo.
    echo uv could not be installed automatically. A proxy or antivirus may have
    echo blocked the download.
    echo Install it by hand from https://docs.astral.sh/uv/getting-started/installation/
    echo then run this file again.
    goto fail
)
set "PATH=%UV_BIN%;%PATH%"

:run
uv run run_demo.py %*
set "RC=%errorlevel%"
if not "%RC%"=="0" goto fail
exit /b 0

:fail
REM A bounded wait rather than `pause`: it still holds a double-clicked window
REM open long enough to read the error, but returns immediately when stdin is
REM redirected, so an automated caller can never hang here.
if defined LAUNCHED_BY_CLICK (
    echo.
    timeout /t 30 >nul 2>&1
)
exit /b 1
