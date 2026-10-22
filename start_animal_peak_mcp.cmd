@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
if defined ANIMAL_PEAK_MCP_PYTHON (
    set "ANIMAL_PEAK_PYTHON=%ANIMAL_PEAK_MCP_PYTHON%"
) else if exist "%~dp0.venv-mcp\Scripts\python.exe" (
    set "ANIMAL_PEAK_PYTHON=%~dp0.venv-mcp\Scripts\python.exe"
) else (
    set "ANIMAL_PEAK_PYTHON=D:\software\Anaconda\envs\GUI\python.exe"
)
if not exist "%ANIMAL_PEAK_PYTHON%" (
    >&2 echo Set ANIMAL_PEAK_MCP_PYTHON to the GUI environment python.exe.
    exit /b 2
)
"%ANIMAL_PEAK_PYTHON%" -u "%~dp0animal_peak_mcp_server.py" %*
exit /b %errorlevel%
