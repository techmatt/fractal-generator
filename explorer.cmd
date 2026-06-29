@echo off
REM Launch the fractal Explorer Flask server and open it in the browser.
cd /d "%~dp0"

REM Open the page once the server has had a moment to boot (smoke test runs first).
start "" cmd /c "timeout /t 4 >nul & start """" http://127.0.0.1:5005"

REM Run the server in this window (Ctrl+C to stop).
uv run python tools/explorer/app.py
