@echo off
if not exist .venv (
    echo ERROR: No .venv found. Run setup.bat first.
    exit /b 1
)
uv run flux-bend %*
