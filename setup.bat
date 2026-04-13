@echo off
echo === flux-bend setup ===

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: uv is not installed. Install from https://docs.astral.sh/uv/
    exit /b 1
)

echo Creating venv with Python 3.12...
uv venv --python 3.12

echo Installing dependencies...
uv pip install -e ".[analysis]" --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install git+https://github.com/huggingface/diffusers.git

echo.
echo Setup complete. Run with: run.bat [command] [args]
echo Example: run.bat test --model black-forest-labs/FLUX.2-klein-base-4B --prompt "test" --seed 42
echo.
echo NOTE: run 'huggingface-cli login' to authenticate with HuggingFace before first use.
