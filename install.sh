#!/usr/bin/env bash
# Set up the meshalign webapp environment (Linux / macOS, conda or venv).
#
# Usage:
#   ./install.sh                  # creates conda env "meshalign"
#   ./install.sh myenv            # custom env name
#   ./install.sh --venv           # use python -m venv instead of conda
#
# After this runs once, start the webapp with: ./start.sh

set -e

ENVNAME="${1:-meshalign}"
USE_VENV=false
[ "$1" = "--venv" ] && { ENVNAME="meshalign"; USE_VENV=true; }

# --- find conda or fall back to venv ------------------------------------
if command -v conda >/dev/null 2>&1; then
    CONDA=conda
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA="$HOME/miniconda3/bin/conda"
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
    CONDA="$HOME/anaconda3/bin/conda"
else
    CONDA=""
    if [ "$USE_VENV" = false ]; then
        echo "conda not found — falling back to python -m venv"
        USE_VENV=true
    fi
fi

# --- install ---------------------------------------------------------------
if [ "$USE_VENV" = true ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    echo "Installing packages..."
    pip install python-multipart -e "$(dirname "$0")" fastapi "uvicorn[standard]" open3d numpy matplotlib scipy pandas
    echo ""
    echo "Setup complete! Start with:"
    echo "  source venv/bin/activate"
    echo "  ./start.sh"
else
    echo "Found conda: $CONDA"
    if $CONDA env list 2>/dev/null | grep -q "^$ENVNAME "; then
        echo "Environment '$ENVNAME' already exists — updating packages..."
    else
        echo "Creating environment '$ENVNAME'..."
        $CONDA create -n "$ENVNAME" python=3.11 -y
    fi
    echo ""
    echo "Installing packages (this may take a few minutes)..."
    $CONDA run -n "$ENVNAME" pip install python-multipart -e "$(dirname "$0")" fastapi "uvicorn[standard]" open3d numpy matplotlib scipy pandas
    echo ""
    echo "Verifying..."
    $CONDA run -n "$ENVNAME" python -c "import fastapi, uvicorn, open3d, numpy; print('All packages OK')"
    echo ""
    echo "Setup complete! Start with:"
    echo "  conda activate $ENVNAME"
    echo "  ./start.sh"
fi
