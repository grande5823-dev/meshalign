#!/usr/bin/env bash
# Start the meshalign alignment console webapp.
#
# Usage:
#   ./start.sh                          # localhost only, port 8766
#   ./start.sh 9000                     # custom port, localhost only
#   ./start.sh --network                # accessible from the network
#   ./start.sh --network 9000           # network + custom port
#   ./start.sh --background             # detached (logs to webapp.log)
#
# Finds the "meshalign" conda env automatically — no need to activate it.
# Alignments are saved to webapp/runs/ and survive restarts.

set -e

PORT=8766
HOST="127.0.0.1"
BACKGROUND=false

# --- parse args -----------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --network)
            HOST="0.0.0.0"
            ;;
        --background)
            BACKGROUND=true
            ;;
        --help|-h)
            echo "Usage: ./start.sh [--network] [--background] [port]"
            echo ""
            echo "  --network     expose to the local network (default: localhost only)"
            echo "  --background  run detached (logs to webapp.log)"
            echo "  port          port number (default: 8766)"
            echo ""
            echo "Examples:"
            echo "  ./start.sh                   localhost:8766"
            echo "  ./start.sh 9000              localhost:9000"
            echo "  ./start.sh --network         network:8766"
            echo "  ./start.sh --network 9000    network:9000"
            exit 0
            ;;
        *)
            PORT="$1"
            ;;
    esac
    shift
done

if [ "$HOST" = "0.0.0.0" ]; then
    URL="http://0.0.0.0:$PORT"
else
    URL="http://localhost:$PORT"
fi

# --- find the conda env's Python ------------------------------------------
PY=""
ENVNAME=meshalign

if command -v conda >/dev/null 2>&1; then
    ENVDIR=$(conda env list 2>/dev/null | grep "^$ENVNAME " | awk '{print $NF}')
    if [ -n "$ENVDIR" ] && [ -x "$ENVDIR/bin/python" ]; then
        PY="$ENVDIR/bin/python"
    fi
fi

if [ -z "$PY" ]; then
    for d in \
        "$HOME/miniconda3/envs/$ENVNAME" \
        "$HOME/anaconda3/envs/$ENVNAME" \
        "/opt/miniconda3/envs/$ENVNAME" \
        "/opt/anaconda3/envs/$ENVNAME" \
        "/usr/local/miniconda3/envs/$ENVNAME"; do
        if [ -x "$d/bin/python" ]; then
            PY="$d/bin/python"
            break
        fi
    done
fi

if [ -z "$PY" ]; then
    for p in python3 python; do
        if command -v "$p" >/dev/null 2>&1; then
            if "$p" -c "import fastapi, uvicorn, open3d" >/dev/null 2>&1; then
                PY="$p"
                break
            fi
        fi
    done
fi

if [ -z "$PY" ]; then
    echo "Could not find the '$ENVNAME' conda environment."
    echo ""
    echo "Make sure you ran ./install.sh first, or try:"
    echo "  conda activate $ENVNAME"
    echo "  ./start.sh"
    exit 1
fi

echo "Starting meshalign console on $URL"
echo "  Python: $PY"
echo "  Alignments saved to: $(dirname "$0")/webapp/runs/"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/webapp"

if [ "$BACKGROUND" = true ]; then
    nohup "$PY" -m uvicorn app:app --host "$HOST" --port "$PORT" \
        > "$SCRIPT_DIR/webapp/webapp.log" 2>&1 &
    echo "Started in background (PID $!)."
    echo "  Logs: $SCRIPT_DIR/webapp/webapp.log"
    echo "  Stop: kill $!"
else
    exec "$PY" -m uvicorn app:app --host "$HOST" --port "$PORT"
fi
