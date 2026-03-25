#!/bin/bash
# Voice Agent launcher — starts Python backend + Electron UI
# Run from Terminal.app for mic access
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# Check python3
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.11+"
    exit 1
fi

# Check node/npx
if ! command -v npx &>/dev/null; then
    echo "ERROR: npx not found. Install Node.js"
    exit 1
fi

# Check electron installed
if [ ! -d "$DIR/electron/node_modules/electron" ]; then
    echo "Installing Electron..."
    cd "$DIR/electron" && npm install
fi

# Load env
[ -f "$DIR/.env" ] && { set -a; source "$DIR/.env"; set +a; }
export PYTHONPATH="$DIR"

# Kill ALL existing instances (app, electron, packaged app)
pkill -f "app.py.*--no-browser" 2>/dev/null || true
pkill -f "Voice Agent" 2>/dev/null || true
pkill -f "Electron.*voice-agent" 2>/dev/null || true
sleep 0.5

# Check if port is free
if lsof -i :8528 -sTCP:LISTEN &>/dev/null; then
    echo "Port 8528 is in use. Killing existing process..."
    lsof -ti :8528 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Fix SSL certs for macOS Python
SSL_CERT=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
if [ -n "$SSL_CERT" ]; then
    export SSL_CERT_FILE="$SSL_CERT"
    export REQUESTS_CA_BUNDLE="$SSL_CERT"
fi

echo "Starting Voice Agent..."

# Start Python backend from /tmp (numpy cwd fix)
cd /tmp
python3 "$DIR/app.py" --no-browser &
PY_PID=$!

# Verify PID is valid
if ! kill -0 $PY_PID 2>/dev/null; then
    echo "ERROR: Python backend failed to start"
    exit 1
fi
echo "  Backend PID: $PY_PID"

# Wait for backend
echo -n "  Waiting for backend"
READY=0
for i in $(seq 1 40); do
    if curl -s http://127.0.0.1:8528/api/health 2>/dev/null | grep -q '"ok":true'; then
        echo " ready!"
        READY=1
        break
    fi
    echo -n "."
    sleep 0.5
done

if [ $READY -eq 0 ]; then
    echo " TIMEOUT — backend didn't start in 20s"
    kill $PY_PID 2>/dev/null
    exit 1
fi

# Start Electron
echo "  Starting UI..."
cd "$DIR/electron"
VOICE_AGENT_EXTERNAL_BACKEND=1 npx electron . &

echo ""
echo "Voice Agent is running."
echo "  Hold Right Option (⌥) to dictate."
echo "  Press Ctrl+C to stop."
echo ""

# Cleanup on exit
cleanup() {
    echo "Stopping..."
    kill $PY_PID 2>/dev/null
    pkill -f "Electron.*voice-agent" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

wait $PY_PID
