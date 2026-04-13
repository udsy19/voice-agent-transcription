#!/bin/bash
# Muse — start script
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Prerequisites ────────────────────────────────────────────────────────────
command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
command -v npx &>/dev/null || { echo "ERROR: npx not found"; exit 1; }
[ -d "$DIR/electron/node_modules/electron" ] || { echo "Installing Electron..."; cd "$DIR/electron" && npm install; }

# ── Kill everything first (aggressive) ───────────────────────────────────────
echo "Cleaning up..."
# Kill every possible holder: python, uvicorn, Electron
pkill -9 -f "python3.*app\.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
pkill -9 -f "Electron.*[Mm]use" 2>/dev/null || true
pkill -9 -f "electron.*main\.js" 2>/dev/null || true
# Nuke both ports by PID — try both TCP states
for port in 8528 8529; do
    lsof -ti :$port 2>/dev/null | xargs kill -9 2>/dev/null || true
done
sleep 1

# Retry loop — keep killing until port 8528 is free
for i in 1 2 3 4 5 6 7 8 9 10; do
    if ! lsof -i :8528 &>/dev/null; then break; fi
    echo "Port 8528 still held — attempt $i..."
    lsof -ti :8528 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1
done

# Final check
if lsof -i :8528 -sTCP:LISTEN &>/dev/null; then
    echo "  ERROR: Port 8528 still in use. Holder:"
    lsof -i :8528 2>/dev/null
    echo "  Run: kill -9 \$(lsof -ti :8528)"
    exit 1
fi

# ── Load env ─────────────────────────────────────────────────────────────────
[ -f "$DIR/.env" ] && { set -a; source "$DIR/.env"; set +a; }
export PYTHONPATH="$DIR"
export TOKENIZERS_PARALLELISM=false

# ── Start backend ────────────────────────────────────────────────────────────
echo "Starting Muse..."
cd /tmp
python3 "$DIR/app.py" --no-browser &
PY_PID=$!

if ! kill -0 $PY_PID 2>/dev/null; then
    echo "ERROR: Backend failed to start"
    exit 1
fi
echo "  Backend PID: $PY_PID"

# Wait for backend
echo -n "  Waiting for backend"
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8528/api/health 2>/dev/null | grep -q '"ok"'; then
        echo " ready!"
        break
    fi
    if ! kill -0 $PY_PID 2>/dev/null; then
        echo ""
        echo "ERROR: Backend crashed during startup"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Verify it actually started
if ! curl -s http://127.0.0.1:8528/api/health 2>/dev/null | grep -q '"ok"'; then
    echo ""
    echo "ERROR: Backend didn't start in 30s"
    kill -9 $PY_PID 2>/dev/null
    exit 1
fi

# ── Start Electron ───────────────────────────────────────────────────────────
echo "  Starting UI..."
cd "$DIR/electron"
VOICE_AGENT_EXTERNAL_BACKEND=1 npx electron . &

echo ""
echo "Muse is running."
echo "  ⌥L = dictate    ⌥R = assistant"
echo "  Ctrl+C to stop"
echo ""

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping..."
    kill $PY_PID 2>/dev/null
    pkill -f "Electron.*muse" 2>/dev/null
    lsof -ti :8528 | xargs kill -9 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

wait $PY_PID
