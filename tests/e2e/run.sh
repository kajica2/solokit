#!/usr/bin/env bash
# Run the solokit e2e test suite.
#
# Starts a uvicorn server in the background on a free port, runs the Puppeteer
# test against it, then tears the server down. Exits non-zero on any failure.
#
# Usage:
#   tests/e2e/run.sh              # auto-pick a port
#   PORT=9000 tests/e2e/run.sh    # use a specific port
#
# Prereqs:
#   - Python deps installed (solokit + [api] extra)
#   - Node deps installed:  (cd tests/e2e && npm install)

set -euo pipefail

# Resolve the project root from this script's location
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PORT="${PORT:-8765}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON=python
fi

LOGFILE="$(mktemp -t solokit-e2e.XXXXXX.log)"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "${KEEP_LOG:-0}" = "1" ]; then
        echo "Server log: $LOGFILE"
    else
        rm -f "$LOGFILE"
    fi
}
trap cleanup EXIT INT TERM

# --- preflight ---
echo "→ Installing npm deps (if needed)..."
(cd tests/e2e && [ -d node_modules ] || npm install --silent)

# --- start server ---
echo "→ Starting solokit API on http://127.0.0.1:$PORT ..."
"$PYTHON" -m uvicorn solokit.api.server:app \
    --host 127.0.0.1 --port "$PORT" --no-access-log \
    > "$LOGFILE" 2>&1 &
SERVER_PID=$!

# Wait for the server to be ready (max 15s)
for i in {1..30}; do
    if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
    echo "✗ Server failed to start. Log:"
    cat "$LOGFILE"
    exit 1
fi
echo "✓ Server up"

# --- run puppeteer ---
echo "→ Running Puppeteer test..."
SOLOKIT_URL="http://127.0.0.1:$PORT" node tests/e2e/frontend.test.mjs
EXIT=$?

# Screenshots get left in place for inspection
echo
echo "Screenshots: tests/e2e/screenshots/"
ls -1 tests/e2e/screenshots/ 2>/dev/null || true

exit $EXIT
