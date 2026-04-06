#!/usr/bin/env bash
# ==============================================================
#  AgriBot – Startup Script
#  Usage:  ./start.sh
# ==============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SERVER_PID=""
TUNNEL_PID=""

# ------------------------------------------------------------------
# Cleanup on exit (Ctrl+C or script end)
# ------------------------------------------------------------------
cleanup() {
    echo ""
    echo "🛑 Shutting down AgriBot…"

    if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "   Stopping Cloudflare tunnel (PID $TUNNEL_PID)…"
        kill "$TUNNEL_PID" 2>/dev/null
        wait "$TUNNEL_PID" 2>/dev/null
    fi

    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "   Stopping uvicorn server (PID $SERVER_PID)…"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi

    echo "✅ AgriBot stopped. Goodbye!"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------
echo "╔══════════════════════════════════════════╗"
echo "║           🌱  AgriBot v1.0  🤖          ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Motor · Camera · Pump · AI Detection   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------------
if [[ ! -d "venv" ]]; then
    echo "⚠️  No virtual environment found. Creating one…"
    python3 -m venv venv
    source ./venv/bin/activate
    echo "📦 Installing dependencies…"
    pip install -r requirements.txt
else
    source ./venv/bin/activate
fi
echo "✅ Virtual environment activated ($(python3 --version))"
echo ""

# ------------------------------------------------------------------
# Start the FastAPI server
# ------------------------------------------------------------------
echo "🚀 Starting AgriBot API server…"
uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo -n "   Waiting for server"
for i in {1..10}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo ""
        echo "✅ Server is live at http://0.0.0.0:8000"
        echo "📄 API docs at   http://0.0.0.0:8000/docs"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# ------------------------------------------------------------------
# Cloudflare tunnel prompt (15 s timeout → defaults to No)
# ------------------------------------------------------------------
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Cloudflare Tunnel lets you access AgriBot"
echo "   from anywhere on the internet."
echo ""

TUNNEL_CHOICE=""
read -t 15 -p "   Start Cloudflare tunnel? (y/N, auto-skip in 15s): " TUNNEL_CHOICE || true
echo ""

if [[ "$TUNNEL_CHOICE" == "y" || "$TUNNEL_CHOICE" == "Y" ]]; then
    if command -v cloudflared &> /dev/null; then
        echo "🔗 Starting Cloudflare tunnel…"
        cloudflared tunnel --url http://localhost:8000 &
        TUNNEL_PID=$!
        sleep 3
        echo "✅ Tunnel is running (PID $TUNNEL_PID)"
    else
        echo "⚠️  cloudflared not found. Install it with:"
        echo "   curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o /usr/local/bin/cloudflared"
        echo "   chmod +x /usr/local/bin/cloudflared"
    fi
else
    echo "⏭️  Skipping Cloudflare tunnel (local network only)"
fi

# ------------------------------------------------------------------
# Keep running
# ------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🟢 AgriBot is running. Press Ctrl+C to stop."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wait $SERVER_PID
