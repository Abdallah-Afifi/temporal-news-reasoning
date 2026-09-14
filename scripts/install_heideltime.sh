#!/bin/bash
# ============================================================
# Install TreeTagger + HeidelTime (py_heideltime)
# ============================================================
# Usage:
#   bash scripts/install_heideltime.sh          # installs to ~/treetagger
#   bash scripts/install_heideltime.sh /opt/tt  # installs to custom path
# ============================================================

set -e

TREETAGGER_DIR="${1:-$HOME/treetagger}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

echo "============================================================"
echo " HeidelTime + TreeTagger installer"
echo " TreeTagger dir: $TREETAGGER_DIR"
echo "============================================================"

# ── 1. Java ─────────────────────────────────────────────────
echo ""
echo "[1/5] Checking Java..."
if ! command -v java &> /dev/null; then
    echo "  Java not found. Installing..."
    sudo apt-get install -y default-jre default-jdk
else
    echo "  Java found: $(java -version 2>&1 | head -1)"
fi

# ── 2. TreeTagger binary ────────────────────────────────────
echo ""
echo "[2/5] Installing TreeTagger to $TREETAGGER_DIR..."
mkdir -p "$TREETAGGER_DIR"
cd "$TREETAGGER_DIR"

TT_URL="https://www.cis.lmu.de/~schmid/tools/TreeTagger/data"
TT_LINUX="tree-tagger-linux-64.tar.gz"
TT_SCRIPTS="tagger-scripts.tar.gz"
TT_INSTALL="install-tagger.sh"
EN_PARAMS="english-utf8.par.gz"

for f in "$TT_LINUX" "$TT_SCRIPTS" "$TT_INSTALL" "$EN_PARAMS"; do
    if [ ! -f "$f" ]; then
        echo "  Downloading $f..."
        wget -q "$TT_URL/$f" -O "$f"
    else
        echo "  Already downloaded: $f"
    fi
done

echo "  Running TreeTagger install script..."
bash install-tagger.sh

echo "  TreeTagger installed."

# ── 3. py_heideltime via pip ────────────────────────────────
echo ""
echo "[3/5] Installing py_heideltime..."
PIP="pip"
for _v in venv .venv; do
    if [ -x "$PROJECT_DIR/$_v/bin/python" ]; then
        PIP="$PROJECT_DIR/$_v/bin/python -m pip"
        echo "  using $PROJECT_DIR/$_v"
        break
    fi
done
$PIP install -q "py_heideltime>=1.0.3"
echo "  py_heideltime installed."

# ── 4. Configure py_heideltime to find TreeTagger ──────────
echo ""
echo "[4/5] Configuring py_heideltime..."
# py_heideltime looks for TreeTagger in its own package config
PYHT_CFG=$(python -c "import py_heideltime; import os; print(os.path.join(os.path.dirname(py_heideltime.__file__), 'settings.py'))" 2>/dev/null || echo "")
if [ -n "$PYHT_CFG" ] && [ -f "$PYHT_CFG" ]; then
    # Patch the settings to point to our TreeTagger installation
    sed -i "s|TreeTagger_PATH.*=.*|TreeTagger_PATH = '$TREETAGGER_DIR'|g" "$PYHT_CFG"
    echo "  Patched py_heideltime settings: $PYHT_CFG"
else
    echo "  Could not find py_heideltime settings file — setting env var instead."
fi

# ── 5. Write env vars to .env ───────────────────────────────
echo ""
echo "[5/5] Writing environment variables to $ENV_FILE..."
# Remove old entries if present
touch "$ENV_FILE"
grep -v "^TREETAGGER_PATH\|^HEIDELTIME_PATH" "$ENV_FILE" > "$ENV_FILE.tmp" && mv "$ENV_FILE.tmp" "$ENV_FILE"
echo "TREETAGGER_PATH=$TREETAGGER_DIR" >> "$ENV_FILE"
echo "HEIDELTIME_PATH=$TREETAGGER_DIR" >> "$ENV_FILE"
echo "  Written to .env"

echo ""
echo "============================================================"
echo " Done! Verifying installation..."
echo "============================================================"
python - <<'PYEOF'
import sys
try:
    from py_heideltime import heideltime
    text = "He was born on 12 March 1990 and died two years later."
    result = heideltime(text, language="English")
    print(f"  [OK] py_heideltime works!")
    print(f"  Input : {text}")
    print(f"  Output: {result}")
except Exception as e:
    print(f"  [WARN] py_heideltime test failed: {e}")
    print("  This may be normal if TreeTagger binaries are not fully set up.")
    print("  Try running: export TREETAGGER_PATH=$TREETAGGER_DIR")
    sys.exit(1)
PYEOF

echo ""
echo "To activate in your shell:"
echo "  source .env  (or add to your shell profile)"
echo "  export TREETAGGER_PATH=$TREETAGGER_DIR"
