#!/bin/bash
# ============================================================
# Install HeidelTime temporal tagger
# ============================================================

set -e

HEIDELTIME_DIR="${1:-/opt/heideltime}"

echo "Installing HeidelTime to $HEIDELTIME_DIR..."

# Check Java
if ! command -v java &> /dev/null; then
    echo "ERROR: Java is required. Install with: sudo apt-get install default-jre default-jdk"
    exit 1
fi

# Clone HeidelTime
if [ ! -d "$HEIDELTIME_DIR" ]; then
    sudo mkdir -p "$HEIDELTIME_DIR"
    sudo chown "$USER:$USER" "$HEIDELTIME_DIR"
    git clone https://github.com/HeidelTime/heideltime.git "$HEIDELTIME_DIR"
fi

cd "$HEIDELTIME_DIR"

echo "HeidelTime installed at $HEIDELTIME_DIR"
echo "Set HEIDELTIME_PATH=$HEIDELTIME_DIR in your .env file"
