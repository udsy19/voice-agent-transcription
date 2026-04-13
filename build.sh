#!/bin/bash
# Muse — macOS build script
# Creates a self-contained Muse.app with bundled Python environment
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
ELECTRON_DIR="$DIR/electron"
DIST_DIR="$ELECTRON_DIR/dist"
APP_NAME="Muse"
VENV_DIR="$DIR/.build-venv"
ARCH=$(uname -m)  # arm64 or x86_64

echo "================================================"
echo "  Building $APP_NAME for macOS ($ARCH)"
echo "================================================"
echo ""

# ── Prerequisites ────────────────────────────────────────────────────────────

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
command -v node &>/dev/null || { echo "ERROR: node not found"; exit 1; }
command -v npm &>/dev/null || { echo "ERROR: npm not found"; exit 1; }

PY_VERSION=$(python3 --version 2>&1)
echo "  Python:   $PY_VERSION"
echo "  Node:     $(node --version)"
echo "  Arch:     $ARCH"
echo ""

# ── Step 1: Create Python virtual environment ───────────────────────────────

echo "[1/5] Creating Python environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Removing old build venv..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "  Installing dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install -r "$DIR/requirements.txt" -q 2>&1 | tail -5

# Verify critical imports
python3 -c "import fastapi, uvicorn, pydantic, pynput, sounddevice, numpy; print('  Core deps OK')"
python3 -c "import mlx_lm; print('  mlx-lm OK')" 2>/dev/null || echo "  mlx-lm: skipped (install manually if needed)"
python3 -c "import groq; print('  Groq SDK OK')" 2>/dev/null || echo "  groq: skipped"

VENV_SIZE=$(du -sh "$VENV_DIR" | cut -f1)
echo "  Venv size: $VENV_SIZE"
echo ""

# ── Step 2: Install Electron dependencies ────────────────────────────────────

echo "[2/5] Installing Electron dependencies..."
cd "$ELECTRON_DIR"
npm install --no-audit --no-fund -q 2>&1 | tail -3
echo ""

# ── Step 3: Build Electron app ───────────────────────────────────────────────

echo "[3/5] Building Electron app..."

# Clean previous build
rm -rf "$DIST_DIR"

# Build (creates dist/mac-arm64/Muse.app or dist/mac/Muse.app)
npx electron-builder --mac --dir 2>&1 | grep -E '(building|Built|packaging|electron-builder|ERROR)' || true
echo ""

# Find the built .app
APP_PATH=""
for candidate in "$DIST_DIR/mac-arm64/$APP_NAME.app" "$DIST_DIR/mac/$APP_NAME.app"; do
    if [ -d "$candidate" ]; then
        APP_PATH="$candidate"
        break
    fi
done

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_NAME.app not found in $DIST_DIR"
    echo "Contents of dist/:"
    ls -la "$DIST_DIR"/ 2>/dev/null || echo "  (empty)"
    exit 1
fi

echo "  App: $APP_PATH"
echo ""

# ── Step 4: Bundle Python environment ────────────────────────────────────────

echo "[4/5] Bundling Python environment..."

RESOURCES="$APP_PATH/Contents/Resources"
BUNDLE_VENV="$RESOURCES/python-env"

# Copy the venv into the app bundle
cp -a "$VENV_DIR" "$BUNDLE_VENV"

# Make the venv relocatable by fixing shebang lines
# Replace absolute paths in bin/ scripts with #!/usr/bin/env python3
find "$BUNDLE_VENV/bin" -type f -maxdepth 1 | while read f; do
    if head -1 "$f" 2>/dev/null | grep -q "^#!.*python"; then
        # macOS sed -i requires backup extension
        sed -i '' "1s|^#!.*python.*|#!/usr/bin/env python3|" "$f" 2>/dev/null || true
    fi
done

# Fix the pyvenv.cfg to be relative (optional — Python resolves this at runtime)
if [ -f "$BUNDLE_VENV/pyvenv.cfg" ]; then
    # Clear the home path so Python uses the bundled binary
    sed -i '' 's|^home = .*|home = bin|' "$BUNDLE_VENV/pyvenv.cfg" 2>/dev/null || true
fi

# Strip __pycache__, .pyc, and test dirs to save space
# NOTE: Do NOT strip .dist-info — transformers/mlx_lm need importlib.metadata for version checks
find "$BUNDLE_VENV" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_VENV" -name '*.pyc' -delete 2>/dev/null || true
find "$BUNDLE_VENV" -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_VENV" -type d -name 'test' -exec rm -rf {} + 2>/dev/null || true

BUNDLE_SIZE=$(du -sh "$BUNDLE_VENV" | cut -f1)
echo "  Bundled venv: $BUNDLE_SIZE"
echo ""

# ── Step 5: Code sign ────────────────────────────────────────────────────────

echo "[5/5] Code signing..."

ENTITLEMENTS="$ELECTRON_DIR/entitlements.mac.plist"

if [ -f "$ENTITLEMENTS" ]; then
    # Ad-hoc sign (no Apple Developer ID needed — works for local distribution)
    codesign --force --deep --sign - --entitlements "$ENTITLEMENTS" "$APP_PATH" 2>&1 || {
        echo "  WARNING: codesign failed — app will still work but Gatekeeper may block it"
    }
    echo "  Signed with ad-hoc identity + entitlements"
else
    echo "  WARNING: No entitlements file found — skipping codesign"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)

echo ""
echo "================================================"
echo "  Build complete!"
echo "================================================"
echo ""
echo "  App:    $APP_PATH"
echo "  Size:   $APP_SIZE"
echo ""
echo "  To run:"
echo "    open \"$APP_PATH\""
echo ""
echo "  To distribute (removes Gatekeeper quarantine):"
echo "    xattr -cr \"$APP_PATH\""
echo ""
echo "  To create a DMG:"
echo "    hdiutil create -volname Muse -srcfolder \"$APP_PATH\" -ov -format UDZO Muse.dmg"
echo ""

# Cleanup build venv (the copy inside the .app is all we need)
deactivate 2>/dev/null || true
rm -rf "$VENV_DIR"
