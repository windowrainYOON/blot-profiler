#!/bin/bash
# Build "Blot Profiler.app" and a .dmg installer on macOS.
#   ./build_mac.sh            -> dist/BlotProfiler-<version>.dmg
#   ./build_mac.sh --install  -> also copies the app into /Applications
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=.venv-build
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt pyinstaller

VERSION=$(python -c "import blot_profiler; print(blot_profiler.__version__)")
APP_NAME="Blot Profiler"

# Icon
mkdir -p build
python packaging/make_icon.py build/icon_1024.png
ICONSET=build/BlotProfiler.iconset
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s build/icon_1024.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  d=$((s * 2))
  sips -z $d $d build/icon_1024.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o build/BlotProfiler.icns

# App bundle
pyinstaller --noconfirm --clean --distpath dist --workpath build/pyinstaller packaging/blot_profiler.spec

# Ad-hoc sign so macOS lets the app start on this machine
codesign --force --deep --sign - "dist/$APP_NAME.app" || true

# DMG with an Applications shortcut for drag-and-drop install
STAGE=build/dmg
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -R "dist/$APP_NAME.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
DMG="dist/BlotProfiler-$VERSION.dmg"
rm -f "$DMG"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
echo "Built $DMG"

if [ "${1:-}" = "--install" ]; then
  rm -rf "/Applications/$APP_NAME.app"
  ditto "dist/$APP_NAME.app" "/Applications/$APP_NAME.app"
  xattr -dr com.apple.quarantine "/Applications/$APP_NAME.app" 2>/dev/null || true
  echo "Installed to /Applications/$APP_NAME.app"
fi
