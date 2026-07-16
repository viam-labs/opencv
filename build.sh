#!/bin/sh
set -e
cd `dirname $0`

VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

$PYTHON -m pip install pyinstaller -Uqq

echo "Building Go utils (orientation math helper)..."
(cd go_utils && go build -o go_utils main.go)

echo "Building arm-planner Go binary..."
mkdir -p bin
go build -o bin/arm-planner ./cmd/arm-planner

mkdir -p dist

echo "Packaging Python module with bundled Go binaries..."
$PYTHON -m PyInstaller --onefile \
    --hidden-import="googleapiclient" \
    --add-binary="./go_utils/go_utils:." \
    --add-binary="./bin/arm-planner:." \
    src/main.py

tar -czvf dist/archive.tar.gz meta.json ./dist/main
