#!/bin/sh
set -e
cd `dirname $0`

VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

$PYTHON -m pip install pyinstaller -Uqq

echo "Building Go utils..."
(cd go_utils && go build -o go_utils main.go)

mkdir -p dist bin
cp go_utils/go_utils .

echo "Building Python auxiliary..."
$PYTHON -m PyInstaller --onefile --name main --hidden-import="googleapiclient" --add-binary="./go_utils:." src/module_server.py

echo "Building Go module entrypoint..."
go build -o bin/opencv-module ./cmd/module

tar -czvf dist/archive.tar.gz meta.json bin/opencv-module dist/main
