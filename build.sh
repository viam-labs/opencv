#!/bin/sh
cd `dirname $0`

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

if ! $PYTHON -m pip install pyinstaller -Uqq; then
    exit 1
fi

if ! $PYTHON -m pip install -r requirements.txt -Uqq; then
    exit 1
fi

# Build the Go binary
echo "Building Go utils..."
cd go_utils
go build -o go_utils main.go
if [ $? -ne 0 ]; then
    echo "Failed to build Go binary"
    exit 1
fi
cd ..

# Copy the Go binary to a location where PyInstaller can find it
mkdir -p dist
cp go_utils/go_utils .

$PYTHON -m PyInstaller --onefile --hidden-import="googleapiclient" --add-binary="./go_utils:." src/main.py
tar -czvf dist/archive.tar.gz meta.json ./dist/main
