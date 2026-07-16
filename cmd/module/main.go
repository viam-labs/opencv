package main

import (
	"context"
	"os"
	"path/filepath"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/resource"

	"go.viam.com/rdk/components/posetracker"
	"go.viam.com/rdk/services/generic"

	"github.com/viam-labs/opencv/ipc"
	"github.com/viam-labs/opencv/models"
)

func main() {
	logger := logging.NewLogger("opencv")

	if len(os.Args) < 2 {
		logger.Fatal("expected viam-server socket path as first argument")
	}
	socketPath := os.Args[1]

	name, args, err := resolvePython()
	if err != nil {
		logger.Fatalf("locate python auxiliary: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	client, err := ipc.Spawn(ctx, logger, socketPath, name, args...)
	if err != nil {
		logger.Fatalf("spawn python auxiliary: %v", err)
	}
	defer client.Close()

	models.Register(client)

	module.ModularMain(
		resource.APIModel{API: posetracker.API, Model: models.ChessboardModel},
		resource.APIModel{API: posetracker.API, Model: models.CharucoModel},
		resource.APIModel{API: generic.API, Model: models.HandEyeCalibrationModel},
		resource.APIModel{API: generic.API, Model: models.CameraCalibrationModel},
	)
}

func resolvePython() (string, []string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", nil, err
	}
	moduleDir := filepath.Dir(filepath.Dir(exe))
	frozen := filepath.Join(moduleDir, "dist", "main")
	if _, err := os.Stat(frozen); err == nil {
		return frozen, nil, nil
	}
	return "python3", []string{filepath.Join(moduleDir, "src", "module_server.py")}, nil
}
