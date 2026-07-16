package models

import (
	"context"

	"go.viam.com/rdk/components/posetracker"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/generic"

	"github.com/viam-labs/opencv/forwarding"
	"github.com/viam-labs/opencv/ipc"
)

var (
	ChessboardModel        = resource.NewModel("viam", "opencv", "chessboard")
	CharucoModel           = resource.NewModel("viam", "opencv", "charuco")
	HandEyeCalibrationModel = resource.NewModel("viam", "opencv", "hand_eye_calibration")
	CameraCalibrationModel = resource.NewModel("viam", "opencv", "camera_calibration")
)

// Register wires every opencv model to its forwarding implementation. The
// shared ipc.Client is captured in each constructor closure.
func Register(client *ipc.Client) {
	resource.RegisterComponent(posetracker.API, ChessboardModel,
		resource.Registration[posetracker.PoseTracker, *forwarding.Config]{
			Constructor: poseTrackerConstructor(client, "chessboard"),
		})

	resource.RegisterComponent(posetracker.API, CharucoModel,
		resource.Registration[posetracker.PoseTracker, *forwarding.Config]{
			Constructor: poseTrackerConstructor(client, "charuco"),
		})

	resource.RegisterService(generic.API, HandEyeCalibrationModel,
		resource.Registration[resource.Resource, *forwarding.Config]{
			Constructor: genericConstructor(client, "hand_eye_calibration"),
		})

	resource.RegisterService(generic.API, CameraCalibrationModel,
		resource.Registration[resource.Resource, *forwarding.Config]{
			Constructor: genericConstructor(client, "camera_calibration"),
		})
}

func poseTrackerConstructor(
	client *ipc.Client,
	model string,
) func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (posetracker.PoseTracker, error) {
	return func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (posetracker.PoseTracker, error) {
		return forwarding.NewPoseTracker(ctx, client, logger, conf.ResourceName(), model, conf)
	}
}

func genericConstructor(
	client *ipc.Client,
	model string,
) func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (resource.Resource, error) {
	return func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (resource.Resource, error) {
		return forwarding.NewGeneric(ctx, client, logger, conf.ResourceName(), model, conf)
	}
}
