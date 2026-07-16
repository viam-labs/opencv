package armplanner

import (
	"context"
	"errors"
	"fmt"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/components/arm"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/motionplan/armplanning"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
)

const (
	KindPlanning  = "planning"
	KindExecution = "execution"
)

type Result struct {
	OK    bool `json:"ok,omitempty"`
	Error *Err `json:"error,omitempty"`
}

type Err struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

type Goal struct {
	JointsDegrees []float64 `json:"joints_degrees,omitempty"`
	Pose          *PoseGoal `json:"pose,omitempty"`
}

type PoseGoal struct {
	X              float64 `json:"x"`
	Y              float64 `json:"y"`
	Z              float64 `json:"z"`
	OX             float64 `json:"o_x"`
	OY             float64 `json:"o_y"`
	OZ             float64 `json:"o_z"`
	Theta          float64 `json:"theta"`
	ReferenceFrame string  `json:"reference_frame"`
}

type PlanMotionFunc func(context.Context, logging.Logger, *armplanning.PlanRequest) (motionplan.Plan, *armplanning.PlanMeta, error)

type Planner struct {
	logger     logging.Logger
	planMotion PlanMotionFunc
}

func NewPlanner(logger logging.Logger) *Planner {
	return &Planner{logger: logger, planMotion: armplanning.PlanMotion}
}

func (p *Planner) Run(
	ctx context.Context,
	a arm.Arm,
	fs *referenceframe.FrameSystem,
	currentInputs referenceframe.FrameSystemInputs,
	goal Goal,
) Result {
	armName := a.Name().Name

	req, err := buildPlanRequest(fs, currentInputs, armName, goal)
	if err != nil {
		return errResult(KindPlanning, err)
	}

	plan, _, err := p.planMotion(ctx, p.logger, req)
	if err != nil {
		return errResult(KindPlanning, err)
	}

	traj, err := extractTrajectory(plan, fs, armName)
	if err != nil {
		return errResult(KindPlanning, err)
	}

	if err := a.MoveThroughJointPositions(ctx, traj, nil, nil); err != nil {
		return errResult(KindExecution, err)
	}
	return Result{OK: true}
}

func errResult(kind string, err error) Result {
	return Result{Error: &Err{Kind: kind, Message: err.Error()}}
}

func buildPlanRequest(
	fs *referenceframe.FrameSystem,
	currentInputs referenceframe.FrameSystemInputs,
	armName string,
	goal Goal,
) (*armplanning.PlanRequest, error) {
	if goal.Pose != nil && goal.JointsDegrees != nil {
		return nil, errors.New("goal must specify exactly one of pose or joints_degrees")
	}
	if goal.Pose == nil && goal.JointsDegrees == nil {
		return nil, errors.New("goal must specify one of pose or joints_degrees")
	}

	var goalState *armplanning.PlanState
	switch {
	case goal.Pose != nil:
		pose := spatialmath.NewPose(
			r3.Vector{X: goal.Pose.X, Y: goal.Pose.Y, Z: goal.Pose.Z},
			&spatialmath.OrientationVectorDegrees{
				OX: goal.Pose.OX, OY: goal.Pose.OY, OZ: goal.Pose.OZ, Theta: goal.Pose.Theta,
			},
		)
		ref := goal.Pose.ReferenceFrame
		if ref == "" {
			ref = referenceframe.World
		}
		goalState = armplanning.NewPlanState(referenceframe.FrameSystemPoses{
			armName: referenceframe.NewPoseInFrame(ref, pose),
		}, nil)
	default:
		inputs, err := degreesToInputs(fs, armName, goal.JointsDegrees)
		if err != nil {
			return nil, err
		}
		goalState = armplanning.NewPlanState(nil, referenceframe.FrameSystemInputs{armName: inputs})
	}

	return &armplanning.PlanRequest{
		FrameSystem: fs,
		StartState:  armplanning.NewPlanState(nil, currentInputs),
		Goals:       []*armplanning.PlanState{goalState},
	}, nil
}

func degreesToInputs(fs *referenceframe.FrameSystem, armName string, degrees []float64) ([]referenceframe.Input, error) {
	frame := fs.Frame(armName)
	if frame == nil {
		return nil, fmt.Errorf("frame %q not in frame system", armName)
	}
	dof := frame.DoF()
	if len(degrees) != len(dof) {
		return nil, fmt.Errorf("goal joints length %d does not match arm DoF %d", len(degrees), len(dof))
	}
	inputs := make([]referenceframe.Input, len(degrees))
	for i, d := range degrees {
		inputs[i] = d * 3.141592653589793 / 180.0
	}
	return inputs, nil
}

func extractTrajectory(plan motionplan.Plan, fs *referenceframe.FrameSystem, armName string) ([][]referenceframe.Input, error) {
	steps := plan.Trajectory()
	if len(steps) == 0 {
		return nil, errors.New("planner returned empty trajectory")
	}
	traj := make([][]referenceframe.Input, 0, len(steps))
	for i, step := range steps {
		inputs, err := step.GetFrameInputs(fs.Frame(armName))
		if err != nil {
			return nil, fmt.Errorf("trajectory step %d: %w", i, err)
		}
		traj = append(traj, inputs)
	}
	return traj, nil
}
