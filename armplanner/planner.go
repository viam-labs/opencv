package armplanner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	commonpb "go.viam.com/api/common/v1"
	"go.viam.com/rdk/components/arm"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/motionplan/armplanning"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/utils"
	"google.golang.org/protobuf/encoding/protojson"
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
	JointsDegrees []float64       `json:"joints_degrees,omitempty"`
	Pose          json.RawMessage `json:"pose,omitempty"`
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
	hasPose := len(goal.Pose) > 0
	hasJoints := goal.JointsDegrees != nil
	if hasPose && hasJoints {
		return nil, errors.New("goal must specify exactly one of pose or joints_degrees")
	}
	if !hasPose && !hasJoints {
		return nil, errors.New("goal must specify one of pose or joints_degrees")
	}

	var goalState *armplanning.PlanState
	if hasPose {
		var pifProto commonpb.PoseInFrame
		if err := protojson.Unmarshal(goal.Pose, &pifProto); err != nil {
			return nil, fmt.Errorf("parse pose: %w", err)
		}
		goalState = armplanning.NewPlanState(referenceframe.FrameSystemPoses{
			armName: referenceframe.ProtobufToPoseInFrame(&pifProto),
		}, nil)
	} else {
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
		inputs[i] = utils.DegToRad(d)
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
