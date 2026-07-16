package armplanner

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"go.viam.com/rdk/components/arm"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/motionplan/armplanning"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/spatialmath"
	"go.viam.com/rdk/testutils/inject"
	"go.viam.com/test"
)

const testArmName = "arm1"

func newTestPlanner(t *testing.T, planFn PlanMotionFunc) *Planner {
	return &Planner{logger: logging.NewTestLogger(t), planMotion: planFn}
}

func newTestArm(t *testing.T, moveErr error) (*inject.Arm, *int) {
	t.Helper()
	a := inject.NewArm(testArmName)
	calls := 0
	a.MoveThroughJointPositionsFunc = func(context.Context, [][]referenceframe.Input, *arm.MoveOptions, map[string]interface{}) error {
		calls++
		return moveErr
	}
	return a, &calls
}

// stubPlan returns a motionplan.Plan with a single trivial trajectory step
// that includes the test arm frame so GetFrameInputs succeeds.
type stubPlan struct{}

func (stubPlan) Path() motionplan.Path { return nil }
func (stubPlan) Trajectory() motionplan.Trajectory {
	return motionplan.Trajectory{
		referenceframe.FrameSystemInputs{testArmName: []referenceframe.Input{}},
	}
}

func TestRunReturnsPlanningErrorWhenPlannerFails(t *testing.T) {
	planFn := func(context.Context, logging.Logger, *armplanning.PlanRequest) (motionplan.Plan, *armplanning.PlanMeta, error) {
		return nil, nil, errors.New("no ik solution")
	}
	a, calls := newTestArm(t, nil)

	res := newTestPlanner(t, planFn).Run(context.Background(), a, nil, nil, Goal{Pose: &PoseGoal{}})

	test.That(t, res.OK, test.ShouldBeFalse)
	test.That(t, res.Error, test.ShouldNotBeNil)
	test.That(t, res.Error.Kind, test.ShouldEqual, KindPlanning)
	test.That(t, res.Error.Message, test.ShouldContainSubstring, "no ik solution")
	test.That(t, *calls, test.ShouldEqual, 0)
}

func TestRunReturnsExecutionErrorWhenArmFails(t *testing.T) {
	planFn := func(context.Context, logging.Logger, *armplanning.PlanRequest) (motionplan.Plan, *armplanning.PlanMeta, error) {
		return stubPlan{}, &armplanning.PlanMeta{}, nil
	}
	a, calls := newTestArm(t, errors.New("pstop: arm is stopped"))

	res := newTestPlanner(t, planFn).Run(context.Background(), a, singleFrameSystem(t), nil, Goal{Pose: &PoseGoal{}})

	test.That(t, res.OK, test.ShouldBeFalse)
	test.That(t, res.Error.Kind, test.ShouldEqual, KindExecution)
	test.That(t, res.Error.Message, test.ShouldContainSubstring, "pstop")
	test.That(t, *calls, test.ShouldEqual, 1)
}

func TestRunReturnsOKOnSuccess(t *testing.T) {
	planFn := func(context.Context, logging.Logger, *armplanning.PlanRequest) (motionplan.Plan, *armplanning.PlanMeta, error) {
		return stubPlan{}, &armplanning.PlanMeta{}, nil
	}
	a, calls := newTestArm(t, nil)

	res := newTestPlanner(t, planFn).Run(context.Background(), a, singleFrameSystem(t), nil, Goal{Pose: &PoseGoal{}})

	test.That(t, res.OK, test.ShouldBeTrue)
	test.That(t, res.Error, test.ShouldBeNil)
	test.That(t, *calls, test.ShouldEqual, 1)
}

func TestBuildPlanRequestRejectsMissingGoal(t *testing.T) {
	_, err := buildPlanRequest(nil, nil, testArmName, Goal{})
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "one of pose or joints_degrees")
}

func TestBuildPlanRequestRejectsBothGoals(t *testing.T) {
	_, err := buildPlanRequest(nil, nil, testArmName, Goal{
		Pose:          &PoseGoal{},
		JointsDegrees: []float64{0, 0},
	})
	test.That(t, err, test.ShouldNotBeNil)
	test.That(t, err.Error(), test.ShouldContainSubstring, "exactly one")
}

func TestResultJSONShape(t *testing.T) {
	ok, err := json.Marshal(Result{OK: true})
	test.That(t, err, test.ShouldBeNil)
	test.That(t, string(ok), test.ShouldEqual, `{"ok":true}`)

	bad, err := json.Marshal(Result{Error: &Err{Kind: KindPlanning, Message: "unreachable"}})
	test.That(t, err, test.ShouldBeNil)
	test.That(t, string(bad), test.ShouldEqual, `{"error":{"kind":"planning","message":"unreachable"}}`)
}

// singleFrameSystem returns a framesystem with just the arm frame — enough to
// let extractTrajectory succeed against stubPlan.
func singleFrameSystem(t *testing.T) *referenceframe.FrameSystem {
	t.Helper()
	fs := referenceframe.NewEmptyFrameSystem("")
	frame, err := referenceframe.NewStaticFrame(testArmName, spatialmath.NewZeroPose())
	test.That(t, err, test.ShouldBeNil)
	test.That(t, fs.AddFrame(frame, fs.World()), test.ShouldBeNil)
	return fs
}
