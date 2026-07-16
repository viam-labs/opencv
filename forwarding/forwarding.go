package forwarding

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/components/posetracker"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/generic"
	"go.viam.com/rdk/spatialmath"

	"github.com/viam-labs/opencv/ipc"
)

type wirePose struct {
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Z     float64 `json:"z"`
	OX    float64 `json:"o_x"`
	OY    float64 `json:"o_y"`
	OZ    float64 `json:"o_z"`
	Theta float64 `json:"theta"`
}

type resourceCreateParams struct {
	Model  string          `json:"model"`
	Name   string          `json:"name"`
	Config json.RawMessage `json:"config"`
	Deps   []string        `json:"deps"`
}

type resourceCallParams struct {
	Name   string          `json:"name"`
	Method string          `json:"method"`
	Args   json.RawMessage `json:"args,omitempty"`
}

type resourceCloseParams struct {
	Name string `json:"name"`
}

// Config is shared by all forwarding resources — a passthrough of the raw
// attributes plus the caller-declared dependency names, so Python can
// resolve deps against viam-server directly.
type Config struct {
	Deps       []string        `json:"deps,omitempty"`
	Attributes json.RawMessage `json:"-"`
}

func (c *Config) Validate(path string) ([]string, []string, error) {
	return c.Deps, nil, nil
}

// PoseTracker is a forwarding pose_tracker resource whose method calls proxy
// to a Python auxiliary process over IPC.
type PoseTracker struct {
	resource.Named
	resource.TriviallyCloseable

	logger logging.Logger
	client *ipc.Client
	model  string
}

func NewPoseTracker(
	ctx context.Context,
	client *ipc.Client,
	logger logging.Logger,
	name resource.Name,
	model string,
	rawConf resource.Config,
) (posetracker.PoseTracker, error) {
	attrs, err := marshalAttributes(rawConf)
	if err != nil {
		return nil, err
	}
	deps, _, _ := rawConf.Validate("", "")

	if _, err := client.Call(ctx, "resource.new", resourceCreateParams{
		Model:  model,
		Name:   name.Name,
		Config: attrs,
		Deps:   deps,
	}); err != nil {
		return nil, fmt.Errorf("python resource.new: %w", err)
	}

	return &PoseTracker{
		Named:  name.AsNamed(),
		logger: logger,
		client: client,
		model:  model,
	}, nil
}

func (p *PoseTracker) Reconfigure(ctx context.Context, deps resource.Dependencies, conf resource.Config) error {
	attrs, err := marshalAttributes(conf)
	if err != nil {
		return err
	}
	reqDeps, _, _ := conf.Validate("", "")
	_, err = p.client.Call(ctx, "resource.reconfigure", resourceCreateParams{
		Model:  p.model,
		Name:   p.Name().Name,
		Config: attrs,
		Deps:   reqDeps,
	})
	return err
}

func (p *PoseTracker) Poses(
	ctx context.Context,
	bodyNames []string,
	extra map[string]interface{},
) (referenceframe.FrameSystemPoses, error) {
	args := map[string]interface{}{"body_names": bodyNames, "extra": extra}
	raw, err := p.callResource(ctx, "get_poses", args)
	if err != nil {
		return nil, err
	}
	var wire map[string]wirePose
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("decode poses: %w", err)
	}
	out := make(referenceframe.FrameSystemPoses, len(wire))
	for name, wp := range wire {
		out[name] = referenceframe.NewPoseInFrame(name, poseFromWire(wp))
	}
	return out, nil
}

func (p *PoseTracker) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return doCommandForward(ctx, p.client, p.Name().Name, cmd)
}

func (p *PoseTracker) Close(ctx context.Context) error {
	_, err := p.client.Call(ctx, "resource.close", resourceCloseParams{Name: p.Name().Name})
	return err
}

func (p *PoseTracker) callResource(ctx context.Context, method string, args any) (json.RawMessage, error) {
	body, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("marshal args: %w", err)
	}
	return p.client.Call(ctx, "resource.call", resourceCallParams{
		Name:   p.Name().Name,
		Method: method,
		Args:   body,
	})
}

// Generic is a forwarding generic-service resource.
type Generic struct {
	resource.Named
	resource.TriviallyCloseable

	logger logging.Logger
	client *ipc.Client
	model  string
}

func NewGeneric(
	ctx context.Context,
	client *ipc.Client,
	logger logging.Logger,
	name resource.Name,
	model string,
	rawConf resource.Config,
) (resource.Resource, error) {
	attrs, err := marshalAttributes(rawConf)
	if err != nil {
		return nil, err
	}
	deps, _, _ := rawConf.Validate("", "")
	if _, err := client.Call(ctx, "resource.new", resourceCreateParams{
		Model:  model,
		Name:   name.Name,
		Config: attrs,
		Deps:   deps,
	}); err != nil {
		return nil, fmt.Errorf("python resource.new: %w", err)
	}

	return &Generic{
		Named:  name.AsNamed(),
		logger: logger,
		client: client,
		model:  model,
	}, nil
}

func (g *Generic) Reconfigure(ctx context.Context, deps resource.Dependencies, conf resource.Config) error {
	attrs, err := marshalAttributes(conf)
	if err != nil {
		return err
	}
	reqDeps, _, _ := conf.Validate("", "")
	_, err = g.client.Call(ctx, "resource.reconfigure", resourceCreateParams{
		Model:  g.model,
		Name:   g.Name().Name,
		Config: attrs,
		Deps:   reqDeps,
	})
	return err
}

func (g *Generic) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return doCommandForward(ctx, g.client, g.Name().Name, cmd)
}

func (g *Generic) Close(ctx context.Context) error {
	_, err := g.client.Call(ctx, "resource.close", resourceCloseParams{Name: g.Name().Name})
	return err
}

// Compile-time interface conformance checks.
var _ posetracker.PoseTracker = (*PoseTracker)(nil)
var _ resource.Resource = (*Generic)(nil)
var _ generic.Service = (*Generic)(nil)

func doCommandForward(ctx context.Context, client *ipc.Client, name string, cmd map[string]interface{}) (map[string]interface{}, error) {
	body, err := json.Marshal(cmd)
	if err != nil {
		return nil, err
	}
	raw, err := client.Call(ctx, "resource.call", resourceCallParams{
		Name:   name,
		Method: "do_command",
		Args:   body,
	})
	if err != nil {
		return nil, err
	}
	if len(raw) == 0 || string(raw) == "null" {
		return map[string]interface{}{}, nil
	}
	var out map[string]interface{}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func marshalAttributes(conf resource.Config) (json.RawMessage, error) {
	if conf.Attributes == nil {
		return json.RawMessage("{}"), nil
	}
	return json.Marshal(map[string]interface{}(conf.Attributes))
}

func poseFromWire(w wirePose) spatialmath.Pose {
	return spatialmath.NewPose(
		r3.Vector{X: w.X, Y: w.Y, Z: w.Z},
		&spatialmath.OrientationVectorDegrees{OX: w.OX, OY: w.OY, OZ: w.OZ, Theta: w.Theta},
	)
}
