package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"go.viam.com/rdk/components/arm"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/robot/client"
	"go.viam.com/utils/rpc"

	"github.com/viam-labs/opencv/armplanner"
)

const defaultTimeout = 5 * time.Minute

func main() {
	os.Exit(run())
}

func run() int {
	var (
		armName    = flag.String("arm", "", "arm resource name")
		parentAddr = flag.String("parent-addr", "", "viam-server address")
		goalJSON   = flag.String("goal", "", "goal as JSON: {\"pose\":{...}} or {\"joints_degrees\":[...]}")
		timeout    = flag.Duration("timeout", defaultTimeout, "hard ceiling on the entire plan+execute cycle")
	)
	flag.Parse()

	if *armName == "" || *parentAddr == "" || *goalJSON == "" {
		fmt.Fprintln(os.Stderr, "arm, parent-addr, and goal are required")
		return 2
	}

	var goal armplanner.Goal
	if err := json.Unmarshal([]byte(*goalJSON), &goal); err != nil {
		return fail(armplanner.KindPlanning, "parse goal", err)
	}

	logger := logging.NewLogger("arm-planner")
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	robot, err := client.New(ctx, *parentAddr, logger,
		client.WithDialOptions(rpc.WithInsecure()),
	)
	if err != nil {
		return fail(armplanner.KindExecution, "connect to viam-server", err)
	}
	defer func() {
		if err := robot.Close(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "close viam-server client: %v\n", err)
		}
	}()

	a, err := arm.FromProvider(robot, *armName)
	if err != nil {
		return fail(armplanner.KindExecution, fmt.Sprintf("resolve arm %q", *armName), err)
	}

	fsCfg, err := robot.FrameSystemConfig(ctx)
	if err != nil {
		return fail(armplanner.KindExecution, "get frame system config", err)
	}
	fs, err := referenceframe.NewFrameSystem("", fsCfg.Parts, nil)
	if err != nil {
		return fail(armplanner.KindExecution, "build frame system", err)
	}

	currentInputs, err := robot.CurrentInputs(ctx)
	if err != nil {
		return fail(armplanner.KindExecution, "get current inputs", err)
	}

	result := armplanner.NewPlanner(logger).Run(ctx, a, fs, currentInputs, goal)
	emit(result)
	if result.OK {
		return 0
	}
	return 1
}

func fail(kind, context string, err error) int {
	emit(armplanner.Result{Error: &armplanner.Err{
		Kind:    kind,
		Message: fmt.Sprintf("%s: %v", context, err),
	}})
	return 1
}

func emit(r armplanner.Result) {
	data, err := json.Marshal(r)
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal result: %v\n", err)
		return
	}
	if _, err := os.Stdout.Write(append(data, '\n')); err != nil {
		fmt.Fprintf(os.Stderr, "write result: %v\n", err)
	}
}
