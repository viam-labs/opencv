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
		emit(armplanner.Result{Error: &armplanner.Err{Kind: armplanner.KindPlanning, Message: fmt.Sprintf("parse goal: %v", err)}})
		return 1
	}

	logger := logging.NewLogger("arm-planner")
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	robot, err := client.New(ctx, *parentAddr, logger,
		client.WithDialOptions(rpc.WithInsecure()),
	)
	if err != nil {
		emit(armplanner.Result{Error: &armplanner.Err{Kind: armplanner.KindExecution, Message: fmt.Sprintf("connect to viam-server: %v", err)}})
		return 1
	}
	defer func() {
		if err := robot.Close(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "close viam-server client: %v\n", err)
		}
	}()

	a, err := arm.FromProvider(robot, *armName)
	if err != nil {
		emit(armplanner.Result{Error: &armplanner.Err{Kind: armplanner.KindExecution, Message: fmt.Sprintf("resolve arm %q: %v", *armName, err)}})
		return 1
	}

	fsCfg, err := robot.FrameSystemConfig(ctx)
	if err != nil {
		emit(armplanner.Result{Error: &armplanner.Err{Kind: armplanner.KindExecution, Message: fmt.Sprintf("get frame system config: %v", err)}})
		return 1
	}
	fs, err := referenceframe.NewFrameSystem("", fsCfg.Parts, nil)
	if err != nil {
		emit(armplanner.Result{Error: &armplanner.Err{Kind: armplanner.KindExecution, Message: fmt.Sprintf("build frame system: %v", err)}})
		return 1
	}

	currentInputs, err := robot.CurrentInputs(ctx)
	if err != nil {
		emit(armplanner.Result{Error: &armplanner.Err{Kind: armplanner.KindExecution, Message: fmt.Sprintf("get current inputs: %v", err)}})
		return 1
	}

	result := armplanner.NewPlanner(logger).Run(ctx, a, fs, currentInputs, goal)
	emit(result)
	if result.OK {
		return 0
	}
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
