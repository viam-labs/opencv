package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/golang/geo/r3"
	"go.viam.com/rdk/spatialmath"
)

func orientationVectorToMatrix(ox, oy, oz, theta float64) {
	// Create orientation vector from the input parameters
	orientationVector := &spatialmath.OrientationVectorDegrees{
		OX:    ox,
		OY:    oy,
		OZ:    oz,
		Theta: theta,
	}

	// Create a pose with zero translation and the given orientation
	pose := spatialmath.NewPose(
		r3.Vector{X: 0, Y: 0, Z: 0}, // Zero translation
		orientationVector,
	)

	// Get the rotation matrix
	rotMatrix := pose.Orientation().RotationMatrix()

	// Output the rotation matrix as 9 space-separated values (row-major order)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%.10f ", rotMatrix.At(i, j))
		}
	}
	fmt.Println() // Final newline
}

func matrixToOrientationVector(m11, m12, m13, m21, m22, m23, m31, m32, m33 float64) {
	// Create rotation matrix from the 9 input values - spatialmath.NewRotationMatrix expects a slice of 9 floats
	matrixData := []float64{m11, m12, m13, m21, m22, m23, m31, m32, m33}

	rotMatrix, err := spatialmath.NewRotationMatrix(matrixData)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating rotation matrix: %v\n", err)
		os.Exit(1)
	}

	// Convert rotation matrix to orientation vector
	orientationVector := rotMatrix.OrientationVectorDegrees()

	// Output the orientation vector components
	fmt.Printf("%.10f %.10f %.10f %.10f\n",
		orientationVector.OX,
		orientationVector.OY,
		orientationVector.OZ,
		orientationVector.Theta)
}

func composePoses(x1, y1, z1, ox1, oy1, oz1, th1, x2, y2, z2, ox2, oy2, oz2, th2 float64) {
	// Create first pose
	ov1 := &spatialmath.OrientationVectorDegrees{
		OX:    ox1,
		OY:    oy1,
		OZ:    oz1,
		Theta: th1,
	}
	pose1 := spatialmath.NewPose(
		r3.Vector{X: x1, Y: y1, Z: z1},
		ov1,
	)

	// Create second pose
	ov2 := &spatialmath.OrientationVectorDegrees{
		OX:    ox2,
		OY:    oy2,
		OZ:    oz2,
		Theta: th2,
	}
	pose2 := spatialmath.NewPose(
		r3.Vector{X: x2, Y: y2, Z: z2},
		ov2,
	)

	// Compose the poses
	result := spatialmath.Compose(pose1, pose2)

	// Extract translation
	translation := result.Point()

	// Extract orientation as orientation vector
	orientation := result.Orientation().OrientationVectorDegrees()

	// Output: x y z ox oy oz theta
	fmt.Printf("%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
		translation.X,
		translation.Y,
		translation.Z,
		orientation.OX,
		orientation.OY,
		orientation.OZ,
		orientation.Theta)
}

func inversePose(x, y, z, ox, oy, oz, theta float64) {
	// Create pose
	ov := &spatialmath.OrientationVectorDegrees{
		OX:    ox,
		OY:    oy,
		OZ:    oz,
		Theta: theta,
	}
	pose := spatialmath.NewPose(
		r3.Vector{X: x, Y: y, Z: z},
		ov,
	)

	// Invert the pose
	invPose := spatialmath.PoseInverse(pose)

	// Extract translation
	translation := invPose.Point()

	// Extract orientation as orientation vector
	orientation := invPose.Orientation().OrientationVectorDegrees()

	// Output: x y z ox oy oz theta
	fmt.Printf("%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
		translation.X,
		translation.Y,
		translation.Z,
		orientation.OX,
		orientation.OY,
		orientation.OZ,
		orientation.Theta)
}

func main() {

	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  Convert orientation vector to rotation matrix:\n")
		fmt.Fprintf(os.Stderr, "    %s ov2mat <ox> <oy> <oz> <theta>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  Convert rotation matrix to orientation vector:\n")
		fmt.Fprintf(os.Stderr, "    %s mat2ov <m11> <m12> <m13> <m21> <m22> <m23> <m31> <m32> <m33>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  Compose two poses:\n")
		fmt.Fprintf(os.Stderr, "    %s compose <x1> <y1> <z1> <ox1> <oy1> <oz1> <th1> <x2> <y2> <z2> <ox2> <oy2> <oz2> <th2>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  Invert a pose:\n")
		fmt.Fprintf(os.Stderr, "    %s inverse_pose <x> <y> <z> <ox> <oy> <oz> <theta>\n", os.Args[0])
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "ov2mat":
		if len(os.Args) != 6 {
			fmt.Fprintf(os.Stderr, "Usage: %s ov2mat <ox> <oy> <oz> <theta>\n", os.Args[0])
			os.Exit(1)
		}

		// Parse orientation vector components
		ox, err := strconv.ParseFloat(os.Args[2], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing ox: %v\n", err)
			os.Exit(1)
		}

		oy, err := strconv.ParseFloat(os.Args[3], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing oy: %v\n", err)
			os.Exit(1)
		}

		oz, err := strconv.ParseFloat(os.Args[4], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing oz: %v\n", err)
			os.Exit(1)
		}

		theta, err := strconv.ParseFloat(os.Args[5], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing theta: %v\n", err)
			os.Exit(1)
		}

		orientationVectorToMatrix(ox, oy, oz, theta)

	case "mat2ov":
		if len(os.Args) != 11 {
			fmt.Fprintf(os.Stderr, "Usage: %s mat2ov <m11> <m12> <m13> <m21> <m22> <m23> <m31> <m32> <m33>\n", os.Args[0])
			os.Exit(1)
		}

		// Parse rotation matrix elements
		var matrixElements [9]float64
		for i := 0; i < 9; i++ {
			val, err := strconv.ParseFloat(os.Args[i+2], 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing matrix element %d: %v\n", i+1, err)
				os.Exit(1)
			}
			matrixElements[i] = val
		}

		matrixToOrientationVector(
			matrixElements[0], matrixElements[1], matrixElements[2], // Row 1
			matrixElements[3], matrixElements[4], matrixElements[5], // Row 2
			matrixElements[6], matrixElements[7], matrixElements[8], // Row 3
		)

	case "compose":
		if len(os.Args) != 16 {
			fmt.Fprintf(os.Stderr, "Usage: %s compose <x1> <y1> <z1> <ox1> <oy1> <oz1> <th1> <x2> <y2> <z2> <ox2> <oy2> <oz2> <th2>\n", os.Args[0])
			os.Exit(1)
		}

		// Parse pose parameters
		var poseParams [14]float64
		for i := 0; i < 14; i++ {
			val, err := strconv.ParseFloat(os.Args[i+2], 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error parsing parameter %d: %v\n", i+1, err)
				os.Exit(1)
			}
			poseParams[i] = val
		}

		composePoses(
			poseParams[0], poseParams[1], poseParams[2], poseParams[3], poseParams[4], poseParams[5], poseParams[6],
			poseParams[7], poseParams[8], poseParams[9], poseParams[10], poseParams[11], poseParams[12], poseParams[13],
		)

	case "inverse_pose":
		if len(os.Args) != 9 {
			fmt.Fprintf(os.Stderr, "Usage: %s inverse_pose <x> <y> <z> <ox> <oy> <oz> <theta>\n", os.Args[0])
			os.Exit(1)
		}

		// Parse pose parameters
		x, err := strconv.ParseFloat(os.Args[2], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing x: %v\n", err)
			os.Exit(1)
		}

		y, err := strconv.ParseFloat(os.Args[3], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing y: %v\n", err)
			os.Exit(1)
		}

		z, err := strconv.ParseFloat(os.Args[4], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing z: %v\n", err)
			os.Exit(1)
		}

		ox, err := strconv.ParseFloat(os.Args[5], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing ox: %v\n", err)
			os.Exit(1)
		}

		oy, err := strconv.ParseFloat(os.Args[6], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing oy: %v\n", err)
			os.Exit(1)
		}

		oz, err := strconv.ParseFloat(os.Args[7], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing oz: %v\n", err)
			os.Exit(1)
		}

		theta, err := strconv.ParseFloat(os.Args[8], 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing theta: %v\n", err)
			os.Exit(1)
		}

		inversePose(x, y, z, ox, oy, oz, theta)

	default:
		// For backward compatibility, if no command is specified but we have 4 args,
		// assume it's the old ov2mat format
		if len(os.Args) == 5 {
			ox, err1 := strconv.ParseFloat(os.Args[1], 64)
			oy, err2 := strconv.ParseFloat(os.Args[2], 64)
			oz, err3 := strconv.ParseFloat(os.Args[3], 64)
			theta, err4 := strconv.ParseFloat(os.Args[4], 64)

			if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
				fmt.Fprintf(os.Stderr, "Invalid command or arguments. Use 'ov2mat' or 'mat2ov'\n")
				os.Exit(1)
			}

			orientationVectorToMatrix(ox, oy, oz, theta)
		} else {
			fmt.Fprintf(os.Stderr, "Invalid command: %s\n", command)
			fmt.Fprintf(os.Stderr, "Use 'ov2mat' or 'mat2ov'\n")
			os.Exit(1)
		}
	}
}
