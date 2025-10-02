# Module opencv 

Provide a description of the purpose of the module and any relevant information.

## Model viam:opencv:chessboard

Provide a description of the model and any relevant information.

### Configuration
The following attribute template can be used to configure this model:

```json
{
"attribute_1": <float>,
"attribute_2": <string>
}
```

#### Attributes

The following attributes are available for this model:

| Name          | Type   | Inclusion | Description                |
|---------------|--------|-----------|----------------------------|
| `camera_name` | string  | Required  | Name of the camera used for checking pose of chessboard. |
| `pattern_size` | list | Required  | Size of the chessboard pattern. |
| `square_size_mm` | int | Required  | Physical size of a square in the chessboard pattern.  |

#### Example Configuration

```json
{
  "camera_name": "cam",
  "pattern_size": [9, 6],
  "square_size_mm": 21
}
```

### Camera Calibration

This model provides camera calibration functionality through the `do_command` interface. You can use it to determine the intrinsic parameters of a camera by providing multiple images of a chessboard pattern.

#### Calibrate Camera Command

Use the `calibrate_camera` command to compute camera intrinsics:

```python
import base64

# Capture and encode images at your own pace
images = []
for i in range(10):
    # Capture image from your camera
    # User can move the chessboard between captures with proper feedback
    img_data = capture_image()  # Your image capture logic
    
    # Encode to base64
    base64_img = base64.b64encode(img_data).decode('utf-8')
    images.append(base64_img)

# Run calibration
result = await chessboard.do_command({
    "calibrate_camera": {
        "images": images  # List of base64 encoded image strings
    }
})
```

**Parameters:**
- `images` (required): List of base64 encoded image strings containing chessboard patterns

**Returns:**
The command returns a dictionary with the following structure:

```json
{
  "success": true,
  "rms_error": 0.234,
  "num_images": 10,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "camera_matrix": {
    "fx": 1234.56,
    "fy": 1235.67,
    "cx": 960.12,
    "cy": 540.34
  },
  "distortion_coefficients": {
    "k1": -0.123,
    "k2": 0.045,
    "p1": -0.001,
    "p2": 0.002,
    "k3": -0.012
  }
}
```

**How it works:**
1. You capture images of the chessboard pattern at your own pace (the user controls when each image is captured)
2. Encode the images as base64 strings and pass them to the command
3. For each image, the system detects and refines the chessboard corners
4. Once all images are processed, it uses OpenCV's `calibrateCamera` function to compute:
   - **Camera matrix (intrinsics)**: Focal lengths (fx, fy) and principal point (cx, cy)
   - **Distortion coefficients**: Radial (k1, k2, k3) and tangential (p1, p2) distortion parameters
   - **RMS error**: Root mean square reprojection error (lower is better)
   - **Number of images**: Number of images successfully used for calibration

**Tips for best results:**
- Capture 10-20 images of the chessboard in different positions and orientations
- Cover different areas of the camera's field of view
- Ensure the chessboard is well-lit and in focus in each image
- Tilt and rotate the chessboard between captures for better calibration
- At least 3 valid images are required for calibration to succeed