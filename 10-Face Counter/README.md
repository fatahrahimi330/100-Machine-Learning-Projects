# Face Counter using OpenCV

A real-time face detection and counting application using OpenCV's Haar Cascade classifier. This project captures video from your webcam and detects/counts the number of faces in the frame.

## Features

- **Real-time Face Detection**: Detects faces in live video stream from your webcam
- **Face Counting**: Counts and labels each detected face
- **Visual Feedback**: Draws green rectangles around detected faces with labels
- **Simple Controls**: Easy to start and stop with keyboard commands

## Demo

The application displays a live video feed with:
- Green rectangles around detected faces
- "face num" labels for each detected face
- Real-time face counting

## Requirements

- Python 3.x
- Webcam/Camera
- macOS (camera permissions required)

## Dependencies

- **opencv-python**: Computer vision library for image processing and face detection
- **numpy**: Numerical computing library

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "10-Face Counter"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install required packages:**
   ```bash
   pip install opencv-python numpy
   ```

## Usage

1. **Run the face counter:**
   ```bash
   python face_counter.py
   ```
   
   Or if using the virtual environment:
   ```bash
   .venv/bin/python face_counter.py
   ```

2. **Grant camera permissions:**
   - On macOS, you'll be prompted to allow camera access
   - Click "OK" to grant permission

3. **Face detection:**
   - Position yourself in front of the camera
   - The application will detect and count faces in real-time
   - Green rectangles will appear around detected faces
   - Each face will be labeled with "face num#"

4. **Stop the application:**
   - Press the **'q'** key while the camera window is active
   - Or press **Ctrl+C** in the terminal

## How It Works

1. **Camera Initialization**: Opens the default webcam using `cv2.VideoCapture(0)`

2. **Face Detection**: Uses OpenCV's Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) for detecting faces

3. **Frame Processing**:
   - Captures frames from the webcam
   - Flips the frame horizontally for mirror effect
   - Converts to grayscale for better detection
   - Detects faces using the cascade classifier

4. **Visualization**:
   - Draws rectangles around detected faces
   - Labels each face with a number
   - Displays the processed frame in real-time

## Project Structure

```
10-Face Counter/
│
├── face_counter.py          # Main application script
├── face_counter.ipynb       # Jupyter notebook version
├── README.md               # Project documentation
└── .venv/                  # Virtual environment (created during setup)
```

## Code Overview

```python
# Key components:
- cv2.VideoCapture(0)                    # Open camera
- cv2.CascadeClassifier()                # Load face detector
- face_cascade.detectMultiScale()        # Detect faces
- cv2.rectangle()                        # Draw boxes around faces
- cv2.putText()                          # Label faces
- cv2.imshow()                           # Display video feed
```

## Troubleshooting

### Camera Access Issues
- **Error: "Could not access camera"**
  - Ensure camera permissions are granted in System Preferences > Security & Privacy > Camera
  - Check if another application is using the camera
  - Try restarting the application

### Module Not Found Errors
- **Error: "No module named 'cv2'"**
  - Ensure packages are installed: `pip install opencv-python numpy`
  - Use the correct Python interpreter (especially if using virtual environment)

### Detection Issues
- **Faces not detected:**
  - Ensure good lighting conditions
  - Face the camera directly
  - Adjust `scaleFactor` and `minNeighbors` parameters in `detectMultiScale()` for better accuracy

## Configuration

You can adjust detection sensitivity by modifying these parameters in `face_counter.py`:

```python
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,      # Adjust between 1.05-1.5 (lower = more sensitive)
    minNeighbors=5,       # Adjust between 3-6 (lower = more detections)
    minSize=(30, 30)      # Minimum face size to detect
)
```

## Technical Details

- **Detection Method**: Haar Cascade Classifier (frontal face)
- **Image Processing**: BGR to Grayscale conversion
- **Frame Rate**: Real-time (depends on system performance)
- **Output**: Live video window with annotated faces

## Future Enhancements

- [ ] Add face recognition capabilities
- [ ] Save snapshots of detected faces
- [ ] Count total unique faces over time
- [ ] Add age and gender detection
- [ ] Support for multiple camera sources
- [ ] Export face count statistics

## License

This project is open source and available for educational purposes.

## Acknowledgments

- OpenCV library for computer vision tools
- Haar Cascade classifier for face detection
- NumPy for numerical operations

## Author

Created as part of the 100+ Machine Learning Projects series.

---

**Note**: This application requires camera access and processes video in real-time. Ensure your system meets the requirements and has adequate processing power for smooth operation.
