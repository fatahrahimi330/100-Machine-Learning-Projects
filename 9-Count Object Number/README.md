# Count Number of Objects using OpenCV

## Project Overview
This project demonstrates object detection and counting using OpenCV, a powerful computer vision library. The application specifically counts coins in an image using image processing techniques such as edge detection, morphological operations, and contour analysis.

## Objective
The goal of this project is to:
- Detect objects (coins) in an image automatically
- Count the number of objects present
- Visualize the detected objects with bounding contours
- Learn image processing fundamentals including filtering, edge detection, and contour detection

## Features
- Downloads sample image from GitHub
- Converts image to grayscale for processing
- Applies Gaussian blur to reduce noise
- Uses Canny edge detection to identify object boundaries
- Applies morphological dilation to connect edges
- Finds and visualizes contours
- Counts total number of detected objects
- Displays results at each processing step

## Requirements
- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib

## Installation

### 1. Clone or download the project
```bash
cd "9-Count Object Number"
```

### 2. Install required packages
```bash
pip install opencv-python numpy matplotlib
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook count_object_number.ipynb
```

## How It Works

### Image Processing Pipeline

1. **Image Download & Loading**
   - Downloads a coin image from the GitHub repository
   - Loads the image using OpenCV

2. **Grayscale Conversion**
   - Converts BGR color image to grayscale
   - Reduces data complexity for processing

3. **Gaussian Blur**
   - Applies Gaussian blur with kernel size (21, 21)
   - Reduces image noise and smooths edges

4. **Canny Edge Detection**
   - Detects edges using Canny algorithm
   - Thresholds: lower=30, upper=150
   - Identifies object boundaries

5. **Morphological Dilation**
   - Applies dilation with kernel (1, 1)
   - Connects fragmented edges to form complete contours
   - Can be adjusted for better results

6. **Contour Detection**
   - Finds all contours in the dilated image
   - Uses external contour retrieval method
   - Provides contour hierarchy information

7. **Visualization & Counting**
   - Draws green contours on the original image
   - Counts total number of detected objects
   - Displays result on screen

## Project Structure
```
9-Count Object Number/
├── count_object_number.ipynb    # Main Jupyter Notebook
├── image.jpg                     # Downloaded sample image (generated at runtime)
└── README.md                     # This file
```

## Usage

1. Open the notebook in Jupyter
2. Run cells sequentially from top to bottom
3. Each cell displays the intermediate results
4. The final output shows the count of detected objects

### Expected Output
```
coins in the image : 7
```

## Parameter Tuning

If the detection accuracy needs improvement, adjust these parameters:

### Gaussian Blur
```python
blure = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
```
- Larger kernel = more smoothing (may miss small objects)
- Smaller kernel = preserve more details (may include noise)

### Canny Edge Detection
```python
canny = cv2.Canny(blure, lower_threshold, upper_threshold, 3)
```
- Lower threshold: 30 (adjust lower to detect fainter edges)
- Upper threshold: 150 (adjust higher to be more selective)

### Dilation
```python
dilated = cv2.dilate(canny, kernel, iterations=n)
```
- Increase iterations for stronger dilation
- Use larger kernel for better edge connection

## Results
The project successfully detects and counts objects in the image by:
- Identifying clear object boundaries through edge detection
- Connecting broken edges through morphological operations
- Counting distinct contours as individual objects

## Limitations
- Works best with well-defined, distinct objects
- Performance depends on image quality and lighting
- Overlapping objects may be counted as one
- Requires parameter tuning for different image types

## Future Improvements
- Add support for different object types
- Implement watershed algorithm for overlapping object separation
- Use machine learning for more robust detection
- Add GUI interface for parameter adjustment
- Batch process multiple images

## Author
Created as part of the 100 Machine Learning Projects series

## References
- [OpenCV Documentation](https://docs.opencv.org/)
- [Computer Vision Basics](https://en.wikipedia.org/wiki/Computer_vision)
- Edge Detection: [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector)

## License
Open source for educational purposes

## Notes
- The image is automatically downloaded from GitHub at runtime
- Ensure stable internet connection for image download
- Results may vary based on image quality and parameters
