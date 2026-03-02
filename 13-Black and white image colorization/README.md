# Black and White Image Colorization with OpenCV (Deep Learning)

## Overview
This project colorizes a grayscale image using OpenCV's DNN module and a pre-trained Caffe model from the colorization research by Richard Zhang et al.

The notebook:
- downloads an input black-and-white image,
- downloads required model files,
- runs inference to predict color channels,
- reconstructs and saves the colorized output.

---

## Project Structure

- `Black_and_white_image_colorization.ipynb` — main notebook with complete workflow
- `image.png` — source image file in the folder

> During execution, additional runtime files are created:
- `black_white.png`
- `colorization_deploy_v2.prototxt`
- `pts_in_hull.npy`
- `colorization_release_v2.caffemodel`
- `colorized.png`

---

## How It Works
The model operates in **LAB color space**:
1. Convert image from BGR to LAB.
2. Use the **L (lightness)** channel as input to the network.
3. Predict **A/B color channels** with the pre-trained model.
4. Merge L + predicted AB.
5. Convert LAB back to BGR/RGB for visualization and saving.

---

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Internet access (for downloading model files)
- Jupyter Notebook / Google Colab / VS Code Notebook

Install dependencies:

```bash
pip install opencv-python numpy matplotlib
```

---

## Model Files Used
The notebook downloads the following files:

1. **Network architecture** (`.prototxt`)
2. **Cluster centers** (`pts_in_hull.npy`)
3. **Pre-trained Caffe weights** (`.caffemodel`)

If download links change, update URLs in the download cells before running.

---

## Run Instructions
1. Open `Black_and_white_image_colorization.ipynb`.
2. Run cells in order from top to bottom.
3. Ensure model download cells complete successfully.
4. Run the inference cell.
5. Check side-by-side visualization in the notebook.
6. Find final result at `colorized.png`.

---

## Output
- **Displayed in notebook**: Original vs Colorized image
- **Saved file**: `colorized.png`

---

## Troubleshooting

### 1) `EOFError: No data left in file` for `pts_in_hull.npy`
Cause: file is empty or corrupted.

Fix:
- Re-run the points download cell.
- Verify file is not zero bytes.
- Re-run the colorization cell.

### 2) `FileNotFoundError` for model files
Cause: one or more downloads failed.

Fix:
- Re-run all model download cells.
- Check internet connection.
- Confirm files exist in the same directory as the notebook.

### 3) `cv2.imshow` not working in notebooks
Use Matplotlib display (already implemented in the notebook).

---

## Notes
- Results depend on image content and model priors.
- Some colors may be plausible but not historically exact.
- This is automatic colorization, not manual artist-guided editing.

---

## Credits
- **Colorization model**: Richard Zhang, Phillip Isola, Alexei A. Efros
- Paper: *Colorful Image Colorization* (ECCV 2016)
- OpenCV DNN for inference pipeline

---

## License
This repository uses external model files from third-party sources. Follow the original model/paper licensing terms when redistributing model assets.