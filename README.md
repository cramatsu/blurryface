# BlurryFace - Face Detection and Blurring with YOLO

BlurryFace is a Python-based tool for detecting and blurring faces in images using a YOLO (You Only Look Once) model. The application is designed for privacy protection, automatically identifying faces in an image and applying a Gaussian blur to each detected face.

> [!NOTE]
  This project uses a YOLOv11 model trained on a dataset of 1,000 images. The model is optimized specifically for face detection, ensuring a high degree of accuracy in detecting faces for privacy-related applications.

> [!IMPORTANT]
  The model is currently in **Beta** version. While it performs well on most images, it may still have some limitations or inaccuracies in certain cases. Sometimes, you may need to play around with the parameters (`conf`, `iou`) to get the best results. I'm continuously working to improve its accuracy and robustness </br>(～￣▽￣)～.
## Features

- **Face Detection**: Detects faces in an image using a custom-trained YOLO model.
- **Gaussian Blur**: Blurs detected faces to ensure privacy.
- **Customizable Parameters**: Set confidence, IoU, and blur radius thresholds as needed.

## Prerequisites

- Python 3.11
- Poetry (for dependency management and virtual environment)

## Installation

 > [!IMPORTANT]
    This project uses [Poetry](https://python-poetry.org/) for dependency management

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/blurryface.git
   cd blurryface

2. **Install dependencies**
    ```bash
    poetry install
    ```
## Usage
Once installed, you can use BlurryFace to process an image by running the following command:
```bash
poetry run blurryface "path/to/your/image.jpg" --conf=0.4 --iou=0.7 --rad=20
```
### Arguments
1. `image_path`: Path to the image you want to process.
2. `--conf`: Confidence threshold for face detection. Default is `0.4`.
3. `--iou`: Intersection over Union (IoU) threshold for face detection. Default is `0.45`.
4. `--rad`: Radius for the Gaussian blur applied to each detected face. Default is `20`.

## Example
To blur faces in `image.jpg` with custom settings, run:
```bash
poetry run blurryface "path/to/image.jpg" --conf=0.5 --iou=0.6 --rad=15
```
