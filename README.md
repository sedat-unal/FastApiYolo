# YOLO Object Detection API

This repository contains a FastAPI-based service for image-based object detection using the YOLOv8 model from Ultralytics. The service allows you to upload an image and receive a JSON response with detected objects, including their labels, confidence scores, and bounding boxes.

## Features

- **FastAPI**: A modern, fast web framework for building APIs with Python 3.7+.
- **YOLOv8**: Leverages the pre-trained YOLOv8 model from Ultralytics for object detection.
- **Async Support**: Fully asynchronous API endpoints for efficient processing.

## Installation

### Prerequisites

- Python 3.7+
- Conda or virtualenv for environment management
- OpenCV and other dependencies specified in `requirements.txt`

### Setup Using Conda (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/yolo-object-detection-api.git
   cd yolo-object-detection-api
    ```
