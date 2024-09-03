from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from cv2 import imdecode, IMREAD_COLOR
from numpy import frombuffer, uint8
from ultralytics import YOLO
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize YOLO model
    app.state.yolo_models = YOLO(os.path.join(os.getcwd(), "ai_models", "yolov8n.pt"))
    yield


app = FastAPI(
    title="YOLO Object Detection API",
    description="A FastAPI service for image-based object detection using YOLOv8, leveraging a pre-trained ultralytics model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", status_code=200)
async def predict(image: UploadFile = File(...)):
    """
    Endpoint for detecting the type of object in an uploaded image.

    Parameters:
    - image: UploadFile - The image file for object type detection.

    Returns:
    - JSON YOLO Response: Response containing the status and detection result.

    Raises:
    - HTTPException: If any error occurs during the type detection process.
    """

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Content type not supported")

    contents = await image.read()
    arr_image = frombuffer(contents, dtype=uint8)
    image = imdecode(arr_image, IMREAD_COLOR)

    # Run prediction
    predictions = app.state.yolo_models(image)

    response = []
    for pred in predictions:
        for box in pred.boxes:
            result = {
                "label": pred.names[int(box.cls)],
                "confidence": float(box.conf),
                "box": box.xyxy.tolist()
            }
            response.append(result)

    return {"prediction": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="debug",
        reload=True,
    )
