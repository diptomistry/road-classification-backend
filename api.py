#!/usr/bin/env python3
"""
FastAPI application for YOLOv9 Road Classification Model
Deploy your trained model as a REST API
"""

import os
import io
import base64
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
import requests

# Initialize FastAPI app
app = FastAPI(
    title="Road Classification API",
    description="YOLOv9 model for classifying roads as Broken, Not Broken, or Manhole",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:3000"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global model variable
model = None

def load_model(model_path: str = "runs/road_classification/yolov9_road_damage_fast/weights/best.pt"):
    """Load the trained YOLOv9 model"""
    global model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    model = YOLO(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model

def process_image(image_bytes: bytes, conf_threshold: float = 0.5) -> Dict[str, Any]:
    """Process image and return predictions"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # Run inference
    results = model(image, conf=conf_threshold)
    result = results[0]
    
    # Process results
    detections = []
    if result.boxes is not None:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class and confidence
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # Get class name
            class_name = result.names[cls]
            
            detection = {
                "class": class_name,
                "class_id": cls,
                "confidence": round(conf, 3),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2)
                }
            }
            detections.append(detection)
    
    return {
        "image_shape": {
            "height": image.shape[0],
            "width": image.shape[1]
        },
        "detections": detections,
        "total_detections": len(detections)
    }

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    CHUNK_SIZE = 32768
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def ensure_model():
    model_path = "runs/road_classification/yolov9_road_damage_fast/weights/best.pt"
    file_id = "12gcuryDv6xPULtndkTNAdq3-euJOWKWj"
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")
        download_file_from_google_drive(file_id, model_path)
        print("Model downloaded.")
    else:
        print("Model already exists.")

# Ensure model is present before loading
ensure_model()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Make sure to train the model first or update the model path")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Road Classification API",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.5
):
    """
    Predict road classification for uploaded image
    
    - **file**: Image file (JPG, PNG, etc.)
    - **confidence**: Confidence threshold (0.0 to 1.0)
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    try:
        image_bytes = await file.read()
        result = process_image(image_bytes, confidence)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence: float = 0.5
):
    """
    Predict road classification for multiple images
    
    - **files**: List of image files
    - **confidence**: Confidence threshold (0.0 to 1.0)
    """
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_bytes = await file.read()
            result = process_image(image_bytes, confidence)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "processed_files": len(results),
        "results": results
    }

@app.get("/model_info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_path": str(model.model),
        "classes": list(model.names.values()) if hasattr(model, 'names') else [],
        "num_classes": len(model.names) if hasattr(model, 'names') else 0
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    ) 