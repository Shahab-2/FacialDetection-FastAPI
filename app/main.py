from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import cv2
import numpy as np
from typing import List, Dict, Any
import io
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the current directory
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Facial Detection System",
    description="A professional facial detection system with FastAPI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load face detection model
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("Face detection model loaded successfully")
except Exception as e:
    logger.error(f"Error loading face detection model: {str(e)}")
    raise

def detect_faces(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect faces in the frame and return their locations and landmarks."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        results = []
        height, width = frame.shape[:2]
        if len(faces) > 0:
            # Find the largest face by area
            largest = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest
            confidence = 0.9
            face_center_x = x + w/2
            face_center_y = y + h/2
            landmarks = [
                {"x": (x + w * 0.3) / width, "y": (y + h * 0.3) / height, "type": "left_eye"},
                {"x": (x + w * 0.7) / width, "y": (y + h * 0.3) / height, "type": "right_eye"},
                {"x": (x + w * 0.5) / width, "y": (y + h * 0.5) / height, "type": "nose"},
                {"x": (x + w * 0.3) / width, "y": (y + h * 0.7) / height, "type": "mouth_left"},
                {"x": (x + w * 0.7) / width, "y": (y + h * 0.7) / height, "type": "mouth_right"},
            ]
            frame_center_x = width / 2
            frame_center_y = height / 2
            horizontal_offset = face_center_x - frame_center_x
            horizontal_threshold = width * 0.2
            vertical_offset = face_center_y - frame_center_y
            vertical_threshold = height * 0.2
            if abs(horizontal_offset) < horizontal_threshold and abs(vertical_offset) < vertical_threshold:
                direction = "center"
            elif horizontal_offset < -horizontal_threshold:
                direction = "left"
            else:
                direction = "right"
            results.append({
                "bounding_box": {
                    "x": x / width,
                    "y": y / height,
                    "width": w / width,
                    "height": h / height
                },
                "confidence": confidence,
                "landmarks": landmarks,
                "posture": {
                    "direction": direction,
                    "horizontal_offset": float(horizontal_offset / width),
                    "vertical_offset": float(vertical_offset / height)
                }
            })
        return results
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return []

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main page."""
    index_path = BASE_DIR / "static" / "index.html"
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

def process_image(file: UploadFile) -> np.ndarray:
    """Convert uploaded file to numpy array."""
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        return img
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Error processing image")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/v1/detect-face")
async def detect_face(file: UploadFile = File(...)):
    """Detect faces in the uploaded image."""
    try:
        img = process_image(file)
        faces = detect_faces(img)
        
        return {
            "faces_detected": len(faces),
            "faces": faces
        }
    except Exception as e:
        logger.error(f"Error in face detection endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/detect-landmarks")
async def detect_landmarks_endpoint(file: UploadFile = File(...)):
    """Detect facial landmarks in the uploaded image."""
    try:
        img = process_image(file)
        faces = detect_faces(img)
        
        if not faces:
            return {"landmarks_detected": False, "landmarks": []}
        
        # Get landmarks for the first face
        landmarks = detect_faces(img)
        
        return {
            "landmarks_detected": len(landmarks) > 0,
            "landmarks": landmarks
        }
    except Exception as e:
        logger.error(f"Error in landmarks endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze-posture")
async def analyze_posture(file: UploadFile = File(...)):
    """Analyze face posture (head pose) in the uploaded image."""
    try:
        img = process_image(file)
        faces = detect_faces(img)
        
        if not faces:
            return {"posture_detected": False, "posture": None}
        
        # Get the first face
        face = faces[0]
        face_center_x = face["bounding_box"]["x"] + face["bounding_box"]["width"] / 2
        
        # Determine head pose
        if abs(face_center_x - 0.5) < 0.1:
            pose = "center"
        elif face_center_x < 0.5:
            pose = "left"
        else:
            pose = "right"
        
        return {
            "posture_detected": True,
            "posture": {
                "direction": pose,
                "confidence": face["confidence"]
            }
        }
    except Exception as e:
        logger.error(f"Error in posture analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/face-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "bytes" in message:
                data = message["bytes"]
            elif "text" in message:
                continue  # Ignore text messages
            else:
                continue

            # Convert bytes to numpy array
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("Failed to decode image")
                continue

            # Detect faces
            faces = detect_faces(frame)

            # Prepare response
            response = {
                "faces_detected": len(faces),
                "faces": faces
            }

            # Send response
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 