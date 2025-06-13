import asyncio
import json
from typing import Dict, Any
import cv2
import numpy as np
import mediapipe as mp

class WebRTCStreamHandler:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame and return detection results."""
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_results = self.face_detection.process(rgb_frame)
        faces = []
        
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                faces.append({
                    "confidence": float(detection.score[0]),
                    "bounding_box": {
                        "x": float(bbox.xmin),
                        "y": float(bbox.ymin),
                        "width": float(bbox.width),
                        "height": float(bbox.height)
                    }
                })
        
        # Detect landmarks
        mesh_results = self.face_mesh.process(rgb_frame)
        landmarks = []
        posture = None
        
        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            
            # Extract landmarks
            for landmark in face_landmarks.landmark:
                landmarks.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z)
                })
            
            # Calculate posture
            nose_tip = face_landmarks.landmark[1]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            nose_position = nose_tip.x
            if abs(nose_position - 0.5) < 0.1:
                pose = "center"
            elif nose_position < 0.5:
                pose = "left"
            else:
                pose = "right"
            
            posture = {
                "direction": pose,
                "confidence": float(face_landmarks.landmark[0].visibility)
            }
        
        return {
            "faces_detected": len(faces),
            "faces": faces,
            "landmarks_detected": len(landmarks) > 0,
            "landmarks": landmarks,
            "posture": posture
        }

    async def handle_webrtc_stream(self, websocket):
        """Handle WebRTC stream and process frames."""
        try:
            while True:
                # Receive frame data from WebRTC
                frame_data = await websocket.receive_bytes()
                
                # Convert bytes to numpy array
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame
                results = await self.process_frame(frame)
                
                # Send results back through WebSocket
                await websocket.send_json(results)
                
        except Exception as e:
            print(f"Error in WebRTC stream handling: {str(e)}")
            raise 