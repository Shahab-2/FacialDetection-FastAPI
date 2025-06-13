import requests
import cv2
import numpy as np
import asyncio
import websockets
import json
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test the health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check test passed")

def test_face_detection():
    """Test face detection endpoint with a sample image."""
    # Create a test image with a face
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple face
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)  # Face
    cv2.circle(img, (280, 220), 20, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (360, 220), 20, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (320, 280), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    # Save the test image
    test_image_path = "test_face.jpg"
    cv2.imwrite(test_image_path, img)
    
    # Test the endpoint
    with open(test_image_path, "rb") as f:
        files = {"file": ("test_face.jpg", f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/detect-face", files=files)
    
    # Clean up
    os.remove(test_image_path)
    
    assert response.status_code == 200
    result = response.json()
    assert "faces_detected" in result
    print("✅ Face detection test passed")

def test_landmarks_detection():
    """Test facial landmarks detection endpoint."""
    # Create a test image with a face
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple face
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)  # Face
    cv2.circle(img, (280, 220), 20, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (360, 220), 20, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (320, 280), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    # Save the test image
    test_image_path = "test_landmarks.jpg"
    cv2.imwrite(test_image_path, img)
    
    # Test the endpoint
    with open(test_image_path, "rb") as f:
        files = {"file": ("test_landmarks.jpg", f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/detect-landmarks", files=files)
    
    # Clean up
    os.remove(test_image_path)
    
    assert response.status_code == 200
    result = response.json()
    assert "landmarks_detected" in result
    print("✅ Landmarks detection test passed")

def test_posture_analysis():
    """Test face posture analysis endpoint."""
    # Create a test image with a face
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple face
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)  # Face
    cv2.circle(img, (280, 220), 20, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (360, 220), 20, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (320, 280), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    # Save the test image
    test_image_path = "test_posture.jpg"
    cv2.imwrite(test_image_path, img)
    
    # Test the endpoint
    with open(test_image_path, "rb") as f:
        files = {"file": ("test_posture.jpg", f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/analyze-posture", files=files)
    
    # Clean up
    os.remove(test_image_path)
    
    assert response.status_code == 200
    result = response.json()
    assert "posture_detected" in result
    print("✅ Posture analysis test passed")

async def test_websocket():
    """Test WebSocket connection for real-time face detection."""
    uri = "ws://localhost:8000/ws/face-detection"
    
    async with websockets.connect(uri) as websocket:
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple face
        cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)  # Face
        cv2.circle(frame, (280, 220), 20, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (360, 220), 20, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(frame, (320, 280), (50, 30), 0, 0, 180, (0, 0, 0), 2)  # Smile
        
        # Encode frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Send frame
        await websocket.send(frame_bytes)
        
        # Receive response
        response = await websocket.recv()
        result = json.loads(response)
        
        assert "faces_detected" in result
        print("✅ WebSocket test passed")

def run_tests():
    """Run all tests."""
    print("Running API tests...")
    
    # Test REST endpoints
    test_health()
    test_face_detection()
    test_landmarks_detection()
    test_posture_analysis()
    
    # Test WebSocket
    print("\nTesting WebSocket connection...")
    asyncio.run(test_websocket())
    
    print("\nAll tests completed successfully! 🎉")

if __name__ == "__main__":
    run_tests() 