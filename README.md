# Facial Detection System with FastAPI

This project implements a professional facial detection system using FastAPI, OpenCV, and MediaPipe. It provides endpoints for face detection, facial landmarks detection, and face posture analysis.

## Features

- Real-time face detection
- Facial landmarks detection
- Face posture analysis (head pose estimation)
- RESTful API endpoints
- WebRTC support for real-time video streaming

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /api/v1/detect-face`: Upload an image for face detection
- `POST /api/v1/detect-landmarks`: Get facial landmarks from an image
- `POST /api/v1/analyze-posture`: Analyze face posture from an image
- `GET /api/v1/health`: Health check endpoint

## WebRTC Integration

The system is designed to work with WebRTC for real-time video streaming. The frontend (Next.js) can connect to these endpoints using WebRTC for live face detection and analysis.

## License

MIT #   F a c i a l D e t e c t i o n - F a s t A P I  
 