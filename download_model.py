import urllib.request
import os
from pathlib import Path

def download_model():
    # Create app directory if it doesn't exist
    app_dir = Path(__file__).resolve().parent / "app"
    app_dir.mkdir(exist_ok=True)
    
    # Model files
    model_files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    for filename, url in model_files.items():
        filepath = app_dir / filename
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"{filename} downloaded successfully!")

if __name__ == "__main__":
    download_model() 