from fastapi.testclient import TestClient
# import the FastAPI application that we are testing
import app.main as main

import io
from PIL import Image


# create a TestClient connected to our application
client = TestClient(main.app)

# send a GET request to the health endpoint and check the response
def test_health_endpoint():

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint(monkeypatch):
    # define a fake model configuration
    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID", 
            "Lung_Opacity", 
            "Normal", 
            "Viral Pneumonia"
            ],
    }

    # create fake model-loading function  
    def fake_load_model():
        return object(), fake_config

    # create a fake prediction function
    def fake_predict_image(image_path, model, config, device):
        return {
            "predicted_class": "Normal",
            "confidence": 0.95,
            "probabilities": {
                "COVID-19": 0.01,
                "Lung_Opacity": 0.02,
                "Normal": 0.95,
                "Viral Pneumonia": 0.02
            }
        }

    # replace the real functions with our fake functions
    monkeypatch.setattr(main, "load_model", fake_load_model)
    monkeypatch.setattr(main, "predict_image", fake_predict_image)

    # create a sample image in memory  
    image_bytes = io.BytesIO()

    Image.new("RGB", (224, 224)).save(
        image_bytes,
        format="JPEG"
    )

    image_bytes.seek(0)

    with TestClient(main.app) as client:
            
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test_xray.jpeg",
                    image_bytes,
                    "image/jpeg"
                )
            }
        )

    # check the response
    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "Normal"
    assert data["confidence"] == 0.95
    assert len(data["probabilities"]) == 4



def test_invalid_file_upload(monkeypatch):

    # Define a fake model configuration
    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    }

    # Mock model loading
    def fake_load_model():
        return object(), fake_config

    monkeypatch.setattr(main, "load_model", fake_load_model)

    # Create fake PDF content
    fake_pdf = io.BytesIO(b"%PDF-1.4 This is a fake PDF file")

    with TestClient(main.app) as client:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "document.pdf",
                    fake_pdf,
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is not a valid image."