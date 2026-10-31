from fastapi.testclient import TestClient
# import the FastAPI application that we are testing
import app.main as main

import io
from PIL import Image
import torch


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


def test_corrupted_image_upload(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    }

    def fake_load_model():
        return object(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    corrupted_image = io.BytesIO(
        b"This is not a real image"
    )

    with TestClient(main.app) as client:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "scan.jpeg",
                    corrupted_image,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "The uploaded file is not a valid image."
    )



def test_empty_image_upload(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    }

    def fake_load_model():
        return object(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    empty_image = io.BytesIO(b"")

    with TestClient(main.app) as client:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "scan.jpeg",
                    empty_image,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "The uploaded file is not a valid image."
    )



def test_missing_file_upload():

    with TestClient(main.app) as client:

        response = client.post("/predict")

    assert response.status_code == 422



def test_oversized_file_upload(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    }

    def fake_load_model():
        return object(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    oversized_file = io.BytesIO(
        b"x" * (10 * 1024 * 1024 + 1)
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "large_image.jpeg",
                    oversized_file,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "File size exceeds the 10 MB limit."
    )




def test_valid_jpeg_upload(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "image_size": 224,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }

    # Create a fake model that returns four class scores
    class FakeModel:

        def __call__(self, image_tensor):
            return torch.tensor([[0.1, 0.2, 0.6, 0.1]])

    def fake_load_model():
        return FakeModel(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    # Create a genuine JPEG image
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

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "Normal"
    assert len(data["probabilities"]) == 4


def test_valid_png_upload(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "image_size": 224,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }

    class FakeModel:

        def __call__(self, image_tensor):
            return torch.tensor([[0.1, 0.2, 0.6, 0.1]])

    def fake_load_model():
        return FakeModel(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    # Create a genuine PNG image
    image_bytes = io.BytesIO()

    Image.new("RGB", (224, 224)).save(
        image_bytes,
        format="PNG"
    )

    image_bytes.seek(0)

    with TestClient(main.app) as client:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test_xray.png",
                    image_bytes,
                    "image/png"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "Normal"
    assert len(data["probabilities"]) == 4


def test_unsupported_gif_upload(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "image_size": 224,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }

    class FakeModel:

        def __call__(self, image_tensor):
            return torch.tensor([[0.1, 0.2, 0.6, 0.1]])

    def fake_load_model():
        return FakeModel(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    # Create a genuine GIF image
    image_bytes = io.BytesIO()

    Image.new("RGB", (224, 224)).save(
        image_bytes,
        format="GIF"
    )

    image_bytes.seek(0)

    with TestClient(main.app) as client:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test_xray.gif",
                    image_bytes,
                    "image/gif"
                )
            }
        )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "Unsupported image format. Please upload a JPEG or PNG image."
    )

def test_unexpected_prediction_error(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    }

    class FakeModel:
        pass

    def fake_load_model():
        return FakeModel(), fake_config

    # Simulate an unexpected inference error
    def fake_predict_image(*args, **kwargs):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    monkeypatch.setattr(
        main,
        "predict_image",
        fake_predict_image
    )

    # Create a small valid JPEG image
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

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "An unexpected error occurred while processing the image."
    )


def test_readiness_endpoint(monkeypatch):

    fake_config = {
        "num_classes": 4,
        "class_names": [
            "COVID",
            "Lung_Opacity",
            "Normal",
            "Viral Pneumonia"
        ]
    }

    def fake_load_model():
        return object(), fake_config

    monkeypatch.setattr(
        main,
        "load_model",
        fake_load_model
    )

    with TestClient(main.app) as client:
        response = client.get("/ready")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ready",
        "model_loaded": True
    }


def test_readiness_endpoint_model_not_loaded():

    with TestClient(main.app) as client:

        main.ml_models.clear()

        response = client.get("/ready")

    assert response.status_code == 503

    assert response.json()["detail"] == "Model is not ready."


def test_empty_file_upload():
    image_bytes = io.BytesIO()

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "empty.jpeg",
                    image_bytes,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The uploaded file is empty."
    )


def test_unsupported_content_type():
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
                    "test.jpeg",
                    image_bytes,
                    "application/pdf"
                )
            }
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported content type. Please upload a JPEG or PNG image."
    )


def test_unsupported_actual_image_format():
    image_bytes = io.BytesIO()

    Image.new("RGB", (224, 224)).save(
        image_bytes,
        format="TIFF"
    )

    image_bytes.seek(0)

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpeg",
                    image_bytes,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 400

