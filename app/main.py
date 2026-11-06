from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# import inference function and PredictionResponse schema
from app.inference import predict_image
from app.schemas import PredictionResponse

from contextlib import asynccontextmanager
from app.model import load_model, DEVICE
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# (1024 bytes)
MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png"
}

# Store the loaded model and configuration
ml_models = {}

# asynccontextmanager turns asynchronous generator function into an asynchronous context manager.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model when the API starts
    model, config = load_model()

    ml_models["model"] = model
    ml_models["config"] = config

    yield

    # clean up when the API shuts down
    ml_models.clear()


# create the FastAPI application

app = FastAPI(
    title="CHEST X-RAY CLASSIFICATION API",
    description="A RESTful API for classifying chest X-ray images into COVID-19, Lung_Opacity, Normal, and Viral Pneumonia categories using a trained DenseNet-121 model.",
    version="1.0.0",
    lifespan=lifespan
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLE_DIR = BASE_DIR / "sample_images"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLE_DIR), name="samples")



@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/model-guide")
def model_guide():
    return FileResponse(FRONTEND_DIR / "index.html")



@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():

    if "model" not in ml_models or "config" not in ml_models:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready."
        )

    return {
        "status": "ready",
        "model_loaded": True
    }



@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to classify a chest X-ray image.

    Accepts an image file and returns the predicted class, confidence score,
    and class-wise probability scores.
    """

    try:
        if "model" not in ml_models or "config" not in ml_models:
            raise HTTPException(
                status_code=503,
                detail="Model is not ready"
            )
        
        # Get the loaded model and its configuration
        model = ml_models["model"]
        config = ml_models["config"]

        

        # Check whether a file was provided
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file was provided."
            )

        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported content type. Please upload a JPEG or PNG image."
            )

        # Check uploaded file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB limit."
            )

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Generate prediction
        result = predict_image(
            image_path=file.file,
            model=model,
            config=config,
            device=DEVICE
        )

        # Return prediction result
        return result

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e

    except Exception:
        logger.exception(
            "Unexpected error during image prediction"
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the image."
        )

    finally:
        # Close the uploaded file after processing
        await file.close()

