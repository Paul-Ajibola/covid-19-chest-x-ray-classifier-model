from fastapi import FastAPI, UploadFile, File, HTTPException

# import inference function and PredictionResponse schema
from app.inference import predict_image
from app.schemas import PredictionResponse

from contextlib import asynccontextmanager
from app.model import load_model, DEVICE

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



@app.get("/")
def home():
    return {"message": "CHEST X-RAY CLASSIFICATION API IS RUNNING!!!"}



@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    """
    Endpoint to classify a chest X-ray image.

    Accepts an image file and returns the predicted class, confidence score,
    and class-wise probability scores.
    """
    
    # Get the model and configuration
    model = ml_models["model"]
    config = ml_models["config"]

    try:
        # Generate prediction
        result = predict_image(
            image_path=file.file,
            model=model,
            config=config,
            device=DEVICE
        )

        return result
    
    except Exception as e:
         raise HTTPException(
              status_code=400,
              detail=f"Image Prediction Failed: {str(e)}"
            )

