# COVID-19 Chest X-Ray Multiclass Classification

**A Deep Learning Application for Chest X-Ray Classification using DenseNet-121, PyTorch, FastAPI, Docker, and GitHub Actions**

[![CI](https://github.com/Paul-Ajibola/covid-19-chest-x-ray-classifier-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Paul-Ajibola/covid-19-chest-x-ray-classifier-model/actions)

## Dataset: (/kaggle/input/datasets/tawsifurrahman/covid19-radiography-database)
## Overview

This project is an end-to-end machine learning application designed to classify chest X-ray images into four categories:

- COVID
- Lung Opacity
- Normal
- Viral Pneumonia

The project combines deep learning, computer vision, API development, automated testing, containerization, and deployment.

A DenseNet-121 convolutional neural network was trained using PyTorch and selected as the final model after evaluation against a ResNet-18 baseline.

The trained model is integrated into a FastAPI application that accepts chest X-ray images and returns a predicted class and associated confidence score.

The application is packaged using Docker and tested through an automated Continuous Integration (CI) pipeline built with GitHub Actions.

Cloud deployment to Render is currently in progress.

> **Important medical disclaimer:** This application is an educational and machine learning engineering project. It is not a medical device, is not clinically validated, and must not be used to diagnose, treat, or make clinical decisions about patients.

---

## Project Objectives

The main objectives of this project were to:

1. Develop a multiclass chest X-ray classification model.
2. Perform exploratory data analysis and prepare the dataset for machine learning.
3. Train and evaluate multiple deep learning architectures.
4. Select an appropriate final model based on evaluation metrics.
5. Package the trained model for inference.
6. Develop a REST API using FastAPI.
7. Implement input validation and error handling.
8. Write automated API tests using pytest.
9. Apply code quality checks using Ruff.
10. Containerize the application using Docker.
11. Automate testing and Docker image validation using GitHub Actions.
12. Deploy the application to a cloud hosting platform.
13. Develop a reproducible and portfolio-ready machine learning engineering project.

---

## Project Architecture

The application follows a modular architecture that separates model loading, inference, API logic, and validation.

```text
                   Chest X-Ray Image
                          |
                          v
                    FastAPI API
                          |
                          v
                  Input Validation
                          |
                          v
                  Image Preprocessing
                          |
                          v
                   DenseNet-121
                          |
                          v
                  Model Prediction
                          |
                          v
                Predicted Class + Confidence
```

### Deployment Architecture

The planned production architecture is:

```text
Developer
   |
   v
GitHub Repository
   |
   v
GitHub Actions
   |
   +-- Ruff Linting
   |
   +-- Pytest
   |
   +-- Docker Image Build
   |
   v
Render
   |
   v
Dockerized FastAPI Application
   |
   v
Public API
```

The GitHub Actions workflow validates the code and builds the Docker image. Render uses the GitHub repository and Dockerfile to build and deploy the application.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Computer vision models and transformations |
| DenseNet-121 | Final classification architecture |
| ResNet-18 | Baseline classification architecture |
| FastAPI | REST API development |
| Pydantic | Data validation and response schemas |
| Pillow | Image processing |
| NumPy | Numerical operations |
| pytest | Automated testing |
| HTTPX | API testing |
| Ruff | Python linting |
| Docker | Application containerization |
| GitHub Actions | Continuous Integration |
| Render | Cloud deployment target |

---

# 1. Dataset and Data Preparation

The project uses chest X-ray images belonging to four classification categories.

### Classification categories

| Class index | Class name |
|---|---|
| 0 | COVID |
| 1 | Lung_Opacity |
| 2 | Normal |
| 3 | Viral Pneumonia |

The dataset preparation and exploratory data analysis stages were completed before model development.

The preparation process included:

- Organizing images according to their classification labels.
- Examining the dataset structure.
- Inspecting class distributions.
- Preparing the data for deep learning.
- Applying image preprocessing and normalization.
- Preparing data for model training and evaluation.

### Image preprocessing

The final model uses a 224 × 224 image input size.

The preprocessing pipeline includes:

1. Resizing the input image.
2. Converting the image to a PyTorch tensor.
3. Normalizing image values using ImageNet normalization parameters.

Normalization parameters:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

These parameters are used to maintain consistency between model training and inference.

The original training dataset is not required to run the deployed API because the trained model and configuration are included in the application artifacts.

> Dataset attribution, the original dataset URL, licensing information, and dataset-specific limitations should be documented here before public release.

---

# 2. Model Development

Two deep learning architectures were investigated.

## ResNet-18 Baseline

ResNet-18 was implemented as the baseline classification model.

The purpose of the baseline was to establish an initial performance benchmark and provide a reference for evaluating the final architecture.

ResNet-18 uses residual connections to support the training of deeper neural networks.

## DenseNet-121 Final Model

DenseNet-121 was selected as the final model.

DenseNet architectures use dense connectivity between layers, allowing feature information to be reused throughout the network.

The pretrained DenseNet-121 architecture was adapted for the four-class classification task.

The original classification head was replaced with a custom classifier containing four output classes.

The final trained model was exported and integrated into the FastAPI application.

### Model artifacts

The project includes:

```text
artifacts/
├── densenet121_chest_xray.pth
└── model_config.json
```

#### `densenet121_chest_xray.pth`

Contains the trained DenseNet-121 model weights.

#### `model_config.json`

Contains configuration information needed to reconstruct the model and maintain consistency between training and inference.

The configuration includes information such as:

- Model architecture.
- Image input size.
- Number of output classes.
- Class names and mapping.
- Image normalization parameters.

The model is loaded during application startup rather than being retrained whenever the API receives a request.

---

# 3. Model Evaluation

Both architectures were evaluated using classification metrics.

The evaluation included:

- Accuracy
- Precision
- Recall
- F1-score

These metrics were used to compare the models and identify the architecture that performed better on the classification task.

## Model comparison

| Metric | ResNet-18 | DenseNet-121 |
|---|---:|---:|
| Accuracy | 84.70% | 85.64% |
| Precision | 87.13% | 86.61% |
| Recall | 83.42% | 85.13% |
| F1-score | 84.75% | 85.80% |

### Selected model: DenseNet-121

DenseNet-121 was selected because it achieved:

- Higher accuracy.
- Higher recall.
- Higher F1-score.

ResNet-18 achieved slightly higher precision, but DenseNet-121 demonstrated better overall performance across the evaluation metrics.

### Interpretation

The final model achieved approximately:

- 85.64% accuracy.
- 85.13% recall.
- 85.80% F1-score.

These results indicate that the model learned useful patterns for distinguishing the four chest X-ray categories.

However, these metrics alone do not establish clinical reliability.

Further validation would be required before considering any real-world medical application.

---

# 4. FastAPI Application

The trained DenseNet-121 model was integrated into a FastAPI application.

FastAPI provides an HTTP interface through which users can submit chest X-ray images for inference.

The application separates API logic from model loading and prediction logic.

## API functionality

The application supports:

- Health checks.
- Readiness checks.
- Image upload.
- Image validation.
- Image preprocessing.
- Model inference.
- Prediction responses.
- Error handling.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Checks whether the API is running |
| `/ready` | GET | Checks whether the application is ready to serve predictions |
| `/predict` | POST | Accepts a chest X-ray image and returns a prediction |
| `/docs` | GET | Interactive Swagger API documentation |

### Health endpoint

```http
GET /health
```

This endpoint checks the application's health.

It is useful for monitoring whether the API process is responding.

### Readiness endpoint

```http
GET /ready
```

This endpoint checks whether the application is ready to receive prediction requests.

It is particularly important because the application depends on a trained model being loaded successfully.

### Prediction endpoint

```http
POST /predict
```

The prediction endpoint accepts an uploaded chest X-ray image.

The application performs the following operations:

1. Receives the uploaded image.
2. Validates the upload.
3. Checks file size and image validity.
4. Opens and preprocesses the image.
5. Converts the image into the format expected by DenseNet-121.
6. Runs inference using the trained model.
7. Determines the predicted class.
8. Returns the prediction and confidence information.

### Interactive API documentation

FastAPI automatically generates Swagger documentation.

After starting the application, the documentation is available at:

```text
http://localhost:8000/docs
```

The Swagger interface allows users to:

- Inspect the available endpoints.
- Upload an X-ray image.
- Execute a prediction request.
- Inspect the HTTP response.

---

# 5. Input Validation and Error Handling

The API was designed to handle incorrect input rather than crashing when invalid files are uploaded.

Validation includes:

- Invalid file uploads.
- Corrupted image files.
- Empty uploads.
- Oversized files.
- Valid image uploads.

The application implements a maximum upload size of 10 MB.

Files exceeding the configured limit are rejected.

This helps protect the application from unnecessarily large requests and reduces the risk of resource exhaustion.

The validation logic is supported by automated tests.

---

# 6. Automated Testing

The application uses pytest for automated testing.

Tests are located in:

```text
tests/
```

The API tests use FastAPI's testing utilities and HTTPX to send requests to the application.

## Testing coverage

The test suite includes tests for:

- Health endpoint behavior.
- Prediction endpoint behavior.
- Valid image uploads.
- Invalid file uploads.
- Corrupted image uploads.
- Empty file uploads.
- Oversized uploads.
- API validation and error handling.

The project previously reached **16 passing pytest tests**.

The tests are designed to verify the API's behavior without requiring users to manually test every request through Swagger.

### Running the tests

From the project root:

```bash
pytest -v
```

To run the tests with coverage information:

```bash
pytest --cov=app --cov-report=term-missing
```

The coverage command requires `pytest-cov` to be installed.

---

# 7. Code Quality with Ruff

Ruff is used to identify Python code quality and style issues.

The project includes Ruff checks as part of the automated CI workflow.

### Run Ruff locally

```bash
ruff check app tests
```

Expected successful output:

```text
All checks passed!
```

Linting helps identify issues such as:

- Unused imports.
- Undefined names.
- Common Python errors.
- Code style violations.

Ruff is executed automatically through GitHub Actions.

---

# 8. Docker Containerization

The application is containerized using Docker.

Docker packages the application, Python environment, dependencies, model artifacts, and startup configuration into a reproducible image.

This allows the application to run in a consistent environment across development and deployment.

## Docker components

The project includes:

```text
Dockerfile
.dockerignore
```

### Dockerfile

The Dockerfile defines:

- The base Python environment.
- Dependency installation.
- Application file copying.
- Model artifact inclusion.
- Application startup configuration.

The container runs the FastAPI application using Uvicorn.

### `.dockerignore`

The `.dockerignore` file excludes unnecessary files from the Docker build context.

Examples include:

- Python virtual environments.
- Python cache files.
- Git metadata.
- IDE configuration.
- Local environment files.
- Test cache files.

The trained model artifacts and application source code must remain included in the Docker build context.

## Build the Docker image

```bash
docker build -t chest-xray-api:local .
```

## Run the container

The following command assumes that the Dockerfile exposes the application on port 8000:

```bash
docker run -d \
  --name chest-xray-api \
  -p 8000:8000 \
  chest-xray-api:local
```

For PowerShell:

```powershell
docker run -d `
  --name chest-xray-api `
  -p 8000:8000 `
  chest-xray-api:local
```

## Inspect running containers

```bash
docker ps
```

## View container logs

```bash
docker logs chest-xray-api
```

## Stop the container

```bash
docker stop chest-xray-api
```

## Remove the container

```bash
docker rm chest-xray-api
```

The application has been built and tested using Docker in GitHub Codespaces.

---

# 9. Continuous Integration with GitHub Actions

The project includes a GitHub Actions CI workflow.

Workflow location:

```text
.github/workflows/ci.yml
```

The workflow is triggered by:

- Push events.
- Pull request events.

## CI workflow

The workflow performs the following steps:

1. Checks out the repository.
2. Sets up Python.
3. Installs dependencies.
4. Runs Ruff linting.
5. Runs pytest.
6. Builds the Docker image.

### Workflow configuration

```yaml
name: Chest X-Ray API CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Ruff linting
        run: ruff check app tests

      - name: Run tests
        run: pytest -v

      - name: Build Docker image
        run: docker build -t chest-xray-api:ci .
```

### Why CI matters

Continuous Integration automatically checks whether new changes break the application.

For example, if a change causes a pytest failure, GitHub Actions reports the failure before the change can be considered validated.

This provides a repeatable quality-checking process.

### Current CI status

The following CI components have been implemented and validated:

- Python environment setup.
- Dependency installation.
- Ruff linting.
- Pytest execution.
- Docker image build validation.

---

# 10. Cloud Deployment with Render

The application is being deployed using Render.

The deployment target is Render's free tier, subject to its current resource limits and availability.

## Deployment approach

Render connects directly to the GitHub repository and uses the project's Dockerfile to build the application.

The deployment process is intended to follow this sequence:

1. Push code changes to GitHub.
2. GitHub Actions executes CI checks.
3. Render builds the Docker image from the repository.
4. Render starts the FastAPI application.
5. The application becomes accessible through a public URL.

### Render configuration

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Docker |
| Repository | `Paul-Ajibola/covid-19-chest-x-ray-classifier-model` |
| Dockerfile path | `./Dockerfile` |
| Docker command | Use the Dockerfile's default command |
| Instance type | Free, where available |

The application must listen on the appropriate port and bind to:

```text
0.0.0.0
```

The exact port configuration must match the Dockerfile and Render service configuration.

### Deployment status

**Render deployment is currently in progress.**

The final public API URL will be added here after deployment succeeds.

```text
Live API: Pending deployment
Swagger documentation: Pending deployment
```

Render's free services may sleep after periods of inactivity and experience cold starts.

Free-tier availability and resource limits may change.

---

# 11. Project Structure

The project follows a modular structure.

```text
covid-19-chest-x-ray-multiclass-classification/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── inference.py
│   └── schemas.py
│
├── artifacts/
│   ├── densenet121_chest_xray.pth
│   └── model_config.json
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

### Main components

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI application and endpoints |
| `app/model.py` | Model architecture and model loading |
| `app/inference.py` | Image preprocessing and prediction logic |
| `app/schemas.py` | API request and response schemas |
| `artifacts/` | Trained model and model configuration |
| `tests/test_api.py` | Automated API tests |
| `Dockerfile` | Docker image configuration |
| `.dockerignore` | Excludes unnecessary Docker build files |
| `.github/workflows/ci.yml` | Automated CI pipeline |
| `requirements.txt` | Python dependencies |

---

# 12. Installation and Local Development

## Prerequisites

The following tools are required for local development:

- Python 3.11
- Git
- Docker, if running the container locally

## Clone the repository

```bash
git clone https://github.com/Paul-Ajibola/covid-19-chest-x-ray-classifier-model.git
```

Navigate into the project:

```bash
cd covid-19-chest-x-ray-classifier-model
```

## Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

## Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start the FastAPI application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be accessible locally at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 13. Example API Usage

## Using Swagger

Open:

```text
http://localhost:8000/docs
```

1. Expand `POST /predict`.
2. Click **Try it out**.
3. Upload a chest X-ray image.
4. Click **Execute**.
5. Inspect the prediction response.

## Using cURL

Example:

```bash
curl -X POST \
  "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_xray.png"
```

Replace `sample_xray.png` with the path to a valid chest X-ray image.

The API returns a response containing the model's prediction information according to the response schema implemented in the application.

---

# 14. Limitations

This project has several important limitations.

### Model limitations

- The model is limited to four predefined classes.
- Classification performance depends on the training dataset.
- Performance may vary across hospitals, populations, imaging equipment, and acquisition protocols.
- The reported evaluation metrics do not establish clinical safety.
- The model may produce incorrect predictions, including predictions with high confidence.

### Dataset limitations

The model's generalization depends on the quality, representativeness, and labeling accuracy of the training data.

Potential dataset biases and class imbalance may affect predictions.

### Deployment limitations

The application is intended for demonstration and portfolio purposes.

Render's free tier has limited resources and may experience cold starts.

The model may require additional optimization before deployment to environments with strict latency or memory requirements.

### Medical limitations

This application is not intended to replace radiologists or other qualified healthcare professionals.

It should not be used for patient diagnosis or treatment decisions.

---


# 15. Skills Demonstrated

This project demonstrates practical experience with:

### Machine Learning

- Deep learning.
- Computer vision.
- Transfer learning.
- Model training.
- Model evaluation.
- Multiclass classification.
- PyTorch inference.

### Backend Engineering

- REST API development.
- FastAPI.
- Pydantic validation.
- Image upload handling.
- API error handling.
- Model serving.

### Software Engineering

- Modular Python application design.
- Automated testing.
- Ruff linting.
- Git and GitHub.
- Docker containerization.

### MLOps

- Model artifact management.
- Reproducible inference.
- Continuous Integration.
- Automated Docker builds.
- Cloud deployment preparation.

---

# 17. Author

**Paul Ajibola**

- GitHub: [Paul-Ajibola](https://github.com/Paul-Ajibola)
- LinkedIn: [Paul Ajibola](https://www.linkedin.com/in/paul-ajibola/)

---

# Conclusion

This project demonstrates the development of an end-to-end machine learning application, beginning with dataset preparation and model training and extending through API development, automated testing, Docker containerization, and cloud deployment.

Rather than stopping at model training, the project focuses on integrating the trained model into a usable and reproducible software application.

The objective is to demonstrate practical machine learning engineering skills, including the ability to develop, test, package, and deploy a deep learning model through a production-oriented workflow.

**The project is intended for educational and portfolio purposes and is not clinically validated.**
