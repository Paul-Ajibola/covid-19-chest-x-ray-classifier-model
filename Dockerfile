# start with Python image
FROM python:3.11-slim

# set the working directory inside the container
WORKDIR /app

# copy the dependency list into the container
COPY requirements.txt .

# install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the application source code
COPY app/ ./app/

# copy the frontend and sample image used by the analysis workspace
COPY frontend/ ./frontend/
COPY sample_images/ ./sample_images/

# copy the trained model artifacts
COPY artifacts/ ./artifacts/

# Expose the port used by FastAPI
EXPOSE 8000

# start the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

