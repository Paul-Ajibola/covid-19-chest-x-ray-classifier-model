import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


# locate the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the model artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "densenet121_chest_xray.pth"

CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"


# select the available device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



def load_model():
    # load model configuration
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    # reconstruct the DenseNet-121 model architecture
    model = models.densenet121(weights=None)

    # replace the final classification layer to match the no. of classes in my dataset
    model.classifier = nn.Linear(
        model.classifier.in_features,
        config["num_classes"]
    )

    # load the trained model weights
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )


    model.load_state_dict(checkpoint["model_state_dict"])


    # move model to the selected device
    model = model.to(DEVICE)

    # set the model to evaluation mode
    model.eval()

    return model, config