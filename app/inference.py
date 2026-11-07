"""
Image inference module for the chest X-ray classification application.

This module handles image preprocessing and prediction using the
trained DenseNet-121 model.

It resizes input X-ray images, converts them into tensors, applies
the normalization parameters used during training, and passes the
processed images through the loaded model.

The module returns the predicted class, prediction confidence,
and class-wise probability scores.
"""


import warnings

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

MAX_IMAGE_PIXELS = 25_000_000


def create_transform(config):
    """
    Prepare an X-ray image using the same preprocessing
    applied during model training. It returns a transformation pipeline
    with the configured image size, normalization mean, and standard deviation
    in my model configuration.
    """
    return transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["mean"], std=config["std"]
        )
    ])



def predict_image(image_path, model, config, device):
    """
    Load an X-ray image and generate a prediction
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            with Image.open(image_path) as image:

                # Check actual image format
                if image.format not in ("JPEG", "PNG"):
                    raise ValueError(
                        "Unsupported image format. Please upload a JPEG or PNG image."
                    )

                # Check image dimensions
                width, height = image.size

                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        "Image dimensions exceed the allowed limit."
                    )

                # Verify image integrity
                image.verify()

            # Reopen the image because verify() invalidates the image object
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                # Force image decoding while the file is open
                image.load()

    except ValueError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning
        ):
        raise ValueError(
            "The uploaded file is not a valid or supported image."
        )


    # apply preprocessing
    transform = create_transform(config)
    image_tensor = transform(image)

    # add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # generate prediction
    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        if not torch.isfinite(probabilities).all():
            raise RuntimeError(
                "Model produced invalid probability values."
            )

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    # convert predicted index to class name
    predicted_class = config["class_names"][
        predicted_index.item()
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "probabilities": {
            class_name: round(probability.item(), 4)
            for class_name, probability in zip(
                config["class_names"], probabilities[0]
            )
        }
    }



