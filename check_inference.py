from app.model import load_model, DEVICE
from app.inference import predict_image



# load the trained model and configuration
model, config = load_model()


# specify the X-ray image path
image_path = r"C:\Users\USER\Downloads\covid-19-chest-x-ray-multiclass-classification\sample_images\COVID-X-RAY.jpeg"


# generate prediction
result = predict_image(
    image_path=image_path,
    model=model,
    config=config,
    device=DEVICE
)


# display results
print("\nINFERENCE RESULTS")
print("-----------------")


print("Predicted class:", result["predicted_class"])
print("Confidence:", result["confidence"])

print("\nClass Probabilities:")

for class_name, probability in result["probabilities"].items():
    print(f"{class_name}: {probability}")


