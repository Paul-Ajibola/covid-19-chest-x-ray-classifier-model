from app.model import load_model


model, config = load_model()


print("Model loaded successfully!")
print("Architecture:", config["model_architecture"])
print("Classes:", config["class_names"])
print("Device:", next(model.parameters()).device)


