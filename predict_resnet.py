
from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from product_dataset import CLASSES


# 1. Project settings
PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_DIR / "best_resnet_finetuned.pth"


# 2. Image preprocessing must match training
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# 3. Re-create the trained model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)

model = model.to(DEVICE)
model.eval()


# 4. Read the image path supplied by the user
if len(sys.argv) < 2:
    print("Usage: run predict_resnet.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"Error: Image file not found: {image_path}")
    sys.exit(1)


# 5. Load and preprocess the image
with Image.open(image_path) as image:
    image = image.convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image)

# Add a batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
image_tensor = image_tensor.unsqueeze(0).to(DEVICE)


# 6. Make a prediction
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

predicted_index = probabilities.argmax().item()
predicted_class = CLASSES[predicted_index]
confidence = probabilities[predicted_index].item() * 100


# 7. Display the result
print(f"Predicted category: {predicted_class}")
print(f"Model confidence: {confidence:.2f}%")

print("\nScores for all categories:")
for index in probabilities.argsort(descending=True).tolist():
    print(f"{CLASSES[index]}: {probabilities[index].item() * 100:.2f}%")
