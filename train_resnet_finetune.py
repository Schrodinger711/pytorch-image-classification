
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from product_dataset import ProductDataset, CLASSES


PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 8
BATCH_SIZE = 32
PRETRAINED_LR = 0.0001
CLASSIFIER_LR = 0.001

START_MODEL_PATH = PROJECT_DIR / "best_resnet_model.pth"
BEST_MODEL_PATH = PROJECT_DIR / "best_resnet_finetuned.pth"

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = ProductDataset(
    PROJECT_DIR / "train.csv",
    transform=RESNET_TRANSFORM,
)
val_dataset = ProductDataset(
    PROJECT_DIR / "val.csv",
    transform=RESNET_TRANSFORM,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

# Re-create the six-class ResNet18 architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

# Load the weights learned during feature extraction
model.load_state_dict(
    torch.load(
        START_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

# Freeze every parameter first
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final residual block and classifier
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# Use a smaller learning rate for pretrained features
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": PRETRAINED_LR},
    {"params": model.fc.parameters(), "lr": CLASSIFIER_LR},
])

best_val_accuracy = -1.0

for epoch in range(NUM_EPOCHS):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = 100 * train_correct / train_total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(
            f"  Best fine-tuned model saved "
            f"(Validation Accuracy: {best_val_accuracy:.2f}%)"
        )

print("\nFine-tuning complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Saved model: {BEST_MODEL_PATH}")
