
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from product_dataset import ProductDataset
from product_model import ProductCNN


# 1. Project settings
PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

BEST_MODEL_PATH = PROJECT_DIR / "best_baseline_model.pth"


# 2. Load the training and validation datasets
train_dataset = ProductDataset(PROJECT_DIR / "train.csv")
val_dataset = ProductDataset(PROJECT_DIR / "val.csv")

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


# 3. Initialize the model, loss function, and optimizer
model = ProductCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_accuracy = -1.0


# 4. Repeat training and validation for each epoch
for epoch in range(NUM_EPOCHS):

    # Training phase
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

    train_loss = train_loss / train_total
    train_accuracy = 100 * train_correct / train_total


    # Validation phase
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


    # 5. Display results for this epoch
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


    # 6. Save the model only when validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH,
        )

        print(
            f"  Best model saved "
            f"(Validation Accuracy: {best_val_accuracy:.2f}%)"
        )


print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Best model saved at: {BEST_MODEL_PATH}")
