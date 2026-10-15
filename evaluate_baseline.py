
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from product_dataset import ProductDataset, CLASSES
from product_model import ProductCNN


PROJECT_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = PROJECT_DIR / "best_baseline_model.pth"
TEST_CSV = PROJECT_DIR / "test.csv"

test_dataset = ProductDataset(TEST_CSV)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)

model = ProductCNN().to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

num_classes = len(CLASSES)
confusion = torch.zeros(
    num_classes, num_classes, dtype=torch.int64
)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for actual, predicted in zip(labels, predictions):
            confusion[actual.item(), predicted.item()] += 1

total = confusion.sum().item()
correct = confusion.diag().sum().item()

print(f"Test images: {total}")
print(f"Correct predictions: {correct}")
print(f"Test accuracy: {100 * correct / total:.2f}%")

print("\nPer-class accuracy:")
for index, class_name in enumerate(CLASSES):
    class_total = confusion[index].sum().item()
    class_correct = confusion[index, index].item()
    accuracy = 100 * class_correct / class_total

    print(
        f"{class_name}: {accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )

print("\nConfusion matrix:")
print("Rows = actual class; columns = predicted class")
print("Class order:", CLASSES)
print(confusion)
