import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms

from model import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

train_dataset, val_dataset = random_split(
    train_dataset,
    [54000, 6000]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


model = CNN()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01)

loss_function = nn.CrossEntropyLoss()



num_epochs = 15
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    total_loss = 0
    for batch_index, (images, labels) in enumerate(train_loader):

        outputs = model(images)
        loss = loss_function(outputs, labels)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    average_loss = total_loss / len(train_loader)
    print("Epoch:", epoch + 1, "Average Loss:", average_loss)

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print("Validation Accuracy:", val_accuracy, "%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

model.load_state_dict(torch.load("best_model.pth", weights_only=True))
model.eval()

#Final evaluation on the test set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Test Accuracy:", accuracy, "%")


all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_predictions)

print(cm)


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

class_accuracy = cm.diagonal() / cm.sum(axis=1)

for i, acc in enumerate(class_accuracy):
    print(class_names[i], ":", acc * 100, "%")