
import torch
from torchvision import transforms
from model import CNN

# Use the same normalization as training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Re-create the architecture
model = CNN()

# Load the saved weights
model.load_state_dict(
    torch.load("best_model.pth", map_location="cpu", weights_only=True)
)

model.eval()

print("Model loaded successfully!")



from torchvision import datasets

# Load the test dataset
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Select one image
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


for i in range(5):
    image, true_label = test_dataset[i]

    # Add batch dimension
    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_batch)
        predicted_label = outputs.argmax(dim=1).item()

    print(
        f"Image {i}: "
        f"Predicted = {class_names[predicted_label]}, "
        f"Actual = {class_names[true_label]}"
    )


# Add a batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
image = image.unsqueeze(0)

# Make a prediction without calculating gradients
with torch.no_grad():
    outputs = model(image)
    predicted_label = outputs.argmax(dim=1).item()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("Predicted:", class_names[predicted_label])
print("Actual:", class_names[true_label])



import matplotlib.pyplot as plt

# Convert the image tensor into a displayable image
display_image = image.squeeze().numpy()

plt.imshow(display_image, cmap="gray")
plt.title(
    f"Predicted: {class_names[predicted_label]}\n"
    f"Actual: {class_names[true_label]}"
)
plt.axis("off")
plt.show()
