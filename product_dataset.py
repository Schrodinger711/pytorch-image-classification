
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Handbags",
    "Sandals",
]

CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASSES)
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class ProductDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = IMAGE_TRANSFORM(image)

        label = CLASS_TO_INDEX[row["articleType"]]

        return image, label


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent

    dataset = ProductDataset(project_dir / "train.csv")
    image, label = dataset[0]

    print("Dataset size:", len(dataset))
    print("Image shape:", image.shape)
    print("Label index:", label)
    print("Label name:", CLASSES[label])
