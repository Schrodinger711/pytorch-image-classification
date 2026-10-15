
from pathlib import Path
import pandas as pd

# Dataset locations
DATA_DIR = Path(r"E:\FashionProductImages")
CSV_PATH = DATA_DIR / "styles.csv"
IMAGE_DIR = DATA_DIR / "images"

# Categories for our first model
CLASSES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Sandals",
    "Handbags",
]

# Read product information
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

# Keep only the selected categories
df = df[df["articleType"].isin(CLASSES)].copy()

# Build each image's file path
df["image_path"] = df["id"].apply(
    lambda image_id: str(IMAGE_DIR / f"{image_id}.jpg")
)

# Remove records whose image files are missing
df = df[df["image_path"].apply(lambda path: Path(path).is_file())]

# Save the cleaned metadata
output_path = Path(__file__).resolve().parent / "product_data.csv"
df.to_csv(output_path, index=False)

print("Dataset preparation complete!")
print("Total selected images:", len(df))
print("\nImages per category:")
print(df["articleType"].value_counts())
print("\nSaved metadata to:", output_path)
