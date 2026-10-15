
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_DIR / "product_data.csv")

# Shuffle records within each category for balanced splits
train_parts = []
val_parts = []
test_parts = []

for category, group in df.groupby("articleType"):
    group = group.sample(frac=1, random_state=42)

    n = len(group)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_parts.append(group.iloc[:train_end])
    val_parts.append(group.iloc[train_end:val_end])
    test_parts.append(group.iloc[val_end:])

train = pd.concat(train_parts).sample(frac=1, random_state=42)
val = pd.concat(val_parts).sample(frac=1, random_state=42)
test = pd.concat(test_parts).sample(frac=1, random_state=42)

for name, data in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    data.to_csv(PROJECT_DIR / f"{name}.csv", index=False)
    print(f"\n{name.upper()}: {len(data)} images")
    print(data["articleType"].value_counts())

print("\nSplits saved successfully!")
