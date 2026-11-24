from pathlib import Path

DATASET = Path("dataset")

classes = ["benign", "malignant"]

print("=" * 60)
print("CURRENT DATASET VERIFICATION")
print("=" * 60)

total_all = 0

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    split_total = 0

    for cls in classes:

        folder = DATASET / split / cls

        count = len(list(folder.glob("*.png")))

        print(f"{cls:10s}: {count:,}")

        split_total += count

    print(f"Total      : {split_total:,}")

    total_all += split_total

print("\n" + "=" * 60)
print(f"TOTAL IMAGES : {total_all:,}")
print("=" * 60)