from pathlib import Path
import random
import shutil

random.seed(42)

SOURCE = Path("IDC_regular_ps50_idx5")
OUTPUT = Path("dataset")

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

for cls in ["benign", "malignant"]:
    (OUTPUT / "train" / cls).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "val" / cls).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "test" / cls).mkdir(parents=True, exist_ok=True)

benign = []
malignant = []

for patient in SOURCE.iterdir():

    if not patient.is_dir():
        continue

    for img in (patient / "0").glob("*.png"):
        benign.append(img)

    for img in (patient / "1").glob("*.png"):
        malignant.append(img)

random.shuffle(benign)
random.shuffle(malignant)


def split_and_copy(images, classname):

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    for img in train:
        shutil.copy(img, OUTPUT / "train" / classname / img.name)

    for img in val:
        shutil.copy(img, OUTPUT / "val" / classname / img.name)

    for img in test:
        shutil.copy(img, OUTPUT / "test" / classname / img.name)

    print(classname)
    print("Train :", len(train))
    print("Val   :", len(val))
    print("Test  :", len(test))
    print()


split_and_copy(benign, "benign")
split_and_copy(malignant, "malignant")

print("=" * 50)
print("Dataset Split Completed")
print("=" * 50)