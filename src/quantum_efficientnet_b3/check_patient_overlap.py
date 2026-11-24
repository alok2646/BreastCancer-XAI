
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_patient_split"

splits = ("train", "val", "test")
patient_sets = {}
image_counts = {}

for split in splits:
    folder = DATASET / split

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    patients = set()
    count = 0
    unparsed = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        count += 1

        # Expected filename: 10256_idx5_x1001_y1001_class0.png
        parts = path.stem.split("_")

        if len(parts) >= 2 and parts[0].isdigit():
            patients.add(parts[0])
        else:
            unparsed.append(path.name)

    patient_sets[split] = patients
    image_counts[split] = count

    print(f"\n{split.upper()}")
    print("Images:", count)
    print("Unique patient IDs:", len(patients))
    print("Unparsed filenames:", len(unparsed))

    if unparsed:
        print("Examples:", unparsed[:5])

print("\n--- PATIENT ID OVERLAP ---")

has_overlap = False

for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = patient_sets[a] & patient_sets[b]
    print(f"{a} vs {b}: {len(overlap)} shared patient IDs")

    if overlap:
        has_overlap = True
        print("Examples:", sorted(overlap)[:10])

print("\nRESULT:", "OVERLAP FOUND" if has_overlap else "NO PARSED ID OVERLAP")
print("Note: Verify filename convention and patient ID semantics.")
