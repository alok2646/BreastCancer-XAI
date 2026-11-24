
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "dataset_patient_split"
CURRENT = ROOT / "dataset_base_final"
EXTENSIONS = {".png", ".jpg", ".jpeg"}

def files_in(folder):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            yield p

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Index the patient-split dataset without modifying it.
source_index = defaultdict(list)
source_count = 0

print("Indexing dataset_patient_split...")

for split in ("train", "val", "test"):
    for path in files_in(SOURCE / split):
        parts = path.stem.split("_")
        patient_id = parts[0] if parts and parts[0].isdigit() else None
        digest = sha256(path)

        source_index[digest].append({
            "split": split,
            "patient": patient_id,
            "class": path.parent.name,
        })
        source_count += 1

    print(f"Indexed {split}")

print("Source images indexed:", source_count)

# Compare current dataset files by exact file-content hash.
print("\nChecking dataset_base_final...")
matched = 0
total = 0
matched_patients = defaultdict(set)
matched_source_splits = defaultdict(int)
matched_classes = defaultdict(int)

for split in ("train", "val", "test"):
    for path in files_in(CURRENT / split):
        total += 1
        digest = sha256(path)
        records = source_index.get(digest, [])

        if records:
            matched += 1
            matched_source_splits[split] += 1

            for record in records:
                if record["patient"] is not None:
                    matched_patients[split].add(record["patient"])

                matched_classes[
                    (split, record["class"])
                ] += 1

    print(
        f"{split}: matched {matched_source_splits[split]} "
        "exact file hashes"
    )

print("\n--- SUMMARY ---")
print("Current dataset images:", total)
print("Exact hash matches:", matched)
print("No exact hash match:", total - matched)

for split in ("train", "val", "test"):
    print(
        f"{split}: source patient IDs represented in exact matches = "
        f"{len(matched_patients[split])}"
    )

print(
    "\nNote: No hash match does not prove images are unrelated; "
    "resizing, re-encoding, or augmentation changes file hashes."
)
