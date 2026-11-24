from pathlib import Path

SOURCE_DIR = Path("IDC_regular_ps50_idx5")

benign = []
malignant = []

for patient in SOURCE_DIR.iterdir():

    if not patient.is_dir():
        continue

    benign_folder = patient / "0"
    malignant_folder = patient / "1"

    if benign_folder.exists():
        benign.extend(list(benign_folder.glob("*.png")))

    if malignant_folder.exists():
        malignant.extend(list(malignant_folder.glob("*.png")))

print("=" * 60)

print("Patients :", len([x for x in SOURCE_DIR.iterdir() if x.is_dir()]))

print("Benign Images :", len(benign))
print("Malignant Images :", len(malignant))
print("Total Images :", len(benign) + len(malignant))

print("=" * 60)