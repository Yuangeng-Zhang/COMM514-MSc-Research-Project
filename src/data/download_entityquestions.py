from pathlib import Path
from urllib.request import urlretrieve
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "https://nlp.cs.princeton.edu/"
    "projects/entity-questions/dataset.zip"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "entityquestions"
)

ZIP_PATH = RAW_DIR / "dataset.zip"


def main():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ZIP_PATH.exists():
        print("Downloading EntityQuestions...")
        urlretrieve(DATA_URL, ZIP_PATH)
    else:
        print("EntityQuestions archive already exists.")

    dev_dir = RAW_DIR / "dataset" / "dev"

    if not dev_dir.exists():
        print("Extracting EntityQuestions...")

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(RAW_DIR)
    else:
        print("EntityQuestions dataset already extracted.")

    if not dev_dir.exists():
        raise FileNotFoundError(
            "EntityQuestions development split was not found "
            "after extraction."
        )

    print("EntityQuestions dataset ready")
    print("-" * 45)
    print(f"Saved to: {RAW_DIR}")


if __name__ == "__main__":
    main()