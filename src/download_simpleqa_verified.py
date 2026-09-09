from pathlib import Path
from urllib.request import urlretrieve
import hashlib


DATA_URL = (
    "https://huggingface.co/datasets/google/simpleqa-verified/"
    "resolve/main/simpleqa_verified.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simpleqa_verified.csv"
)


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print("Dataset already exists. Download skipped.")
    else:
        print("Downloading SimpleQA Verified...")
        urlretrieve(DATA_URL, OUTPUT_PATH)
        print("Download completed.")

    print(f"File: {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size:,} bytes")
    print(f"SHA256: {calculate_sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()