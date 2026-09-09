from pathlib import Path
import csv
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "simpleqa_verified.csv"


def shorten(text, limit=180):
    text = str(text).replace("\n", " ").strip()

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "SimpleQA Verified dataset was not found. "
            "Run download_simpleqa_verified.py first."
        )

    with open(DATA_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames
        records = list(reader)

    print("SimpleQA Verified dataset inspection")
    print("-" * 45)

    print(f"File: {DATA_PATH}")
    print(f"Number of records: {len(records):,}")
    print(f"Number of fields: {len(fieldnames)}")
    print(f"Fields: {fieldnames}")

    print("\nEmpty values by field:")

    for field in fieldnames:
        empty_count = sum(
            not record.get(field, "").strip()
            for record in records
        )

        print(
            f"  {field}: "
            f"{empty_count:,}"
        )

    if "problem" in fieldnames:
        normalized_questions = [
            " ".join(record["problem"].split()).casefold()
            for record in records
        ]

        counts = Counter(normalized_questions)

        duplicate_groups = sum(
            count > 1
            for count in counts.values()
        )

        repeated_records = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        print("\nDuplicate problems:")
        print(
            f"  Duplicate groups: "
            f"{duplicate_groups:,}"
        )
        print(
            f"  Additional repeated records: "
            f"{repeated_records:,}"
        )

    print("\nFirst record:")

    first = records[0]

    for field in fieldnames:
        print(f"\n{field}:")
        print(shorten(first[field]))

    print("\nInspection completed.")


if __name__ == "__main__":
    main()