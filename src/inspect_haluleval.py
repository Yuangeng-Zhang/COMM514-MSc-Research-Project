from pathlib import Path
import json
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qa_data.json"

EXPECTED_FIELDS = {
    "knowledge",
    "question",
    "right_answer",
    "hallucinated_answer",
}


def load_records(file_path):
    """
    Load the dataset while allowing for either:
    1. a standard JSON array, or
    2. JSON Lines format.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)

            if isinstance(data, list):
                return data, "JSON array"

            raise ValueError(
                f"Expected a list of records, but found {type(data).__name__}."
            )

        except json.JSONDecodeError:
            file.seek(0)

            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

            return records, "JSON Lines"


def shorten(text, limit=200):
    text = str(text).replace("\n", " ").strip()

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "HaluEval QA dataset was not found. "
            "Run download_haluleval.py first."
        )

    records, file_format = load_records(DATA_PATH)

    print("HaluEval QA dataset inspection")
    print("-" * 40)

    print(f"File: {DATA_PATH}")
    print(f"Detected format: {file_format}")
    print(f"Number of records: {len(records):,}")

    if not records:
        raise ValueError("The dataset contains no records.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Not all dataset records are JSON objects.")

    field_patterns = Counter(
        tuple(sorted(record.keys()))
        for record in records
    )

    print(f"Number of distinct field structures: {len(field_patterns)}")

    print("\nField structure(s):")
    for fields, count in field_patterns.items():
        print(f"  {count:,} records: {list(fields)}")

    missing_expected = sum(
        not EXPECTED_FIELDS.issubset(record.keys())
        for record in records
    )

    print(
        f"\nRecords missing expected fields: "
        f"{missing_expected:,}"
    )

    first = records[0]

    print("\nFirst record:")
    for key, value in first.items():
        print(f"\n{key}:")
        print(shorten(value))


if __name__ == "__main__":
    main()