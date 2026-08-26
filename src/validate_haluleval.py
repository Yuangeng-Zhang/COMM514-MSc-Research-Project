from pathlib import Path
from collections import Counter
from statistics import median
import json

from inspect_haluleval import load_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qa_data.json"

FIELDS = [
    "knowledge",
    "question",
    "right_answer",
    "hallucinated_answer",
]


def normalize_text(text):
    return " ".join(str(text).split()).casefold()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "HaluEval QA dataset was not found. "
            "Run download_haluleval.py first."
        )

    records, _ = load_records(DATA_PATH)

    print("HaluEval QA dataset validation")
    print("-" * 40)
    print(f"Number of records: {len(records):,}")

    # 1. Check for empty or non-string values
    print("\nEmpty or non-string values:")
    for field in FIELDS:
        non_string = sum(
            not isinstance(record.get(field), str)
            for record in records
        )

        empty = sum(
            isinstance(record.get(field), str)
            and not record[field].strip()
            for record in records
        )

        print(
            f"  {field}: "
            f"{empty:,} empty, {non_string:,} non-string"
        )

    # 2. Check for duplicate questions
    normalized_questions = [
        normalize_text(record["question"])
        for record in records
    ]

    question_counts = Counter(normalized_questions)

    duplicate_question_groups = sum(
        count > 1 for count in question_counts.values()
    )

    extra_duplicate_questions = sum(
        count - 1
        for count in question_counts.values()
        if count > 1
    )

    print("\nDuplicate questions:")
    print(
        f"  Duplicate question groups: "
        f"{duplicate_question_groups:,}"
    )
    print(
        f"  Additional repeated records: "
        f"{extra_duplicate_questions:,}"
    )

    # 3. Check for exact duplicate records
    serialized_records = [
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
        )
        for record in records
    ]

    exact_record_counts = Counter(serialized_records)

    exact_duplicates = sum(
        count - 1
        for count in exact_record_counts.values()
        if count > 1
    )

    print("\nExact duplicate records:")
    print(f"  Additional duplicates: {exact_duplicates:,}")

    # 4. Check whether right and hallucinated answers are identical
    same_answer_indices = []

    for index, record in enumerate(records):
        right_answer = normalize_text(record["right_answer"])
        hallucinated_answer = normalize_text(
            record["hallucinated_answer"]
        )

        if right_answer == hallucinated_answer:
            same_answer_indices.append(index)

    print("\nRight answer vs hallucinated answer:")
    print(
        f"  Identical after normalisation: "
        f"{len(same_answer_indices):,}"
    )

    if same_answer_indices:
        print(
            "  First affected record indices: "
            f"{same_answer_indices[:10]}"
        )

    # 5. Basic field-length summary
    print("\nCharacter-length summary:")

    for field in FIELDS:
        lengths = [
            len(record[field])
            for record in records
            if isinstance(record[field], str)
        ]

        print(
            f"  {field}: "
            f"min={min(lengths):,}, "
            f"median={median(lengths):,.1f}, "
            f"max={max(lengths):,}"
        )

    print("\nValidation completed.")


if __name__ == "__main__":
    main()