from pathlib import Path
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_shared_error_cases.jsonl"
)

DEV_SET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simpleqa_verified_dev50.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_shared_error_coding.csv"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_questions():
    with open(
        DEV_SET_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return {
        int(row["dataset_index"]): row["problem"]
        for row in rows
    }


def main():
    records = load_jsonl(INPUT_PATH)
    questions = load_questions()

    shared_cases = [
        record
        for record in records
        if record.get("group") == "shared_error"
    ]

    fieldnames = [
        "dataset_index",
        "question",
        "reference_answer",
        "gemma_answer",
        "gptoss_answer",
        "cross_prediction",
        "shared_error_type",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in shared_cases:
            dataset_index = int(record["dataset_index"])

            writer.writerow(
                {
                    "dataset_index": dataset_index,
                    "question": questions[dataset_index],
                    "reference_answer":
                        record["reference_answer"],
                    "gemma_answer":
                        record["gemma_answer"],
                    "gptoss_answer":
                        record["gptoss_answer"],
                    "cross_prediction":
                        record["cross_prediction"],
                    "shared_error_type": "",
                }
            )

    print("Shared-error coding file prepared")
    print("-" * 45)
    print(f"Cases exported: {len(shared_cases)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()