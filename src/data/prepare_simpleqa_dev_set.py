from pathlib import Path
import csv
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "simpleqa"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simpleqa_verified.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simpleqa_verified_dev50.csv"
)

DEV_SIZE = 50
SEED = 514

REQUIRED_COLUMNS = {
    "original_index",
    "problem",
    "answer",
    "topic",
    "answer_type",
    "multi_step",
    "requires_reasoning",
    "urls",
}


def load_dataset():
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = list(csv.DictReader(file))

    if not records:
        raise ValueError("SimpleQA Verified dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(records[0].keys())

    if missing_columns:
        raise ValueError(
            f"SimpleQA Verified is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return records


def select_questions(records):
    rng = random.Random(SEED)

    indices = rng.sample(
        range(len(records)),
        DEV_SIZE,
    )

    selected = []

    for dataset_index in indices:
        record = records[dataset_index]

        reference_answer = record["answer"].strip()

        selected.append(
            {
                "dataset_name": DATASET_NAME,
                "dataset_index": dataset_index,
                "question": record["problem"].strip(),
                "reference_answer": reference_answer,
                "acceptable_answers": json.dumps(
                    [reference_answer],
                    ensure_ascii=False,
                ),
                "original_index": record["original_index"],
                "topic": record["topic"],
                "answer_type": record["answer_type"],
                "multi_step": record["multi_step"],
                "requires_reasoning": record["requires_reasoning"],
                "urls": record["urls"],
            }
        )

    return selected


def save_dataset(records):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset_name",
        "dataset_index",
        "question",
        "reference_answer",
        "acceptable_answers",
        "original_index",
        "topic",
        "answer_type",
        "multi_step",
        "requires_reasoning",
        "urls",
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
        writer.writerows(records)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "SimpleQA Verified dataset was not found."
        )

    records = load_dataset()

    if DEV_SIZE > len(records):
        raise ValueError(
            "Development size is larger than the dataset."
        )

    selected = select_questions(records)
    save_dataset(selected)

    print("SimpleQA Verified development set prepared")
    print("-" * 45)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Dataset records: {len(records):,}")
    print(f"Development questions: {len(selected)}")
    print(f"Random seed: {SEED}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()