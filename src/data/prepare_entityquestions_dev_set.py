from pathlib import Path
import csv
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "entityquestions"

DEV_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "entityquestions"
    / "dataset"
    / "dev"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "entityquestions_dev50.csv"
)

DEV_SIZE = 50
SEED = 514


def load_dataset():
    records = []

    files = sorted(DEV_DIR.glob("P*.dev.json"))

    if not files:
        raise FileNotFoundError(
            "No EntityQuestions development files were found."
        )

    for file_path in files:
        relation = file_path.name.split(".")[0]

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as file:
            items = json.load(file)

        for source_index, item in enumerate(items):
            question = item.get("question")
            answers = item.get("answers")

            if not question or not answers:
                continue

            records.append(
                {
                    "question": question,
                    "answers": answers,
                    "relation": relation,
                    "source_file": file_path.name,
                    "source_index": source_index,
                }
            )

    if not records:
        raise ValueError(
            "EntityQuestions development data is empty."
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

        acceptable_answers = [
            str(answer).strip()
            for answer in record["answers"]
            if str(answer).strip()
        ]

        if not acceptable_answers:
            continue

        selected.append(
            {
                "dataset_name": DATASET_NAME,
                "dataset_index": dataset_index,
                "question": record["question"].strip(),
                "reference_answer": acceptable_answers[0],
                "acceptable_answers": json.dumps(
                    acceptable_answers,
                    ensure_ascii=False,
                ),
                "relation": record["relation"],
                "source_file": record["source_file"],
                "source_index": record["source_index"],
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
        "relation",
        "source_file",
        "source_index",
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
    if not DEV_DIR.exists():
        raise FileNotFoundError(
            "EntityQuestions development directory was not found. "
            "Run download_entityquestions.py first."
        )

    records = load_dataset()

    if DEV_SIZE > len(records):
        raise ValueError(
            "Development size is larger than the dataset."
        )

    selected = select_questions(records)

    if len(selected) != DEV_SIZE:
        raise ValueError(
            "The expected number of development questions "
            "was not selected."
        )

    save_dataset(selected)

    print("EntityQuestions development set prepared")
    print("-" * 45)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Dataset records: {len(records):,}")
    print(f"Development questions: {len(selected)}")
    print(f"Random seed: {SEED}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()