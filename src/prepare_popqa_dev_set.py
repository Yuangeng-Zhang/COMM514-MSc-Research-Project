from pathlib import Path
import csv
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "popqa.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "popqa_dev50.csv"
)

DEV_SIZE = 50
SEED = 514


def load_dataset():
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def select_questions(records):
    rng = random.Random(SEED)

    indices = rng.sample(
        range(len(records)),
        DEV_SIZE,
    )

    selected = []

    for dataset_index in indices:
        record = records[dataset_index]

        selected.append(
            {
                "dataset_index": dataset_index,
                "id": record["id"],
                "question": record["question"],
                "reference_answer": record["obj"],
                "possible_answers": record["possible_answers"],
                "property": record["prop"],
                "subject_popularity": record["s_pop"],
                "object_popularity": record["o_pop"],
            }
        )

    return selected


def save_dataset(records):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset_index",
        "id",
        "question",
        "reference_answer",
        "possible_answers",
        "property",
        "subject_popularity",
        "object_popularity",
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
            "PopQA dataset was not found."
        )

    records = load_dataset()

    if DEV_SIZE > len(records):
        raise ValueError(
            "Development size is larger than the dataset."
        )

    selected = select_questions(records)
    save_dataset(selected)

    print("PopQA development set prepared")
    print("-" * 45)
    print(f"Dataset records: {len(records):,}")
    print(f"Development questions: {len(selected)}")
    print(f"Random seed: {SEED}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()