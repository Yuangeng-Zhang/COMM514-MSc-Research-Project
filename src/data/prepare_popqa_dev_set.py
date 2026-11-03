from pathlib import Path
import ast
import csv
import json
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "popqa"

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

REQUIRED_COLUMNS = {
    "id",
    "question",
    "obj",
    "possible_answers",
    "prop",
    "s_pop",
    "o_pop",
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
        raise ValueError("PopQA dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(records[0].keys())

    if missing_columns:
        raise ValueError(
            f"PopQA is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return records


def parse_possible_answers(value, reference_answer):
    answers = []

    if value:
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                answers = [
                    str(answer).strip()
                    for answer in parsed
                    if str(answer).strip()
                ]
            elif parsed:
                answers = [str(parsed).strip()]

        except (ValueError, SyntaxError):
            answers = [value.strip()]

    reference_answer = reference_answer.strip()

    if reference_answer and reference_answer not in answers:
        answers.insert(0, reference_answer)

    return answers


def select_questions(records):
    rng = random.Random(SEED)

    indices = rng.sample(
        range(len(records)),
        DEV_SIZE,
    )

    selected = []

    for dataset_index in indices:
        record = records[dataset_index]

        reference_answer = record["obj"].strip()

        acceptable_answers = parse_possible_answers(
            record["possible_answers"],
            reference_answer,
        )

        selected.append(
            {
                "dataset_name": DATASET_NAME,
                "dataset_index": dataset_index,
                "question": record["question"].strip(),
                "reference_answer": reference_answer,
                "acceptable_answers": json.dumps(
                    acceptable_answers,
                    ensure_ascii=False,
                ),
                "id": record["id"],
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
        "dataset_name",
        "dataset_index",
        "question",
        "reference_answer",
        "acceptable_answers",
        "id",
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
    print(f"Dataset: {DATASET_NAME}")
    print(f"Dataset records: {len(records):,}")
    print(f"Development questions: {len(selected)}")
    print(f"Random seed: {SEED}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()