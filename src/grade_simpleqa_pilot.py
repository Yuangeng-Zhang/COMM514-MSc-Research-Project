from pathlib import Path
from datetime import datetime, timezone
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simpleqa_verified.csv"
)

GENERATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_gemma4_generations.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_ground_truth.jsonl"
)


def load_dataset():
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def load_generations():
    records = []

    with open(
        GENERATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_completed():
    if not OUTPUT_PATH.exists():
        return set()

    completed = set()

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                completed.add(record["dataset_index"])

    return completed


def save_grade(item, dataset_record, label, note):
    result = {
        "dataset_index": item["dataset_index"],
        "original_index": item["original_index"],
        "problem": item["problem"],
        "reference_answer": item["reference_answer"],
        "generated_answer": item["generated_answer"],
        "generator_model": item["generator_model"],
        "ground_truth_label": label,
        "grading_method": "manual_reference_check",
        "grading_note": note,
        "supporting_urls": dataset_record["urls"],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        OUTPUT_PATH,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "SimpleQA Verified dataset was not found."
        )

    if not GENERATION_PATH.exists():
        raise FileNotFoundError(
            "Generation pilot results were not found."
        )

    dataset = load_dataset()
    generations = load_generations()
    completed = load_completed()

    remaining = [
        item
        for item in generations
        if item["dataset_index"] not in completed
    ]

    print("SimpleQA pilot manual grading")
    print("-" * 50)
    print(f"Generated answers: {len(generations)}")
    print(f"Already graded: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print()
    print("Labels:")
    print("  C = CORRECT")
    print("  I = INCORRECT")
    print("  N = NOT_ATTEMPTED")
    print("  S = SKIP for later review")
    print("  Q = QUIT")
    print()

    for number, item in enumerate(
        remaining,
        start=1,
    ):
        index = item["dataset_index"]
        dataset_record = dataset[index]

        print("=" * 70)
        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index={index}"
        )

        print("\nQuestion:")
        print(item["problem"])

        print("\nGold answer:")
        print(item["reference_answer"])

        print("\nGenerated answer:")
        print(item["generated_answer"])

        print("\nAnswer type:")
        print(dataset_record["answer_type"])

        while True:
            choice = input(
                "\nGrade [C/I/N/S/Q]: "
            ).strip().upper()

            if choice in {
                "C",
                "I",
                "N",
                "S",
                "Q",
            }:
                break

            print(
                "Please enter C, I, N, S, or Q."
            )

        if choice == "Q":
            print("\nGrading stopped.")
            return

        if choice == "S":
            print("Skipped for later review.")
            continue

        label_map = {
            "C": "CORRECT",
            "I": "INCORRECT",
            "N": "NOT_ATTEMPTED",
        }

        label = label_map[choice]

        note = input(
            "Optional note "
            "(press Enter to leave blank): "
        ).strip()

        save_grade(
            item,
            dataset_record,
            label,
            note,
        )

        print(f"Saved: {label}")

    print()
    print("Manual grading completed.")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()