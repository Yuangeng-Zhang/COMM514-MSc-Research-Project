from pathlib import Path
from collections import Counter
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANUAL_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_ground_truth.jsonl"
)

AUTO_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_pilot_auto_grades_v2.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_pilot_grade_comparison_v2.jsonl"
)

VALID_LABELS = {
    "CORRECT",
    "INCORRECT",
    "NOT_ATTEMPTED",
}


def load_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def latest_valid_auto_grades(records):
    latest = {}

    for record in records:
        label = record.get("ground_truth_label")

        if label in VALID_LABELS:
            latest[record["dataset_index"]] = record

    return latest


def manual_grades_by_index(records):
    return {
        record["dataset_index"]: record
        for record in records
    }


def save_comparison(records):
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    if not MANUAL_PATH.exists():
        raise FileNotFoundError(
            "Manual grading results were not found."
        )

    if not AUTO_PATH.exists():
        raise FileNotFoundError(
            "Automatic grading results were not found."
        )

    manual_records = load_jsonl(MANUAL_PATH)
    auto_records = load_jsonl(AUTO_PATH)

    manual = manual_grades_by_index(manual_records)
    auto = latest_valid_auto_grades(auto_records)

    shared_indices = sorted(
        set(manual) & set(auto)
    )

    comparisons = []

    for index in shared_indices:
        manual_record = manual[index]
        auto_record = auto[index]

        manual_label = manual_record["ground_truth_label"]
        auto_label = auto_record["ground_truth_label"]

        comparisons.append(
            {
                "dataset_index": index,
                "original_index":
                    manual_record["original_index"],
                "problem":
                    manual_record["problem"],
                "reference_answer":
                    manual_record["reference_answer"],
                "generated_answer":
                    manual_record["generated_answer"],
                "manual_label":
                    manual_label,
                "automatic_label":
                    auto_label,
                "agreement":
                    manual_label == auto_label,
                "grader_model":
                    auto_record["grader_model"],
                "prompt_version":
                    auto_record["prompt_version"],
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_comparison(comparisons)

    agreements = sum(
        record["agreement"]
        for record in comparisons
    )

    total = len(comparisons)

    pair_counts = Counter(
        (
            record["manual_label"],
            record["automatic_label"],
        )
        for record in comparisons
    )

    print("SimpleQA grading comparison")
    print("-" * 45)
    print(f"Manual records: {len(manual)}")
    print(f"Automatic valid records: {len(auto)}")
    print(f"Matched records: {total}")
    print()

    if total:
        print(
            f"Agreement: {agreements}/{total} "
            f"({agreements / total:.1%})"
        )

    print()
    print("Label pairs:")

    for (
        manual_label,
        auto_label,
    ), count in sorted(pair_counts.items()):
        print(
            f"  {manual_label:14} -> "
            f"{auto_label:14} : {count}"
        )

    disagreements = [
        record
        for record in comparisons
        if not record["agreement"]
    ]

    print()
    print(
        f"Disagreements: {len(disagreements)}"
    )

    for record in disagreements:
        print("=" * 70)
        print(
            f"dataset_index="
            f"{record['dataset_index']}"
        )
        print(
            f"manual="
            f"{record['manual_label']}"
        )
        print(
            f"automatic="
            f"{record['automatic_label']}"
        )
        print("\nQuestion:")
        print(record["problem"])
        print("\nReference answer:")
        print(record["reference_answer"])
        print("\nGenerated answer:")
        print(record["generated_answer"])

    print()
    print(f"Comparison saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()