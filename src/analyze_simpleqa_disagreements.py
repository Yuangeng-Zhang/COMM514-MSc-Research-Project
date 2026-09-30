from pathlib import Path
from collections import Counter
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASE_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_evaluation_cases.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_disagreements.jsonl"
)


def load_cases():
    records = []

    with open(
        CASE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def pair_cases(records):
    paired = {}

    for record in records:
        index = int(record["dataset_index"])

        if index not in paired:
            paired[index] = {}

        paired[index][
            record["verification_type"]
        ] = record

    return paired


def classify_pair(self_case, cross_case):
    self_correct = self_case["verification_correct"]
    cross_correct = cross_case["verification_correct"]

    if self_correct and cross_correct:
        return "both_correct"

    if self_correct and not cross_correct:
        return "self_only_correct"

    if not self_correct and cross_correct:
        return "cross_only_correct"

    return "both_wrong"


def save_disagreements(records):
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
    if not CASE_PATH.exists():
        raise FileNotFoundError(
            "Evaluation case results were not found."
        )

    cases = load_cases()
    paired = pair_cases(cases)

    categories = Counter()
    disagreements = []

    for dataset_index, pair in paired.items():
        if "self" not in pair or "cross" not in pair:
            continue

        self_case = pair["self"]
        cross_case = pair["cross"]

        category = classify_pair(
            self_case,
            cross_case,
        )

        categories[category] += 1

        if self_case["prediction"] != cross_case["prediction"]:
            disagreements.append(
                {
                    "dataset_index": dataset_index,
                    "problem":
                        self_case["problem"],
                    "reference_answer":
                        self_case["reference_answer"],
                    "generated_answer":
                        self_case["generated_answer"],
                    "ground_truth_label":
                        self_case["ground_truth_label"],
                    "self_prediction":
                        self_case["prediction"],
                    "cross_prediction":
                        cross_case["prediction"],
                    "outcome":
                        category,
                }
            )

    save_disagreements(disagreements)

    print("SimpleQA self vs cross disagreement analysis")
    print("-" * 50)
    print(f"Paired attempted answers: {len(paired)}")
    print()

    print("Verification outcomes:")
    print(
        f"  Both correct: "
        f"{categories['both_correct']}"
    )
    print(
        f"  Self only correct: "
        f"{categories['self_only_correct']}"
    )
    print(
        f"  Cross only correct: "
        f"{categories['cross_only_correct']}"
    )
    print(
        f"  Both wrong: "
        f"{categories['both_wrong']}"
    )

    print()
    print(
        f"Different predictions: "
        f"{len(disagreements)}"
    )

    print()
    print("Disagreement cases:")

    for record in disagreements:
        print("=" * 70)
        print(
            f"dataset_index="
            f"{record['dataset_index']}"
        )
        print(
            f"ground_truth="
            f"{record['ground_truth_label']}"
        )
        print(
            f"self="
            f"{record['self_prediction']}"
        )
        print(
            f"cross="
            f"{record['cross_prediction']}"
        )
        print(
            f"outcome="
            f"{record['outcome']}"
        )
        print("\nQuestion:")
        print(record["problem"])
        print("\nGenerated answer:")
        print(record["generated_answer"])
        print("\nReference answer:")
        print(record["reference_answer"])

    print()
    print(
        f"Disagreement cases saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()