from pathlib import Path
from collections import Counter
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_ground_truth.jsonl"
)

VERIFICATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_verifications.jsonl"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_evaluation_summary.json"
)

CASE_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_case_analysis.csv"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def calculate_metrics(cases, prediction_key):
    valid_cases = [
        case
        for case in cases
        if case["ground_truth_label"]
        in {"CORRECT", "INCORRECT"}
        and case[prediction_key]
        in {"CORRECT", "INCORRECT"}
    ]

    tp = sum(
        case["ground_truth_label"] == "INCORRECT"
        and case[prediction_key] == "INCORRECT"
        for case in valid_cases
    )

    tn = sum(
        case["ground_truth_label"] == "CORRECT"
        and case[prediction_key] == "CORRECT"
        for case in valid_cases
    )

    fp = sum(
        case["ground_truth_label"] == "CORRECT"
        and case[prediction_key] == "INCORRECT"
        for case in valid_cases
    )

    fn = sum(
        case["ground_truth_label"] == "INCORRECT"
        and case[prediction_key] == "CORRECT"
        for case in valid_cases
    )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "evaluated_cases": len(valid_cases),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "Ground-truth file was not found."
        )

    if not VERIFICATION_PATH.exists():
        raise FileNotFoundError(
            "Verification results were not found."
        )

    ground_truth = load_jsonl(GROUND_TRUTH_PATH)
    verifications = load_jsonl(VERIFICATION_PATH)

    print("SimpleQA pilot evaluation")
    print("-" * 50)
    print(f"Ground-truth records: {len(ground_truth)}")
    print(f"Verification records: {len(verifications)}")

    if len(ground_truth) != 20:
        print(
            "WARNING: Expected 20 ground-truth records."
        )

    verification_lookup = {}

    for record in verifications:
        key = (
            record["dataset_index"],
            record["verification_type"],
        )

        verification_lookup[key] = record

    cases = []

    for item in ground_truth:
        index = item["dataset_index"]

        self_record = verification_lookup.get(
            (index, "self")
        )

        cross_record = verification_lookup.get(
            (index, "cross")
        )

        if self_record is None or cross_record is None:
            print(
                f"WARNING: Missing verification "
                f"for dataset_index={index}"
            )
            continue

        cases.append(
            {
                "dataset_index": index,
                "original_index":
                    item["original_index"],
                "problem":
                    item["problem"],
                "reference_answer":
                    item["reference_answer"],
                "generated_answer":
                    item["generated_answer"],
                "ground_truth_label":
                    item["ground_truth_label"],
                "self_prediction":
                    self_record["prediction"],
                "cross_prediction":
                    cross_record["prediction"],
                "self_model":
                    self_record["verifier_model"],
                "cross_model":
                    cross_record["verifier_model"],
            }
        )

    ground_truth_counts = Counter(
        case["ground_truth_label"]
        for case in cases
    )

    print("\nGround-truth distribution:")

    for label in [
        "CORRECT",
        "INCORRECT",
        "NOT_ATTEMPTED",
    ]:
        print(
            f"  {label}: "
            f"{ground_truth_counts.get(label, 0)}"
        )

    self_metrics = calculate_metrics(
        cases,
        "self_prediction",
    )

    cross_metrics = calculate_metrics(
        cases,
        "cross_prediction",
    )

    print("\nSelf-verification:")
    print(
        f"  Evaluated cases: "
        f"{self_metrics['evaluated_cases']}"
    )
    print(
        f"  TP={self_metrics['true_positive']}, "
        f"TN={self_metrics['true_negative']}, "
        f"FP={self_metrics['false_positive']}, "
        f"FN={self_metrics['false_negative']}"
    )
    print(
        f"  Accuracy: "
        f"{self_metrics['accuracy']:.3f}"
    )
    print(
        f"  Precision: "
        f"{self_metrics['precision']:.3f}"
    )
    print(
        f"  Recall: "
        f"{self_metrics['recall']:.3f}"
    )
    print(
        f"  F1: "
        f"{self_metrics['f1']:.3f}"
    )

    print("\nCross-model verification:")
    print(
        f"  Evaluated cases: "
        f"{cross_metrics['evaluated_cases']}"
    )
    print(
        f"  TP={cross_metrics['true_positive']}, "
        f"TN={cross_metrics['true_negative']}, "
        f"FP={cross_metrics['false_positive']}, "
        f"FN={cross_metrics['false_negative']}"
    )
    print(
        f"  Accuracy: "
        f"{cross_metrics['accuracy']:.3f}"
    )
    print(
        f"  Precision: "
        f"{cross_metrics['precision']:.3f}"
    )
    print(
        f"  Recall: "
        f"{cross_metrics['recall']:.3f}"
    )
    print(
        f"  F1: "
        f"{cross_metrics['f1']:.3f}"
    )

    comparable_cases = [
        case
        for case in cases
        if case["ground_truth_label"]
        in {"CORRECT", "INCORRECT"}
        and case["self_prediction"]
        in {"CORRECT", "INCORRECT"}
        and case["cross_prediction"]
        in {"CORRECT", "INCORRECT"}
    ]

    both_correct = 0
    self_only_correct = 0
    cross_only_correct = 0
    both_wrong = 0

    for case in comparable_cases:
        truth = case["ground_truth_label"]

        self_is_correct = (
            case["self_prediction"] == truth
        )

        cross_is_correct = (
            case["cross_prediction"] == truth
        )

        if self_is_correct and cross_is_correct:
            both_correct += 1

        elif self_is_correct:
            self_only_correct += 1

        elif cross_is_correct:
            cross_only_correct += 1

        else:
            both_wrong += 1

    print("\nPaired comparison:")
    print(f"  Both correct: {both_correct}")
    print(
        f"  Self only correct: "
        f"{self_only_correct}"
    )
    print(
        f"  Cross only correct: "
        f"{cross_only_correct}"
    )
    print(f"  Both wrong: {both_wrong}")

    disagreements = [
        case
        for case in comparable_cases
        if (
            case["self_prediction"]
            != case["cross_prediction"]
        )
    ]

    print(
        f"\nSelf/cross disagreements: "
        f"{len(disagreements)}"
    )

    for case in disagreements:
        print()
        print(
            f"dataset_index="
            f"{case['dataset_index']}"
        )
        print(
            f"  Ground truth: "
            f"{case['ground_truth_label']}"
        )
        print(
            f"  Self: "
            f"{case['self_prediction']}"
        )
        print(
            f"  Cross: "
            f"{case['cross_prediction']}"
        )
        print(
            f"  Answer: "
            f"{case['generated_answer']}"
        )

    summary = {
        "total_cases": len(cases),
        "ground_truth_distribution":
            dict(ground_truth_counts),
        "self_verification":
            self_metrics,
        "cross_verification":
            cross_metrics,
        "paired_comparison": {
            "both_correct": both_correct,
            "self_only_correct":
                self_only_correct,
            "cross_only_correct":
                cross_only_correct,
            "both_wrong":
                both_wrong,
            "disagreements":
                len(disagreements),
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    fieldnames = [
        "dataset_index",
        "original_index",
        "problem",
        "reference_answer",
        "generated_answer",
        "ground_truth_label",
        "self_prediction",
        "cross_prediction",
        "self_model",
        "cross_model",
    ]

    with open(
        CASE_PATH,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(cases)

    print()
    print("Evaluation completed.")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Case analysis: {CASE_PATH}")


if __name__ == "__main__":
    main()