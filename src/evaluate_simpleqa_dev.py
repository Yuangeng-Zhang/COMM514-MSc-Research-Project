from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_gemma4_auto_grades.jsonl"
)

VERIFICATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_verifications.jsonl"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_evaluation_summary.json"
)

CASE_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_evaluation_cases.jsonl"
)

VALID_GROUND_TRUTH = {
    "CORRECT",
    "INCORRECT",
    "NOT_ATTEMPTED",
}

VALID_PREDICTIONS = {
    "CORRECT",
    "INCORRECT",
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


def load_ground_truth():
    records = load_jsonl(GROUND_TRUTH_PATH)

    return {
        int(record["dataset_index"]): record
        for record in records
        if record["ground_truth_label"]
        in VALID_GROUND_TRUTH
    }


def latest_valid_verifications():
    records = load_jsonl(VERIFICATION_PATH)
    latest = {}

    for record in records:
        if record["prediction"] not in VALID_PREDICTIONS:
            continue

        key = (
            int(record["dataset_index"]),
            record["verification_type"],
        )

        latest[key] = record

    return latest


def calculate_metrics(cases):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for case in cases:
        actual = case["ground_truth_label"]
        predicted = case["prediction"]

        if actual == "INCORRECT":
            if predicted == "INCORRECT":
                tp += 1
            else:
                fn += 1

        elif actual == "CORRECT":
            if predicted == "CORRECT":
                tn += 1
            else:
                fp += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total
        else None
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else None
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else None
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (
            precision is not None
            and recall is not None
            and precision + recall > 0
        )
        else None
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else None
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def format_metric(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


def main():
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            "Development ground-truth results were not found."
        )

    if not VERIFICATION_PATH.exists():
        raise FileNotFoundError(
            "Development verification results were not found."
        )

    ground_truth = load_ground_truth()
    verifications = latest_valid_verifications()

    label_counts = {
        "CORRECT": 0,
        "INCORRECT": 0,
        "NOT_ATTEMPTED": 0,
    }

    for record in ground_truth.values():
        label_counts[
            record["ground_truth_label"]
        ] += 1

    cases = []

    for dataset_index, truth_record in ground_truth.items():
        ground_truth_label = truth_record["ground_truth_label"]

        if ground_truth_label == "NOT_ATTEMPTED":
            continue

        for verification_type in ("self", "cross"):
            key = (
                dataset_index,
                verification_type,
            )

            if key not in verifications:
                continue

            verification = verifications[key]

            cases.append(
                {
                    "dataset_index": dataset_index,
                    "original_index":
                        truth_record["original_index"],
                    "problem":
                        truth_record["problem"],
                    "reference_answer":
                        truth_record["reference_answer"],
                    "generated_answer":
                        truth_record["generated_answer"],
                    "ground_truth_label":
                        ground_truth_label,
                    "verification_type":
                        verification_type,
                    "verifier_model":
                        verification["verifier_model"],
                    "prediction":
                        verification["prediction"],
                    "verification_correct":
                        verification["prediction"]
                        == ground_truth_label,
                }
            )

    self_cases = [
        case
        for case in cases
        if case["verification_type"] == "self"
    ]

    cross_cases = [
        case
        for case in cases
        if case["verification_type"] == "cross"
    ]

    self_metrics = calculate_metrics(self_cases)
    cross_metrics = calculate_metrics(cross_cases)

    summary = {
        "development_questions": len(ground_truth),
        "ground_truth_distribution": label_counts,
        "attempted_answers":
            label_counts["CORRECT"]
            + label_counts["INCORRECT"],
        "self_cases": len(self_cases),
        "cross_cases": len(cross_cases),
        "self_metrics": self_metrics,
        "cross_metrics": cross_metrics,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        CASE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for case in cases:
            file.write(
                json.dumps(
                    case,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("SimpleQA development evaluation")
    print("-" * 50)
    print(
        f"Development questions: "
        f"{len(ground_truth)}"
    )
    print(
        f"Correct: "
        f"{label_counts['CORRECT']}"
    )
    print(
        f"Incorrect: "
        f"{label_counts['INCORRECT']}"
    )
    print(
        f"Not attempted: "
        f"{label_counts['NOT_ATTEMPTED']}"
    )
    print(
        f"Attempted answers: "
        f"{summary['attempted_answers']}"
    )
    print()

    print("Self-verification")
    print(
        f"  Cases: "
        f"{len(self_cases)}"
    )
    print(
        f"  TP={self_metrics['tp']} "
        f"TN={self_metrics['tn']} "
        f"FP={self_metrics['fp']} "
        f"FN={self_metrics['fn']}"
    )
    print(
        f"  Accuracy: "
        f"{format_metric(self_metrics['accuracy'])}"
    )
    print(
        f"  Precision: "
        f"{format_metric(self_metrics['precision'])}"
    )
    print(
        f"  Recall: "
        f"{format_metric(self_metrics['recall'])}"
    )
    print(
        f"  F1: "
        f"{format_metric(self_metrics['f1'])}"
    )
    print(
        f"  False-positive rate: "
        f"{format_metric(self_metrics['false_positive_rate'])}"
    )
    print()

    print("Cross-verification")
    print(
        f"  Cases: "
        f"{len(cross_cases)}"
    )
    print(
        f"  TP={cross_metrics['tp']} "
        f"TN={cross_metrics['tn']} "
        f"FP={cross_metrics['fp']} "
        f"FN={cross_metrics['fn']}"
    )
    print(
        f"  Accuracy: "
        f"{format_metric(cross_metrics['accuracy'])}"
    )
    print(
        f"  Precision: "
        f"{format_metric(cross_metrics['precision'])}"
    )
    print(
        f"  Recall: "
        f"{format_metric(cross_metrics['recall'])}"
    )
    print(
        f"  F1: "
        f"{format_metric(cross_metrics['f1'])}"
    )
    print(
        f"  False-positive rate: "
        f"{format_metric(cross_metrics['false_positive_rate'])}"
    )
    print()

    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Case results saved to: {CASE_PATH}")


if __name__ == "__main__":
    main()