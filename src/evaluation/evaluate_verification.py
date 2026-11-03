from pathlib import Path
import argparse
import csv
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "simpleqa": "SimpleQA",
    "popqa": "PopQA",
    "entityquestions": "EntityQuestions",
}

DEFAULT_GENERATOR_MODEL = "gemma4:cloud"
DEFAULT_CROSS_MODEL = "gpt-oss:20b-cloud"
DEFAULT_EVALUATOR_MODEL = "gpt-oss:20b-cloud"

REFERENCE_LABELS = {
    "CORRECT",
    "INCORRECT",
    "NOT_ATTEMPTED",
}

VERIFICATION_LABELS = {
    "CORRECT",
    "INCORRECT",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate self- and cross-verification "
            "against reference-based labels."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    parser.add_argument(
        "--generator-model",
        default=DEFAULT_GENERATOR_MODEL,
    )

    parser.add_argument(
        "--cross-model",
        default=DEFAULT_CROSS_MODEL,
    )

    parser.add_argument(
        "--evaluator-model",
        default=DEFAULT_EVALUATOR_MODEL,
    )

    return parser.parse_args()


def model_slug(model):
    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        model,
    ).strip("_").lower()


def get_paths(
    dataset,
    generator_model,
    evaluator_model,
):
    generator_slug = model_slug(
        generator_model
    )

    evaluator_slug = model_slug(
        evaluator_model
    )

    base_dir = (
        PROJECT_ROOT
        / "results"
        / "development"
        / dataset
    )

    verification_path = (
        base_dir
        / "verification"
        / (
            f"dev50_{generator_slug}"
            f"_verifications.jsonl"
        )
    )

    reference_path = (
        base_dir
        / "reference_evaluation"
        / (
            f"dev50_{generator_slug}"
            f"_evaluated_by_{evaluator_slug}.jsonl"
        )
    )

    metrics_dir = (
        base_dir
        / "metrics"
    )

    summary_path = (
        metrics_dir
        / (
            f"dev50_{generator_slug}"
            f"_verification_metrics.json"
        )
    )

    cases_path = (
        metrics_dir
        / (
            f"dev50_{generator_slug}"
            f"_evaluation_cases.csv"
        )
    )

    return (
        verification_path,
        reference_path,
        summary_path,
        cases_path,
    )


def load_jsonl(path):
    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None

    return numerator / denominator


def calculate_metrics(cases):
    tp = sum(
        1
        for case in cases
        if (
            case["reference_label"] == "INCORRECT"
            and case["prediction"] == "INCORRECT"
        )
    )

    fn = sum(
        1
        for case in cases
        if (
            case["reference_label"] == "INCORRECT"
            and case["prediction"] == "CORRECT"
        )
    )

    fp = sum(
        1
        for case in cases
        if (
            case["reference_label"] == "CORRECT"
            and case["prediction"] == "INCORRECT"
        )
    )

    tn = sum(
        1
        for case in cases
        if (
            case["reference_label"] == "CORRECT"
            and case["prediction"] == "CORRECT"
        )
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if (
        precision is None
        or recall is None
        or precision + recall == 0
    ):
        f1 = None
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn,
    )

    return {
        "cases": len(cases),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def load_reference_labels(records):
    labels = {}

    for record in records:
        label = record.get("label")

        if label not in REFERENCE_LABELS:
            continue

        index = int(
            record["dataset_index"]
        )

        labels[index] = {
            "question":
                record["question"],
            "reference_label":
                label,
        }

    return labels


def load_verification_predictions(
    records,
    generator_model,
    cross_model,
):
    predictions = {}

    for record in records:
        prediction = record.get(
            "prediction"
        )

        if prediction not in VERIFICATION_LABELS:
            continue

        verification_type = record.get(
            "verification_type"
        )

        verifier_model = record.get(
            "verifier_model"
        )

        if (
            verification_type == "self"
            and verifier_model != generator_model
        ):
            continue

        if (
            verification_type == "cross"
            and verifier_model != cross_model
        ):
            continue

        if verification_type not in {
            "self",
            "cross",
        }:
            continue

        key = (
            int(record["dataset_index"]),
            verification_type,
        )

        predictions[key] = prediction

    return predictions


def build_cases(
    reference_labels,
    predictions,
):
    rows = []

    for dataset_index in sorted(
        reference_labels
    ):
        reference = (
            reference_labels[
                dataset_index
            ]
        )

        rows.append(
            {
                "dataset_index":
                    dataset_index,
                "question":
                    reference["question"],
                "reference_label":
                    reference[
                        "reference_label"
                    ],
                "self_prediction":
                    predictions.get(
                        (
                            dataset_index,
                            "self",
                        ),
                        "",
                    ),
                "cross_prediction":
                    predictions.get(
                        (
                            dataset_index,
                            "cross",
                        ),
                        "",
                    ),
            }
        )

    return rows


def cases_for_mode(
    rows,
    verification_type,
):
    prediction_field = (
        f"{verification_type}_prediction"
    )

    return [
        {
            "reference_label":
                row["reference_label"],
            "prediction":
                row[prediction_field],
        }
        for row in rows
        if (
            row["reference_label"]
            in {"CORRECT", "INCORRECT"}
            and row[prediction_field]
            in VERIFICATION_LABELS
        )
    ]


def count_reference_labels(rows):
    return {
        label: sum(
            1
            for row in rows
            if row["reference_label"]
            == label
        )
        for label in sorted(
            REFERENCE_LABELS
        )
    }


def save_cases(rows, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "dataset_index",
        "question",
        "reference_label",
        "self_prediction",
        "cross_prediction",
    ]

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def format_metric(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


def main():
    args = parse_args()

    display_name = DATASETS[
        args.dataset
    ]

    (
        verification_path,
        reference_path,
        summary_path,
        cases_path,
    ) = get_paths(
        args.dataset,
        args.generator_model,
        args.evaluator_model,
    )

    if not verification_path.exists():
        raise FileNotFoundError(
            "Verification results were not found: "
            f"{verification_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            "Reference evaluation results "
            f"were not found: {reference_path}"
        )

    verification_records = load_jsonl(
        verification_path
    )

    reference_records = load_jsonl(
        reference_path
    )

    reference_labels = (
        load_reference_labels(
            reference_records
        )
    )

    predictions = (
        load_verification_predictions(
            verification_records,
            args.generator_model,
            args.cross_model,
        )
    )

    rows = build_cases(
        reference_labels,
        predictions,
    )

    self_cases = cases_for_mode(
        rows,
        "self",
    )

    cross_cases = cases_for_mode(
        rows,
        "cross",
    )

    self_metrics = calculate_metrics(
        self_cases
    )

    cross_metrics = calculate_metrics(
        cross_cases
    )

    reference_counts = (
        count_reference_labels(rows)
    )

    summary = {
        "dataset":
            args.dataset,
        "generator_model":
            args.generator_model,
        "cross_verifier_model":
            args.cross_model,
        "reference_evaluator_model":
            args.evaluator_model,
        "reference_counts":
            reference_counts,
        "binary_metric_scope":
            (
                "Reference labels CORRECT and "
                "INCORRECT only. NOT_ATTEMPTED "
                "is reported separately and "
                "excluded from binary metrics."
            ),
        "positive_class":
            "INCORRECT",
        "self":
            self_metrics,
        "cross":
            cross_metrics,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_cases(
        rows,
        cases_path,
    )

    print(
        f"{display_name} verification evaluation"
    )
    print("-" * 55)

    print(
        "Reference labels: "
        f"CORRECT={reference_counts['CORRECT']}, "
        f"INCORRECT={reference_counts['INCORRECT']}, "
        f"NOT_ATTEMPTED="
        f"{reference_counts['NOT_ATTEMPTED']}"
    )

    print()

    print("Self-verification")
    print(
        f"  cases={self_metrics['cases']}"
    )
    print(
        f"  TP={self_metrics['tp']} "
        f"TN={self_metrics['tn']} "
        f"FP={self_metrics['fp']} "
        f"FN={self_metrics['fn']}"
    )
    print(
        f"  precision="
        f"{format_metric(self_metrics['precision'])}"
    )
    print(
        f"  recall="
        f"{format_metric(self_metrics['recall'])}"
    )
    print(
        f"  F1="
        f"{format_metric(self_metrics['f1'])}"
    )
    print(
        f"  accuracy="
        f"{format_metric(self_metrics['accuracy'])}"
    )

    print()

    print("Cross-verification")
    print(
        f"  cases={cross_metrics['cases']}"
    )
    print(
        f"  TP={cross_metrics['tp']} "
        f"TN={cross_metrics['tn']} "
        f"FP={cross_metrics['fp']} "
        f"FN={cross_metrics['fn']}"
    )
    print(
        f"  precision="
        f"{format_metric(cross_metrics['precision'])}"
    )
    print(
        f"  recall="
        f"{format_metric(cross_metrics['recall'])}"
    )
    print(
        f"  F1="
        f"{format_metric(cross_metrics['f1'])}"
    )
    print(
        f"  accuracy="
        f"{format_metric(cross_metrics['accuracy'])}"
    )

    print()
    print(f"Summary: {summary_path}")
    print(f"Cases: {cases_path}")


if __name__ == "__main__":
    main()