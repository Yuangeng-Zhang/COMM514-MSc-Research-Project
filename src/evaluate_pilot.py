from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "haluleval_pilot_gemma4_cloud.jsonl"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "haluleval_pilot_gemma4_cloud_summary.json"
)

ERROR_PATH = (
    PROJECT_ROOT
    / "results"
    / "haluleval_pilot_gemma4_cloud_errors.jsonl"
)


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            "Pilot result file was not found."
        )

    results = []

    with open(RESULT_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))

    valid_results = [
        item
        for item in results
        if item["prediction"]
        in {"SUPPORTED", "HALLUCINATED"}
    ]

    invalid_results = [
        item
        for item in results
        if item["prediction"]
        not in {"SUPPORTED", "HALLUCINATED"}
    ]

    # Treat HALLUCINATED as the positive class.
    tp = sum(
        item["true_label"] == "HALLUCINATED"
        and item["prediction"] == "HALLUCINATED"
        for item in valid_results
    )

    tn = sum(
        item["true_label"] == "SUPPORTED"
        and item["prediction"] == "SUPPORTED"
        for item in valid_results
    )

    fp = sum(
        item["true_label"] == "SUPPORTED"
        and item["prediction"] == "HALLUCINATED"
        for item in valid_results
    )

    fn = sum(
        item["true_label"] == "HALLUCINATED"
        and item["prediction"] == "SUPPORTED"
        for item in valid_results
    )

    total = len(valid_results)

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    errors = [
        item
        for item in valid_results
        if item["prediction"] != item["true_label"]
    ]

    summary = {
        "total_records_in_result_file": len(results),
        "valid_predictions": total,
        "invalid_predictions": len(invalid_results),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "misclassified_cases": len(errors),
    }

    print("HaluEval QA pilot evaluation")
    print("-" * 40)

    print(f"Total results: {len(results)}")
    print(f"Valid predictions: {total}")
    print(f"Invalid predictions: {len(invalid_results)}")

    print("\nConfusion matrix:")
    print(f"  True positive:  {tp}")
    print(f"  True negative:  {tn}")
    print(f"  False positive: {fp}")
    print(f"  False negative: {fn}")

    print("\nMetrics:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    print(
        f"\nMisclassified cases: "
        f"{len(errors)}"
    )

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

    with open(
        ERROR_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for item in errors:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"\nSummary saved to: {SUMMARY_PATH}")
    print(f"Errors saved to: {ERROR_PATH}")


if __name__ == "__main__":
    main()