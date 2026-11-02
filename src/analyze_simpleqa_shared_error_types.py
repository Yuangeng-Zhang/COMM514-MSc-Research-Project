from pathlib import Path
import csv
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_shared_error_coding_marked.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_shared_error_type_summary.json"
)

VALID_TYPES = {
    "same_factual_error",
    "different_factual_error",
    "indeterminate",
}

VALID_PREDICTIONS = {
    "CORRECT",
    "INCORRECT",
}

EXPECTED_CASES = 25


def load_cases():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def summarize(cases):
    detected = sum(
        1
        for case in cases
        if case["cross_prediction"] == "INCORRECT"
    )

    missed = sum(
        1
        for case in cases
        if case["cross_prediction"] == "CORRECT"
    )

    rate = (
        detected / len(cases)
        if cases
        else None
    )

    return {
        "cases": len(cases),
        "detected": detected,
        "missed": missed,
        "detection_rate": rate,
    }


def format_rate(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Marked coding file not found: {INPUT_PATH}"
        )

    cases = load_cases()

    if len(cases) != EXPECTED_CASES:
        raise ValueError(
            f"Expected {EXPECTED_CASES} cases, "
            f"but found {len(cases)}."
        )

    invalid_types = [
        case
        for case in cases
        if case.get("shared_error_type")
        not in VALID_TYPES
    ]

    if invalid_types:
        print("Invalid or missing shared_error_type:")
        for case in invalid_types:
            print(
                f"  dataset_index="
                f"{case.get('dataset_index')} "
                f"type="
                f"{case.get('shared_error_type')!r}"
            )

        raise SystemExit(1)

    invalid_predictions = [
        case
        for case in cases
        if case.get("cross_prediction")
        not in VALID_PREDICTIONS
    ]

    if invalid_predictions:
        print("Invalid cross_prediction:")
        for case in invalid_predictions:
            print(
                f"  dataset_index="
                f"{case.get('dataset_index')} "
                f"prediction="
                f"{case.get('cross_prediction')!r}"
            )

        raise SystemExit(1)

    same_cases = [
        case
        for case in cases
        if case["shared_error_type"]
        == "same_factual_error"
    ]

    different_cases = [
        case
        for case in cases
        if case["shared_error_type"]
        == "different_factual_error"
    ]

    indeterminate_cases = [
        case
        for case in cases
        if case["shared_error_type"]
        == "indeterminate"
    ]

    same_summary = summarize(same_cases)
    different_summary = summarize(different_cases)
    indeterminate_summary = summarize(
        indeterminate_cases
    )

    if (
        same_summary["detection_rate"] is not None
        and
        different_summary["detection_rate"] is not None
    ):
        rate_difference = (
            same_summary["detection_rate"]
            - different_summary["detection_rate"]
        )
    else:
        rate_difference = None

    summary = {
        "total_shared_error_items": len(cases),
        "same_factual_error": same_summary,
        "different_factual_error":
            different_summary,
        "indeterminate":
            indeterminate_summary,
        "same_minus_different_detection_rate":
            rate_difference,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("SimpleQA shared factual-error analysis")
    print("-" * 60)

    print(
        "Same factual error:"
        f" cases={same_summary['cases']},"
        f" detected={same_summary['detected']},"
        f" missed={same_summary['missed']},"
        f" rate="
        f"{format_rate(same_summary['detection_rate'])}"
    )

    print(
        "Different factual error:"
        f" cases={different_summary['cases']},"
        f" detected={different_summary['detected']},"
        f" missed={different_summary['missed']},"
        f" rate="
        f"{format_rate(different_summary['detection_rate'])}"
    )

    print(
        "Indeterminate:"
        f" cases={indeterminate_summary['cases']},"
        f" detected={indeterminate_summary['detected']},"
        f" missed={indeterminate_summary['missed']},"
        f" rate="
        f"{format_rate(indeterminate_summary['detection_rate'])}"
    )

    print()

    print(
        "Same minus different detection rate: "
        f"{format_rate(rate_difference)}"
    )

    print()
    print(f"Summary saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()