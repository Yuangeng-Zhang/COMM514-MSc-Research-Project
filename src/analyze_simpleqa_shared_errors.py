from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GEMMA_GRADES_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_gemma4_auto_grades.jsonl"
)

GPTOSS_GRADES_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_gptoss_auto_grades.jsonl"
)

VERIFICATIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_verifications.jsonl"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_shared_error_summary.json"
)

CASES_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_shared_error_cases.jsonl"
)

VALID_GRADES = {
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

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def load_grades(path):
    records = load_jsonl(path)

    return {
        int(record["dataset_index"]): record
        for record in records
        if record.get("ground_truth_label") in VALID_GRADES
    }


def load_latest_cross_verifications():
    records = load_jsonl(VERIFICATIONS_PATH)
    latest = {}

    for record in records:
        if record.get("verification_type") != "cross":
            continue

        if record.get("prediction") not in VALID_PREDICTIONS:
            continue

        dataset_index = int(record["dataset_index"])
        latest[dataset_index] = record

    return latest


def detection_summary(cases):
    detected = sum(
        1
        for case in cases
        if case["cross_prediction"] == "INCORRECT"
    )

    missed = len(cases) - detected

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
    gemma = load_grades(GEMMA_GRADES_PATH)
    gptoss = load_grades(GPTOSS_GRADES_PATH)
    cross = load_latest_cross_verifications()

    common_indices = sorted(
        set(gemma)
        & set(gptoss)
    )

    gemma_errors = {
        index
        for index in common_indices
        if gemma[index]["ground_truth_label"] == "INCORRECT"
    }

    gptoss_errors = {
        index
        for index in common_indices
        if gptoss[index]["ground_truth_label"] == "INCORRECT"
    }

    intersection = gemma_errors & gptoss_errors
    union = gemma_errors | gptoss_errors

    jaccard = (
        len(intersection) / len(union)
        if union
        else None
    )

    cases = []

    for index in sorted(gemma_errors):
        if index not in cross:
            continue

        gptoss_label = gptoss[index]["ground_truth_label"]

        if gptoss_label == "INCORRECT":
            group = "shared_error"
        elif gptoss_label == "CORRECT":
            group = "gptoss_correct"
        else:
            group = "gptoss_not_attempted"

        verification = cross[index]

        cases.append(
            {
                "dataset_index": index,
                "question":
                    gemma[index].get("question"),
                "reference_answer":
                    gemma[index].get("reference_answer"),
                "gemma_answer":
                    gemma[index].get("generated_answer"),
                "gptoss_answer":
                    gptoss[index].get("generated_answer"),
                "gemma_label":
                    gemma[index]["ground_truth_label"],
                "gptoss_label":
                    gptoss_label,
                "group":
                    group,
                "cross_prediction":
                    verification["prediction"],
                "cross_detected_error":
                    verification["prediction"]
                    == "INCORRECT",
            }
        )

    shared_cases = [
        case
        for case in cases
        if case["group"] == "shared_error"
    ]

    correct_cases = [
        case
        for case in cases
        if case["group"] == "gptoss_correct"
    ]

    abstained_cases = [
        case
        for case in cases
        if case["group"] == "gptoss_not_attempted"
    ]

    shared_summary = detection_summary(shared_cases)
    correct_summary = detection_summary(correct_cases)
    abstained_summary = detection_summary(abstained_cases)

    summary = {
        "questions_compared": len(common_indices),
        "gemma_incorrect": len(gemma_errors),
        "gptoss_incorrect": len(gptoss_errors),
        "shared_incorrect_items": len(intersection),
        "incorrect_union": len(union),
        "error_jaccard": jaccard,
        "cross_verification_on_gemma_errors": {
            "shared_error": shared_summary,
            "gptoss_correct": correct_summary,
            "gptoss_not_attempted": abstained_summary,
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
            ensure_ascii=False,
            indent=2,
        )

    with open(
        CASES_PATH,
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

    print("SimpleQA shared-error analysis")
    print("-" * 55)

    print(f"Questions compared: {len(common_indices)}")
    print(f"Gemma incorrect: {len(gemma_errors)}")
    print(f"GPT-OSS incorrect: {len(gptoss_errors)}")
    print(f"Shared incorrect items: {len(intersection)}")
    print(f"Incorrect union: {len(union)}")
    print(
        "Error Jaccard: "
        f"{format_rate(jaccard)}"
    )

    print()
    print("GPT-OSS cross-verification on Gemma errors")
    print("-" * 55)

    print(
        "Shared error:"
        f" cases={shared_summary['cases']},"
        f" detected={shared_summary['detected']},"
        f" missed={shared_summary['missed']},"
        f" rate={format_rate(shared_summary['detection_rate'])}"
    )

    print(
        "GPT-OSS independently correct:"
        f" cases={correct_summary['cases']},"
        f" detected={correct_summary['detected']},"
        f" missed={correct_summary['missed']},"
        f" rate={format_rate(correct_summary['detection_rate'])}"
    )

    print(
        "GPT-OSS abstained:"
        f" cases={abstained_summary['cases']},"
        f" detected={abstained_summary['detected']},"
        f" missed={abstained_summary['missed']},"
        f" rate={format_rate(abstained_summary['detection_rate'])}"
    )

    print()
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Cases saved to: {CASES_PATH}")


if __name__ == "__main__":
    main()