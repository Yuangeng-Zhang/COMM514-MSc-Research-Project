from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SIMPLEQA_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_evaluation_summary.json"
)

POPQA_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "popqa_dev50_evaluation_summary.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "benchmark_comparison_summary.json"
)


def load_summary(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def prepare_result(name, summary):
    distribution = summary["ground_truth_distribution"]
    self_metrics = summary["self_metrics"]
    cross_metrics = summary["cross_metrics"]

    return {
        "benchmark": name,
        "questions": summary["development_questions"],
        "attempted_answers": summary["attempted_answers"],
        "correct": distribution["CORRECT"],
        "incorrect": distribution["INCORRECT"],
        "not_attempted": distribution["NOT_ATTEMPTED"],
        "self_recall": self_metrics["recall"],
        "cross_recall": cross_metrics["recall"],
        "self_f1": self_metrics["f1"],
        "cross_f1": cross_metrics["f1"],
        "recall_gain":
            cross_metrics["recall"]
            - self_metrics["recall"],
        "f1_gain":
            cross_metrics["f1"]
            - self_metrics["f1"],
    }


def format_metric(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


def main():
    if not SIMPLEQA_PATH.exists():
        raise FileNotFoundError(
            "SimpleQA evaluation summary was not found."
        )

    if not POPQA_PATH.exists():
        raise FileNotFoundError(
            "PopQA evaluation summary was not found."
        )

    simpleqa = prepare_result(
        "SimpleQA",
        load_summary(SIMPLEQA_PATH),
    )

    popqa = prepare_result(
        "PopQA",
        load_summary(POPQA_PATH),
    )

    results = [simpleqa, popqa]

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Development benchmark comparison")
    print("-" * 78)
    print(
        f"{'Benchmark':<12}"
        f"{'Attempted':>10}"
        f"{'Self Recall':>14}"
        f"{'Cross Recall':>15}"
        f"{'Self F1':>11}"
        f"{'Cross F1':>11}"
    )
    print("-" * 78)

    for result in results:
        print(
            f"{result['benchmark']:<12}"
            f"{result['attempted_answers']:>10}"
            f"{format_metric(result['self_recall']):>14}"
            f"{format_metric(result['cross_recall']):>15}"
            f"{format_metric(result['self_f1']):>11}"
            f"{format_metric(result['cross_f1']):>11}"
        )

    print()
    print("Ground-truth distributions:")

    for result in results:
        print(
            f"  {result['benchmark']}: "
            f"{result['correct']} correct, "
            f"{result['incorrect']} incorrect, "
            f"{result['not_attempted']} not attempted"
        )

    print()
    print("Cross minus self:")

    for result in results:
        print(
            f"  {result['benchmark']}: "
            f"recall {result['recall_gain']:+.3f}, "
            f"F1 {result['f1_gain']:+.3f}"
        )

    print()
    print(
        "Comparison saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()