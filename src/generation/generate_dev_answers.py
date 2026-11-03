from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import json
import re
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "simpleqa": {
        "display_name": "SimpleQA",
        "data_path": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "simpleqa_verified_dev50.csv"
        ),
    },
    "popqa": {
        "display_name": "PopQA",
        "data_path": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "popqa_dev50.csv"
        ),
    },
    "entityquestions": {
        "display_name": "EntityQuestions",
        "data_path": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "entityquestions_dev50.csv"
        ),
    },
}

DEFAULT_MODEL = "gemma4:cloud"

REQUIRED_COLUMNS = {
    "dataset_name",
    "dataset_index",
    "question",
    "reference_answer",
    "acceptable_answers",
}

client = Client(
    host="http://localhost:11434",
    timeout=300.0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate answers for a processed "
            "development benchmark."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS.keys(),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N remaining items.",
    )

    return parser.parse_args()


def model_slug(model):
    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        model,
    ).strip("_").lower()


def get_result_path(dataset, model):
    return (
        PROJECT_ROOT
        / "results"
        / "development"
        / dataset
        / "generations"
        / f"dev50_{model_slug(model)}_generations.jsonl"
    )


def load_dataset(data_path, dataset):
    with open(
        data_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = list(csv.DictReader(file))

    if not records:
        raise ValueError(
            f"{dataset} development set is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(records[0].keys())
    )

    if missing_columns:
        raise ValueError(
            f"{dataset} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return records


def create_prompt(question):
    return f"""Answer the following factual question.

Give a concise answer. If you do not know the answer, say "I don't know" rather than guessing.

Question:
{question}

Answer:"""


def load_completed(result_path):
    if not result_path.exists():
        return set()

    completed = set()

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("error") is None:
                completed.add(
                    int(record["dataset_index"])
                )

    return completed


def generate_answer(item, model):
    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": create_prompt(
                            item["question"]
                        ),
                    }
                ],
                options={
                    "temperature": 0,
                },
            )

            result = dict(item)

            result["dataset_index"] = int(
                item["dataset_index"]
            )
            result["generated_answer"] = (
                response.message.content.strip()
            )
            result["generator_model"] = model
            result["timestamp_utc"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )
            result["error"] = None

            return result

        except Exception as error:
            last_error = str(error)

            if attempt < 3:
                wait_seconds = attempt * 5

                print(
                    f"  Request failed. "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

    result = dict(item)

    result["dataset_index"] = int(
        item["dataset_index"]
    )
    result["generated_answer"] = ""
    result["generator_model"] = model
    result["timestamp_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )
    result["error"] = last_error

    return result


def save_result(result, result_path):
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        result_path,
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
    args = parse_args()

    config = DATASETS[args.dataset]
    data_path = config["data_path"]
    display_name = config["display_name"]

    if not data_path.exists():
        raise FileNotFoundError(
            f"{display_name} development set "
            f"was not found: {data_path}"
        )

    if args.limit is not None and args.limit < 1:
        raise ValueError(
            "--limit must be at least 1."
        )

    result_path = get_result_path(
        args.dataset,
        args.model,
    )

    records = load_dataset(
        data_path,
        args.dataset,
    )

    completed = load_completed(
        result_path
    )

    remaining = [
        item
        for item in records
        if int(item["dataset_index"])
        not in completed
    ]

    if args.limit is not None:
        remaining = remaining[:args.limit]

    print(
        f"{display_name} development generation"
    )
    print("-" * 45)
    print(f"Model: {args.model}")
    print(f"Development questions: {len(records)}")
    print(f"Already completed: {len(completed)}")
    print(f"Generating now: {len(remaining)}")
    print(f"Results: {result_path}")
    print()

    for number, item in enumerate(
        remaining,
        start=1,
    ):
        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index="
            f"{item['dataset_index']}"
        )

        result = generate_answer(
            item,
            args.model,
        )

        save_result(
            result,
            result_path,
        )

        if result["error"] is None:
            print(
                f"  answer="
                f"{result['generated_answer'][:100]}"
            )
        else:
            print(
                f"  ERROR: "
                f"{result['error']}"
            )

        time.sleep(0.5)

    print()
    print("Development generation completed.")


if __name__ == "__main__":
    main()