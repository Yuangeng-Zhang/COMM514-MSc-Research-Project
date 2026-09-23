from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simpleqa_verified_dev50.csv"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_dev50_gemma4_generations.jsonl"
)

MODEL = "gemma4:cloud"

client = Client(
    host="http://localhost:11434",
    timeout=300.0,
)


def load_dataset():
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def create_prompt(problem):
    return f"""Answer the following factual question.

Give a concise answer. If you do not know the answer, say "I don't know" rather than guessing.

Question:
{problem}

Answer:"""


def load_completed():
    if not RESULT_PATH.exists():
        return set()

    completed = set()

    with open(
        RESULT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                record = json.loads(line)

                if record.get("error") is None:
                    completed.add(
                        int(record["dataset_index"])
                    )

    return completed


def generate_answer(item):
    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": create_prompt(
                            item["problem"]
                        ),
                    }
                ],
                options={
                    "temperature": 0,
                },
            )

            return {
                "dataset_index":
                    int(item["dataset_index"]),
                "original_index":
                    item["original_index"],
                "problem":
                    item["problem"],
                "reference_answer":
                    item["answer"],
                "topic":
                    item["topic"],
                "answer_type":
                    item["answer_type"],
                "generated_answer":
                    response.message.content.strip(),
                "generator_model":
                    MODEL,
                "timestamp_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "error":
                    None,
            }

        except Exception as error:
            last_error = str(error)

            if attempt < 3:
                wait_seconds = attempt * 5

                print(
                    f"  Request failed. "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

    return {
        "dataset_index":
            int(item["dataset_index"]),
        "original_index":
            item["original_index"],
        "problem":
            item["problem"],
        "reference_answer":
            item["answer"],
        "topic":
            item["topic"],
        "answer_type":
            item["answer_type"],
        "generated_answer":
            "",
        "generator_model":
            MODEL,
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "error":
            last_error,
    }


def save_result(result):
    with open(
        RESULT_PATH,
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
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "SimpleQA development set was not found."
        )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_dataset()
    completed = load_completed()

    remaining = [
        item
        for item in records
        if int(item["dataset_index"])
        not in completed
    ]

    print("SimpleQA development generation")
    print("-" * 45)
    print(f"Model: {MODEL}")
    print(f"Development questions: {len(records)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Results: {RESULT_PATH}")
    print()

    for number, item in enumerate(
        remaining,
        start=1,
    ):
        index = item["dataset_index"]

        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index={index}"
        )

        result = generate_answer(item)
        save_result(result)

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