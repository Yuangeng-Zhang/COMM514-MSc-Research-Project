from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import random
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simpleqa_verified.csv"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_gemma4_generations.jsonl"
)

MODEL = "gemma4:cloud"
PILOT_SIZE = 20
SEED = 514

client = Client(host="http://localhost:11434")


def load_dataset():
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def select_questions(records):
    rng = random.Random(SEED)
    indices = rng.sample(
        range(len(records)),
        PILOT_SIZE,
    )

    return [
        {
            "dataset_index": index,
            "original_index": records[index]["original_index"],
            "problem": records[index]["problem"],
            "reference_answer": records[index]["answer"],
            "topic": records[index]["topic"],
            "answer_type": records[index]["answer_type"],
        }
        for index in indices
    ]


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
                result = json.loads(line)
                completed.add(result["dataset_index"])

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
                **item,
                "generated_answer":
                    response.message.content.strip(),
                "generator_model": MODEL,
                "timestamp_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "error": None,
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
        **item,
        "generated_answer": "",
        "generator_model": MODEL,
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "error": last_error,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "SimpleQA Verified dataset was not found."
        )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_dataset()
    selected = select_questions(records)
    completed = load_completed()

    remaining = [
        item
        for item in selected
        if item["dataset_index"] not in completed
    ]

    print("SimpleQA Verified generation pilot")
    print("-" * 45)
    print(f"Model: {MODEL}")
    print(f"Dataset records: {len(records):,}")
    print(f"Pilot questions: {PILOT_SIZE}")
    print(f"Random seed: {SEED}")
    print(f"Already completed: {len(completed)}")
    print(f"Results: {RESULT_PATH}")
    print()

    for number, item in enumerate(
        remaining,
        start=1,
    ):
        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index={item['dataset_index']}"
        )

        result = generate_answer(item)

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

        print(
            f"  answer="
            f"{result['generated_answer'][:100]}"
        )

        time.sleep(0.5)

    print()
    print("Generation pilot completed.")


if __name__ == "__main__":
    main()