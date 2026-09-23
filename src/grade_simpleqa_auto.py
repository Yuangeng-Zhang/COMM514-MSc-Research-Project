from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "simpleqa_verified.csv"
)

GENERATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_gemma4_generations.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "development"
    / "simpleqa_pilot_auto_grades_v2.jsonl"
)

GRADER_MODEL = "gpt-oss:20b-cloud"
PROMPT_VERSION = "simpleqa_reference_v2"

VALID_LABELS = {
    "CORRECT",
    "INCORRECT",
    "NOT_ATTEMPTED",
}

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


def load_generations():
    records = []

    with open(
        GENERATION_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def create_prompt(problem, reference_answer, candidate_answer):
    return f"""You are grading an answer to a factual question using a provided reference answer.

Classify the candidate answer as exactly one of:

CORRECT
INCORRECT
NOT_ATTEMPTED

Apply the labels in this order:

1. Use NOT_ATTEMPTED if the candidate does not make a substantive attempt to answer the question. This includes responses such as "I don't know", expressions of uncertainty without an answer, or refusals to answer.

2. Otherwise, use CORRECT if the candidate answer is factually consistent with the reference answer and answers the question. Equivalent wording is acceptable.

3. Otherwise, use INCORRECT if the candidate gives the wrong fact, contradicts the reference answer, or contains a material factual error.

Return exactly one label and nothing else.

Question:
{problem}

Reference answer:
{reference_answer}

Candidate answer:
{candidate_answer}

Label:"""


def parse_label(text):
    cleaned = text.strip().upper()
    cleaned = cleaned.strip("*_` \t\r\n")
    cleaned = cleaned.rstrip(".").strip()

    if cleaned in VALID_LABELS:
        return cleaned

    return "UNPARSED"


def load_completed():
    if not OUTPUT_PATH.exists():
        return set()

    completed = set()

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                record = json.loads(line)

                if record["ground_truth_label"] in VALID_LABELS:
                    completed.add(record["dataset_index"])

    return completed


def grade_answer(item, dataset_record):
    reference_answer = dataset_record["answer"]

    prompt = create_prompt(
        item["problem"],
        reference_answer,
        item["generated_answer"],
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=GRADER_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options={
                    "temperature": 0,
                },
            )

            raw_response = response.message.content.strip()
            label = parse_label(raw_response)

            return {
                "dataset_index": item["dataset_index"],
                "original_index": item["original_index"],
                "problem": item["problem"],
                "reference_answer": reference_answer,
                "generated_answer": item["generated_answer"],
                "generator_model": item["generator_model"],
                "ground_truth_label": label,
                "grading_method": "automatic_reference_check",
                "grader_model": GRADER_MODEL,
                "prompt_version": PROMPT_VERSION,
                "raw_response": raw_response,
                "supporting_urls": dataset_record["urls"],
                "timestamp_utc": datetime.now(
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
        "dataset_index": item["dataset_index"],
        "original_index": item["original_index"],
        "problem": item["problem"],
        "reference_answer": reference_answer,
        "generated_answer": item["generated_answer"],
        "generator_model": item["generator_model"],
        "ground_truth_label": "ERROR",
        "grading_method": "automatic_reference_check",
        "grader_model": GRADER_MODEL,
        "prompt_version": PROMPT_VERSION,
        "raw_response": "",
        "supporting_urls": dataset_record["urls"],
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "error": last_error,
    }


def save_result(result):
    with open(
        OUTPUT_PATH,
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
            "SimpleQA Verified dataset was not found."
        )

    if not GENERATION_PATH.exists():
        raise FileNotFoundError(
            "Generation pilot results were not found."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()
    generations = load_generations()
    completed = load_completed()

    remaining = [
        item
        for item in generations
        if item["dataset_index"] not in completed
    ]

    print("SimpleQA automatic grading development run")
    print("-" * 50)
    print(f"Generated answers: {len(generations)}")
    print(f"Grader model: {GRADER_MODEL}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Results: {OUTPUT_PATH}")
    print()

    for number, item in enumerate(
        remaining,
        start=1,
    ):
        index = item["dataset_index"]
        dataset_record = dataset[index]

        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index={index}"
        )

        result = grade_answer(
            item,
            dataset_record,
        )

        save_result(result)

        print(
            f"  label="
            f"{result['ground_truth_label']}"
        )

        time.sleep(0.5)

    print()
    print("Automatic grading development run completed.")


if __name__ == "__main__":
    main()