from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import re
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "simpleqa": "SimpleQA",
    "popqa": "PopQA",
    "entityquestions": "EntityQuestions",
}

DEFAULT_GENERATOR_MODEL = "gemma4:cloud"
DEFAULT_EVALUATOR_MODEL = "gpt-oss:20b-cloud"

VALID_LABELS = {
    "CORRECT",
    "INCORRECT",
    "NOT_ATTEMPTED",
}

client = Client(
    host="http://localhost:11434",
    timeout=300.0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated factual answers using "
            "benchmark reference answers."
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
        "--evaluator-model",
        default=DEFAULT_EVALUATOR_MODEL,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N remaining answers.",
    )

    return parser.parse_args()


def model_slug(model):
    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        model,
    ).strip("_").lower()


def get_generation_path(dataset, generator_model):
    return (
        PROJECT_ROOT
        / "results"
        / "development"
        / dataset
        / "generations"
        / (
            f"dev50_{model_slug(generator_model)}"
            f"_generations.jsonl"
        )
    )


def get_result_path(
    dataset,
    generator_model,
    evaluator_model,
):
    return (
        PROJECT_ROOT
        / "results"
        / "development"
        / dataset
        / "reference_evaluation"
        / (
            f"dev50_{model_slug(generator_model)}"
            f"_evaluated_by_"
            f"{model_slug(evaluator_model)}.jsonl"
        )
    )


def load_generations(generation_path):
    records = []

    with open(
        generation_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("error") is not None:
                continue

            records.append(record)

    return records


def parse_acceptable_answers(item):
    value = item.get(
        "acceptable_answers",
        "",
    )

    if value:
        try:
            answers = json.loads(value)

            if isinstance(answers, list):
                return [
                    str(answer).strip()
                    for answer in answers
                    if str(answer).strip()
                ]

        except json.JSONDecodeError:
            pass

    reference_answer = item.get(
        "reference_answer",
        "",
    ).strip()

    if reference_answer:
        return [reference_answer]

    return []


def create_prompt(
    question,
    reference_answer,
    acceptable_answers,
    candidate_answer,
):
    acceptable_text = json.dumps(
        acceptable_answers,
        ensure_ascii=False,
    )

    return f"""Evaluate a candidate answer to a factual question using the benchmark reference information.

Return exactly one label and nothing else:

CORRECT
INCORRECT
NOT_ATTEMPTED

Use these rules:

CORRECT:
The candidate gives the correct factual answer. Equivalent wording or an acceptable answer variant is allowed.

INCORRECT:
The candidate gives a factual answer that conflicts with, or does not correctly answer, the benchmark question.

NOT_ATTEMPTED:
The candidate does not provide a substantive answer, for example by saying "I don't know" or explicitly refusing to answer.

Question:
{question}

Primary reference answer:
{reference_answer}

Acceptable answers:
{acceptable_text}

Candidate answer:
{candidate_answer}

Label:"""


def parse_label(text):
    cleaned = text.strip().upper()
    cleaned = cleaned.strip("*_` \t\r\n")
    cleaned = cleaned.rstrip(".").strip()
    cleaned = cleaned.replace("-", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)

    if cleaned in VALID_LABELS:
        return cleaned

    return "UNPARSED"


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

            if record.get("label") in VALID_LABELS:
                completed.add(
                    int(record["dataset_index"])
                )

    return completed


def evaluate_answer(item, evaluator_model):
    acceptable_answers = (
        parse_acceptable_answers(item)
    )

    prompt = create_prompt(
        item["question"],
        item["reference_answer"],
        acceptable_answers,
        item["generated_answer"],
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=evaluator_model,
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

            raw_response = (
                response.message.content.strip()
            )

            return {
                "dataset_name":
                    item["dataset_name"],
                "dataset_index":
                    int(item["dataset_index"]),
                "question":
                    item["question"],
                "reference_answer":
                    item["reference_answer"],
                "acceptable_answers":
                    acceptable_answers,
                "generated_answer":
                    item["generated_answer"],
                "generator_model":
                    item["generator_model"],
                "evaluator_model":
                    evaluator_model,
                "label":
                    parse_label(raw_response),
                "raw_response":
                    raw_response,
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
        "dataset_name":
            item["dataset_name"],
        "dataset_index":
            int(item["dataset_index"]),
        "question":
            item["question"],
        "reference_answer":
            item["reference_answer"],
        "acceptable_answers":
            acceptable_answers,
        "generated_answer":
            item["generated_answer"],
        "generator_model":
            item["generator_model"],
        "evaluator_model":
            evaluator_model,
        "label":
            "ERROR",
        "raw_response":
            "",
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "error":
            last_error,
    }


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

    if args.limit is not None and args.limit < 1:
        raise ValueError(
            "--limit must be at least 1."
        )

    display_name = DATASETS[args.dataset]

    generation_path = get_generation_path(
        args.dataset,
        args.generator_model,
    )

    result_path = get_result_path(
        args.dataset,
        args.generator_model,
        args.evaluator_model,
    )

    if not generation_path.exists():
        raise FileNotFoundError(
            f"{display_name} generation results "
            f"were not found: {generation_path}"
        )

    generations = load_generations(
        generation_path
    )

    completed = load_completed(
        result_path
    )

    remaining = [
        item
        for item in generations
        if int(item["dataset_index"])
        not in completed
    ]

    if args.limit is not None:
        remaining = remaining[:args.limit]

    print(
        f"{display_name} reference evaluation"
    )
    print("-" * 45)
    print(
        f"Generator: {args.generator_model}"
    )
    print(
        f"Reference evaluator: "
        f"{args.evaluator_model}"
    )
    print(
        f"Generated answers: "
        f"{len(generations)}"
    )
    print(
        f"Already completed: "
        f"{len(completed)}"
    )
    print(
        f"Evaluating now: "
        f"{len(remaining)}"
    )
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

        result = evaluate_answer(
            item,
            args.evaluator_model,
        )

        save_result(
            result,
            result_path,
        )

        print(
            f"  label="
            f"{result['label']}"
        )

        time.sleep(0.5)

    print()
    print(
        "Reference evaluation completed."
    )


if __name__ == "__main__":
    main()