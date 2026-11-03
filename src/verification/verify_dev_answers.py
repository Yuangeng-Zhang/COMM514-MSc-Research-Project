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
DEFAULT_CROSS_MODEL = "gpt-oss:20b-cloud"

VALID_LABELS = {
    "CORRECT",
    "INCORRECT",
}

client = Client(
    host="http://localhost:11434",
    timeout=300.0,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify generated factual answers using "
            "self- or cross-model verification."
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
        "--mode",
        choices=["self", "cross", "both"],
        default="both",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Use only the first N generated answers. "
            "With mode=both, each answer receives "
            "both self and cross verification."
        ),
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


def get_result_path(dataset, generator_model):
    return (
        PROJECT_ROOT
        / "results"
        / "development"
        / dataset
        / "verification"
        / (
            f"dev50_{model_slug(generator_model)}"
            f"_verifications.jsonl"
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

            if not record.get("generated_answer", "").strip():
                continue

            records.append(record)

    return records


def create_prompt(question, candidate_answer):
    return f"""You are verifying a candidate answer to a factual question.

Decide whether the candidate answer is factually correct.

Use your own knowledge only. Do not assume the answer is correct simply because it is confidently written.

Return exactly one label and nothing else:

CORRECT
INCORRECT

Question:
{question}

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

            if record.get("prediction") in VALID_LABELS:
                completed.add(
                    (
                        int(record["dataset_index"]),
                        record["verification_type"],
                        record["verifier_model"],
                    )
                )

    return completed


def verify(item, verifier_model, verification_type):
    prompt = create_prompt(
        item["question"],
        item["generated_answer"],
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=verifier_model,
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
                "generated_answer":
                    item["generated_answer"],
                "generator_model":
                    item["generator_model"],
                "verification_type":
                    verification_type,
                "verifier_model":
                    verifier_model,
                "prediction":
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
        "generated_answer":
            item["generated_answer"],
        "generator_model":
            item["generator_model"],
        "verification_type":
            verification_type,
        "verifier_model":
            verifier_model,
        "prediction":
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


def build_cases(
    generations,
    generator_model,
    cross_model,
    mode,
):
    cases = []

    if (
        mode in {"cross", "both"}
        and cross_model == generator_model
    ):
        raise ValueError(
            "Cross-verification requires a verifier "
            "different from the generator model."
        )

    for item in generations:
        if mode in {"self", "both"}:
            cases.append(
                (
                    item,
                    generator_model,
                    "self",
                )
            )

        if mode in {"cross", "both"}:
            cases.append(
                (
                    item,
                    cross_model,
                    "cross",
                )
            )

    return cases


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
    )

    if not generation_path.exists():
        raise FileNotFoundError(
            f"{display_name} generation results "
            f"were not found: {generation_path}"
        )

    generations = load_generations(
        generation_path
    )

    if args.limit is not None:
        generations = generations[:args.limit]

    completed = load_completed(
        result_path
    )

    cases = build_cases(
        generations,
        args.generator_model,
        args.cross_model,
        args.mode,
    )

    remaining = [
        case
        for case in cases
        if (
            int(case[0]["dataset_index"]),
            case[2],
            case[1],
        )
        not in completed
    ]

    print(
        f"{display_name} development verification"
    )
    print("-" * 45)
    print(
        f"Generator: {args.generator_model}"
    )

    if args.mode in {"self", "both"}:
        print(
            f"Self verifier: "
            f"{args.generator_model}"
        )

    if args.mode in {"cross", "both"}:
        print(
            f"Cross verifier: "
            f"{args.cross_model}"
        )

    print(f"Mode: {args.mode}")
    print(
        f"Generated answers loaded: "
        f"{len(generations)}"
    )
    print(f"Verification cases: {len(cases)}")
    print(
        f"Already completed: "
        f"{len(cases) - len(remaining)}"
    )
    print(f"Remaining: {len(remaining)}")
    print(f"Results: {result_path}")
    print()

    for number, (
        item,
        verifier_model,
        verification_type,
    ) in enumerate(
        remaining,
        start=1,
    ):
        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index="
            f"{item['dataset_index']} "
            f"type={verification_type}"
        )

        result = verify(
            item,
            verifier_model,
            verification_type,
        )

        save_result(
            result,
            result_path,
        )

        print(
            f"  prediction="
            f"{result['prediction']}"
        )

        time.sleep(0.5)

    print()
    print(
        "Development verification completed."
    )


if __name__ == "__main__":
    main()