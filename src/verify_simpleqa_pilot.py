from pathlib import Path
from datetime import datetime, timezone
import json
import re
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_gemma4_generations.jsonl"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "simpleqa_pilot_verifications.jsonl"
)

SELF_MODEL = "gemma4:cloud"
CROSS_MODEL = "gpt-oss:20b-cloud"

client = Client(
    host="http://localhost:11434",
    timeout=300.0,
)


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


def create_prompt(problem, candidate_answer):
    return f"""You are verifying a candidate answer to a factual question.

Decide whether the candidate answer is factually correct.

Use your own knowledge only. Do not assume the answer is correct simply because it is confidently written.

Return exactly one label and nothing else:

CORRECT
INCORRECT

Question:
{problem}

Candidate answer:
{candidate_answer}

Label:"""


def parse_label(text):
    cleaned = text.strip().upper()

    # Allow minor Markdown formatting such as **CORRECT**
    cleaned = re.sub(r"[*_`]", "", cleaned).strip()

    if cleaned == "CORRECT":
        return "CORRECT"

    if cleaned == "INCORRECT":
        return "INCORRECT"

    return "UNPARSED"


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

                if result["prediction"] != "ERROR":
                    completed.add(
                        (
                            result["dataset_index"],
                            result["verification_type"],
                        )
                    )
    return completed


def verify(item, model, verification_type):
    prompt = create_prompt(
        item["problem"],
        item["generated_answer"],
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=model,
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
                "dataset_index":
                    item["dataset_index"],
                "original_index":
                    item["original_index"],
                "problem":
                    item["problem"],
                "reference_answer":
                    item["reference_answer"],
                "generated_answer":
                    item["generated_answer"],
                "generator_model":
                    item["generator_model"],
                "verification_type":
                    verification_type,
                "verifier_model":
                    model,
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
        "dataset_index":
            item["dataset_index"],
        "original_index":
            item["original_index"],
        "problem":
            item["problem"],
        "reference_answer":
            item["reference_answer"],
        "generated_answer":
            item["generated_answer"],
        "generator_model":
            item["generator_model"],
        "verification_type":
            verification_type,
        "verifier_model":
            model,
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
    if not GENERATION_PATH.exists():
        raise FileNotFoundError(
            "Generation pilot results were not found."
        )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generations = load_generations()
    completed = load_completed()

    print("SimpleQA verification pilot")
    print("-" * 45)
    print(f"Generated answers: {len(generations)}")
    print(f"Generator: {SELF_MODEL}")
    print(f"Self verifier: {SELF_MODEL}")
    print(f"Cross verifier: {CROSS_MODEL}")
    print(f"Already completed: {len(completed)}")
    print(f"Results: {RESULT_PATH}")
    print()

    cases = []

    for item in generations:
        cases.append(
            (
                item,
                SELF_MODEL,
                "self",
            )
        )

        cases.append(
            (
                item,
                CROSS_MODEL,
                "cross",
            )
        )

    remaining = [
        case
        for case in cases
        if (
            case[0]["dataset_index"],
            case[2],
        )
        not in completed
    ]

    for number, (
        item,
        model,
        verification_type,
    ) in enumerate(
        remaining,
        start=1,
    ):
        print(
            f"[{number}/{len(remaining)}] "
            f"dataset_index={item['dataset_index']} "
            f"type={verification_type} "
            f"model={model}"
        )

        result = verify(
            item,
            model,
            verification_type,
        )

        save_result(result)

        print(
            f"  prediction="
            f"{result['prediction']}"
        )

        time.sleep(0.5)

    print()
    print("Verification pilot completed.")


if __name__ == "__main__":
    main()