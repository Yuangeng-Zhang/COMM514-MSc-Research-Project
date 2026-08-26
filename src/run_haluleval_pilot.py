from pathlib import Path
from datetime import datetime, timezone
import json
import random
import re
import time

from ollama import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "qa_data.json"
RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "haluleval_pilot_gemma4_cloud.jsonl"
)

MODEL = "gemma4:cloud"

PILOT_QUESTIONS = 50
SEED = 514

client = Client(host="http://localhost:11434")


def load_dataset():
    records = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_cases(records):
    rng = random.Random(SEED)

    selected_indices = rng.sample(
        range(len(records)),
        PILOT_QUESTIONS,
    )

    cases = []

    for index in selected_indices:
        record = records[index]

        cases.append(
            {
                "case_id": f"{index}_right",
                "dataset_index": index,
                "knowledge": record["knowledge"],
                "question": record["question"],
                "candidate_answer": record["right_answer"],
                "true_label": "SUPPORTED",
            }
        )

        cases.append(
            {
                "case_id": f"{index}_hallucinated",
                "dataset_index": index,
                "knowledge": record["knowledge"],
                "question": record["question"],
                "candidate_answer": record["hallucinated_answer"],
                "true_label": "HALLUCINATED",
            }
        )

    # Avoid always presenting the correct answer first.
    random.Random(SEED + 1).shuffle(cases)

    return cases


def create_prompt(case):
    return f"""You are checking whether a candidate answer is supported by the supplied knowledge.

Use only the supplied knowledge when making the judgement.

Return exactly one of these two labels and nothing else:

SUPPORTED
HALLUCINATED

Use SUPPORTED when the candidate answer is supported by the knowledge.
Use HALLUCINATED when the candidate answer contains a factual claim that is contradicted by, or unsupported by, the knowledge.

Knowledge:
{case["knowledge"]}

Question:
{case["question"]}

Candidate answer:
{case["candidate_answer"]}

Label:"""


def parse_prediction(text):
    cleaned = text.strip().upper()

    match = re.search(
        r"\b(SUPPORTED|HALLUCINATED)\b",
        cleaned,
    )

    if match:
        return match.group(1)

    return "UNPARSED"


def load_completed_cases():
    if not RESULT_PATH.exists():
        return set()

    completed = set()

    with open(RESULT_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                result = json.loads(line)
                completed.add(result["case_id"])

    return completed


def run_case(case):
    prompt = create_prompt(case)

    last_error = None

    for attempt in range(1, 4):
        try:
            response = client.chat(
                model=MODEL,
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

            return {
                **case,
                "prediction": parse_prediction(raw_response),
                "raw_response": raw_response,
                "model": MODEL,
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "prompt_eval_count": getattr(
                    response,
                    "prompt_eval_count",
                    None,
                ),
                "eval_count": getattr(
                    response,
                    "eval_count",
                    None,
                ),
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
        **case,
        "prediction": "ERROR",
        "raw_response": "",
        "model": MODEL,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "prompt_eval_count": None,
        "eval_count": None,
        "error": last_error,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "HaluEval QA dataset was not found."
        )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_dataset()
    cases = build_cases(records)

    completed = load_completed_cases()

    print("HaluEval QA pilot")
    print("-" * 40)
    print(f"Model: {MODEL}")
    print(f"Dataset records: {len(records):,}")
    print(f"Selected questions: {PILOT_QUESTIONS}")
    print(f"Classification cases: {len(cases)}")
    print(f"Random seed: {SEED}")
    print(f"Already completed: {len(completed)}")
    print(f"Results: {RESULT_PATH}")
    print()

    remaining = [
        case
        for case in cases
        if case["case_id"] not in completed
    ]

    for number, case in enumerate(
        remaining,
        start=1,
    ):
        print(
            f"[{number}/{len(remaining)}] "
            f"{case['case_id']} "
            f"(true={case['true_label']})"
        )

        result = run_case(case)

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
            f"  prediction={result['prediction']}"
        )

        # Small pause between cloud requests.
        time.sleep(0.5)

    print()
    print("Pilot run completed.")


if __name__ == "__main__":
    main()