from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "popqa.csv"
)

DATASET_NAME = "akariasai/PopQA"
SPLIT = "test"


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading PopQA")
    print("-" * 45)

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
    )

    print(f"Records: {len(dataset):,}")
    print(f"Columns: {dataset.column_names}")
    print()

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("First record:")

    first = dataset[0]

    print(f"Question: {first['question']}")
    print(f"Answer: {first['obj']}")
    print(
        f"Possible answers: "
        f"{first['possible_answers']}"
    )


if __name__ == "__main__":
    main()