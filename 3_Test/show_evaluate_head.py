#!/usr/bin/env python3
from pathlib import Path
from datasets import load_from_disk
import json


def main():
    project_root = Path(__file__).resolve().parent
    evaluate_dir = project_root / "tasks" / "musique" / "modfied_test"

    dataset = load_from_disk(str(evaluate_dir))

    num_to_show = 10
    dataset_length = len(dataset)
    count = num_to_show if dataset_length >= num_to_show else dataset_length

    for index in range(count):
        example = dataset[index]
        print(f"===== Example {index} =====")
        print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()