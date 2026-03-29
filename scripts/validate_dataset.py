"""
scripts/validate_dataset.py — Golden Dataset Validator
Validates the golden_dataset.json file before evaluation runs.
Ensures all required fields are present and values are correct.
Usage: python scripts/validate_dataset.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import GOLDEN_DATASET_PATH

# Validation Rules

REQUIRED_FIELDS = {"id", "question", "expected_answer", "source_document", "difficulty", "category"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
MIN_QUESTIONS = 25


def validate_dataset(path: str) -> tuple[bool, list[str]]:
    """
    Validate the golden dataset file.

    Args:
        path: Path to the golden_dataset.json file.

    Returns:
        Tuple of (is_valid, list_of_errors).
        is_valid is True only if there are zero errors.
    """
    errors: list[str] = []

    # Check file exists
    if not Path(path).exists():
        return False, [f"Dataset file not found: {path}"]

    # Load JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    # Check it's a list
    if not isinstance(dataset, list):
        return False, ["Dataset must be a JSON array of objects."]

    # Check minimum number of questions
    if len(dataset) < MIN_QUESTIONS:
        errors.append(
            f"Dataset has {len(dataset)} questions. Minimum required: {MIN_QUESTIONS}."
        )

    # Check for duplicate IDs
    ids = [entry.get("id") for entry in dataset]
    duplicate_ids = {id_ for id_ in ids if ids.count(id_) > 1}
    if duplicate_ids:
        errors.append(f"Duplicate IDs found: {duplicate_ids}")

    # Validate each entry
    for i, entry in enumerate(dataset):
        entry_id = entry.get("id", f"entry_{i}")

        # Check required fields
        missing = REQUIRED_FIELDS - set(entry.keys())
        if missing:
            errors.append(f"[{entry_id}] Missing fields: {missing}")
            continue

        # Check no empty strings
        for field in REQUIRED_FIELDS:
            value = entry.get(field, "")
            if not isinstance(value, str) or not value.strip():
                errors.append(f"[{entry_id}] Field '{field}' is empty or not a string.")

        # Check difficulty value
        if entry.get("difficulty") not in VALID_DIFFICULTIES:
            errors.append(
                f"[{entry_id}] Invalid difficulty '{entry.get('difficulty')}'. "
                f"Must be one of: {VALID_DIFFICULTIES}"
            )

        # Check question is meaningful (more than 10 chars)
        if len(entry.get("question", "")) < 10:
            errors.append(f"[{entry_id}] Question is too short (less than 10 characters).")

        # Check expected answer is meaningful (more than 20 chars)
        if len(entry.get("expected_answer", "")) < 20:
            errors.append(f"[{entry_id}] Expected answer is too short (less than 20 characters).")

    return len(errors) == 0, errors


# Entry Point
if __name__ == "__main__":
    print("=" * 60)
    print("Golden Dataset Validator")
    print("=" * 60)
    print(f"Validating: {GOLDEN_DATASET_PATH}")
    print()

    is_valid, errors = validate_dataset(GOLDEN_DATASET_PATH)

    if is_valid:
        # Load and show summary
        with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for entry in dataset:
            difficulties[entry["difficulty"]] += 1

        print(f"✅ Dataset is valid!")
        print(f"   Total questions : {len(dataset)}")
        print(f"   Easy            : {difficulties['easy']}")
        print(f"   Medium          : {difficulties['medium']}")
        print(f"   Hard            : {difficulties['hard']}")
        sys.exit(0)

    else:
        print(f"❌ Dataset validation failed with {len(errors)} error(s):\n")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)