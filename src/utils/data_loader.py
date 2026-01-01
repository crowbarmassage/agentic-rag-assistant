"""Utilities for loading FAQ data from JSON files."""

import json
from pathlib import Path

from src.models import QAPair, DepartmentFAQs, Department, UserType


def load_department_faqs(file_path: str | Path) -> DepartmentFAQs:
    """
    Load FAQ data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        DepartmentFAQs object
    """
    file_path = Path(file_path)

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert QA pairs
    qa_pairs = []
    for qa_data in data["qa_pairs"]:
        qa = QAPair(
            id=qa_data["id"],
            question=qa_data["question"],
            answer=qa_data["answer"],
            department=Department(qa_data["department"]),
            user_type=UserType(qa_data["user_type"]),
            keywords=qa_data.get("keywords", [])
        )
        qa_pairs.append(qa)

    return DepartmentFAQs(
        department=Department(data["department"]),
        user_type=UserType(data["user_type"]),
        description=data["description"],
        qa_pairs=qa_pairs
    )


def load_all_faqs(data_dir: str = "./data/raw") -> dict[str, DepartmentFAQs]:
    """
    Load all FAQ data from a directory.

    Args:
        data_dir: Directory containing JSON files

    Returns:
        Dictionary mapping department keys to DepartmentFAQs
    """
    data_path = Path(data_dir)
    results = {}

    for json_file in data_path.glob("*_faqs.json"):
        dept_key = json_file.stem.replace("_faqs", "")
        results[dept_key] = load_department_faqs(json_file)
        print(f"Loaded {results[dept_key].count} FAQs for {dept_key}")

    return results


def get_all_qa_pairs(data_dir: str = "./data/raw") -> list[QAPair]:
    """
    Load all QA pairs as a flat list.

    Args:
        data_dir: Directory containing JSON files

    Returns:
        List of all QAPair objects
    """
    all_faqs = load_all_faqs(data_dir)
    all_pairs = []

    for dept_faqs in all_faqs.values():
        all_pairs.extend(dept_faqs.qa_pairs)

    return all_pairs
