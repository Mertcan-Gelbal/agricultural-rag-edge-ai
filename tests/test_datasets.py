"""Smoke tests for the classification dataset assets.

These run without a GPU, network access, or API keys — they only validate the
committed CSVs so a training run is not started against a broken split.
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

PRIMARY_CATEGORIES = {
    "plant_disease",
    "crop_management",
    "plant_genetics",
    "environmental_factors",
    "food_security",
    "technology",
}


def _read(name: str) -> list[dict[str, str]]:
    with open(DATA_DIR / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_splits_exist_and_have_expected_columns() -> None:
    for name in ("train.csv", "val.csv", "test.csv"):
        rows = _read(name)
        assert rows, f"{name} is empty"
        assert set(rows[0].keys()) == {"text", "label"}, f"{name} unexpected columns"


def test_split_sizes_match_documentation() -> None:
    # Documented in README: train 1,262 / val 270 / test 271
    assert len(_read("train.csv")) == 1262
    assert len(_read("val.csv")) == 270
    assert len(_read("test.csv")) == 271


def test_primary_categories_present_in_every_split() -> None:
    for name in ("train.csv", "val.csv", "test.csv"):
        labels = {r["label"] for r in _read(name)}
        missing = PRIMARY_CATEGORIES - labels
        assert not missing, f"{name} missing categories: {missing}"


def test_no_empty_text_fields() -> None:
    for name in ("train.csv", "val.csv", "test.csv"):
        for i, row in enumerate(_read(name)):
            assert row["text"].strip(), f"{name} row {i} has empty text"
