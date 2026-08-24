from __future__ import annotations

import json
from pathlib import Path

import pytest

from openllmops_eval.dataset import DatasetValidationError, load_jsonl, render_prompt


def test_load_multiple_choice_dataset(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "history-1",
                "task_type": "multiple_choice",
                "category": "history",
                "question": "唐朝的建立者是谁？",
                "choices": {"A": "李渊", "B": "李世民", "C": "朱元璋", "D": "赵匡胤"},
                "answer": "A",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    samples, fingerprint = load_jsonl(path)

    assert len(samples) == 1
    assert len(fingerprint) == 64
    assert "只输出正确选项的字母" in render_prompt(samples[0], "instruct")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    row = {"id": "same", "question": "1+1?", "choices": ["1", "2"], "answer": "B"}
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join(json.dumps(row) for _ in range(2)), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="样本 ID 重复"):
        load_jsonl(path)

