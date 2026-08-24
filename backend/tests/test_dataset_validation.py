import io
import json
from pathlib import Path

import pytest
from openllmops_eval.dataset import DatasetValidationError, load_jsonl, parse_row

from app.models.enums import DatasetType
from app.services.dataset_files import validate_and_store_jsonl


def test_invalid_sft_jsonl_is_not_persisted(tmp_path: Path) -> None:
    temporary = tmp_path / ".invalid.part"
    final = tmp_path / "invalid.jsonl"

    with pytest.raises(ValueError, match="instruction"):
        validate_and_store_jsonl(
            io.BytesIO(b'{"text":"not an sft record"}\n'),
            temporary,
            final,
            DatasetType.SFT,
        )

    assert not temporary.exists()
    assert not final.exists()


def test_evaluation_jsonl_uses_execution_runtime_contract(tmp_path: Path) -> None:
    rows = [
        {
            "id": "mcq-1",
            "task_type": "multiple_choice",
            "question": "2+2 等于多少？",
            "choices": {"A": "3", "B": "4"},
            "answer": "B",
            "category": "math",
        },
        {
            "id": "classification-1",
            "task_type": "classification",
            "question": "这条评论很满意",
            "answer": "positive",
        },
        {
            "id": "qa-1",
            "task_type": "short_qa",
            "question": "中国的首都是哪里？",
            "answers": ["北京", "北京市"],
            "context": "中国地理常识",
        },
    ]
    raw = b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows) + b"\n\n"
    temporary = tmp_path / ".evaluation.part"
    final = tmp_path / "evaluation.jsonl"

    count, size, sha256, errors, summary = validate_and_store_jsonl(
        io.BytesIO(raw),
        temporary,
        final,
        DatasetType.EVALUATION,
    )
    runtime_samples, runtime_sha256 = load_jsonl(final)
    assert count == len(runtime_samples) == 3
    assert size == len(raw)
    assert errors == []
    assert sha256 == runtime_sha256
    assert {sample.task_type.value for sample in runtime_samples} == {
        "multiple_choice",
        "classification",
        "short_qa",
    }
    assert summary["format"] == "jsonl"


@pytest.mark.parametrize(
    "row",
    [
        {"input": "旧格式", "label": "positive"},
        {"task_type": "multiple_choice", "question": "缺选项", "answer": "A"},
        {"task_type": "short_qa", "question": "缺答案"},
        {"task_type": "unsupported", "question": "问题", "answer": "答案"},
    ],
)
def test_invalid_evaluation_vectors_fail_in_control_plane_and_runtime(
    tmp_path: Path,
    row: dict,
) -> None:
    with pytest.raises(DatasetValidationError):
        parse_row(row, 1)
    with pytest.raises(ValueError):
        validate_and_store_jsonl(
            io.BytesIO(json.dumps(row, ensure_ascii=False).encode() + b"\n"),
            tmp_path / ".invalid-evaluation.part",
            tmp_path / "invalid-evaluation.jsonl",
            DatasetType.EVALUATION,
        )


def test_evaluation_duplicate_ids_and_runtime_line_limit_are_rejected(tmp_path: Path) -> None:
    duplicate = (
        b'{"id":"same","task_type":"short_qa","question":"Q1","answer":"A1"}\n'
        b'{"id":"same","task_type":"short_qa","question":"Q2","answer":"A2"}\n'
    )
    with pytest.raises(ValueError, match="样本 ID 重复"):
        validate_and_store_jsonl(
            io.BytesIO(duplicate),
            tmp_path / ".duplicate.part",
            tmp_path / "duplicate.jsonl",
            DatasetType.EVALUATION,
        )

    oversized = {
        "task_type": "short_qa",
        "question": "Q",
        "answer": "A",
        "context": "x" * 1_048_576,
    }
    with pytest.raises(ValueError, match="1048576"):
        validate_and_store_jsonl(
            io.BytesIO(json.dumps(oversized).encode() + b"\n"),
            tmp_path / ".oversized.part",
            tmp_path / "oversized.jsonl",
            DatasetType.EVALUATION,
        )


@pytest.mark.parametrize("field", ["id", "category"])
def test_evaluation_source_prefix_fields_are_limited_to_191_chars(
    tmp_path: Path,
    field: str,
) -> None:
    row = {
        "id": "sample",
        "task_type": "short_qa",
        "question": "问题",
        "answer": "答案",
        "category": "domain",
        field: "x" * 192,
    }
    with pytest.raises(ValueError, match="191"):
        validate_and_store_jsonl(
            io.BytesIO(json.dumps(row, ensure_ascii=False).encode() + b"\n"),
            tmp_path / f".{field}.part",
            tmp_path / f"{field}.jsonl",
            DatasetType.EVALUATION,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"id":"one","id":"two","task_type":"short_qa","question":"Q","answer":"A"}\n',
        b'{"task_type":"short_qa","question":"Q","answer":"A","metadata":{"score":NaN}}\n',
        b'{"task_type":"short_qa","question":"Q","answer":"A","category":123}\n',
        (b'{"task_type":"short_qa","question":"Q","answer":"A","metadata":{"openllmops_source":{}}}\n'),
    ],
)
def test_evaluation_rejects_ambiguous_or_reserved_json(
    tmp_path: Path,
    raw: bytes,
) -> None:
    with pytest.raises(ValueError):
        validate_and_store_jsonl(
            io.BytesIO(raw),
            tmp_path / ".unsafe.part",
            tmp_path / "unsafe.jsonl",
            DatasetType.EVALUATION,
        )


def test_training_dataset_requires_one_consistent_record_format(tmp_path: Path) -> None:
    mixed_sft = (
        b'{"instruction":"Q1","output":"A1"}\n'
        b'{"messages":[{"role":"user","content":"Q2"},{"role":"assistant","content":"A2"}]}\n'
    )
    with pytest.raises(ValueError, match="全文件一致"):
        validate_and_store_jsonl(
            io.BytesIO(mixed_sft),
            tmp_path / ".mixed.part",
            tmp_path / "mixed.jsonl",
            DatasetType.SFT,
        )

    mixed_cpt = b'{"text":"one"}\n{"content":"two"}\n'
    with pytest.raises(ValueError, match="全文件一致"):
        validate_and_store_jsonl(
            io.BytesIO(mixed_cpt),
            tmp_path / ".mixed-cpt.part",
            tmp_path / "mixed-cpt.jsonl",
            DatasetType.CPT,
        )
