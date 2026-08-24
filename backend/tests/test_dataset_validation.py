import io
import json
import os
from pathlib import Path

import pytest
from openllmops_eval.dataset import DatasetValidationError, load_jsonl, parse_row
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.routes.datasets import upload_dataset
from app.core.config import get_settings
from app.models.enums import DatasetType
from app.services import dataset_files
from app.services.dataset_files import validate_and_store_jsonl


def test_startup_cleanup_only_removes_old_strict_upload_parts(tmp_path: Path) -> None:
    now = 2_000_000_000.0
    old_part = tmp_path / ".123e4567-e89b-42d3-a456-426614174000.jsonl.part"
    recent_part = tmp_path / ".123e4567-e89b-42d3-a456-426614174001.jsonl.part"
    unrelated = tmp_path / ".not-an-openllmops-upload.jsonl.part"
    target = tmp_path / "operator-data.jsonl"
    symlink = tmp_path / ".123e4567-e89b-42d3-a456-426614174002.jsonl.part"
    for path in (old_part, recent_part, unrelated, target):
        path.write_bytes(b"{}\n")
    symlink.symlink_to(target)
    os.utime(old_part, (now - 90_000, now - 90_000))
    os.utime(recent_part, (now - 60, now - 60))
    os.utime(unrelated, (now - 90_000, now - 90_000))

    removed = dataset_files.cleanup_stale_upload_parts(
        tmp_path,
        older_than_seconds=86_400,
        now=now,
    )

    assert removed == 1
    assert not old_part.exists()
    assert recent_part.exists()
    assert unrelated.exists()
    assert symlink.is_symlink()
    assert target.exists()


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


def test_oversized_physical_line_is_read_and_drained_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundedReader(io.BytesIO):
        requested_sizes: list[int]

        def __init__(self, value: bytes) -> None:
            super().__init__(value)
            self.requested_sizes = []

        def readline(self, size: int = -1, /) -> bytes:
            self.requested_sizes.append(size)
            assert 0 < size <= 17
            return super().readline(size)

    monkeypatch.setattr(dataset_files, "MAX_LINE_BYTES", 16)
    monkeypatch.setattr(dataset_files, "MAX_DATASET_BYTES", 1024)
    source = BoundedReader(b'{"text":"' + b"x" * 200 + b'"}\n')
    temporary = tmp_path / ".oversized-physical-line.part"

    with pytest.raises(ValueError, match="单行超过 16"):
        validate_and_store_jsonl(
            source,
            temporary,
            tmp_path / "oversized-physical-line.jsonl",
            DatasetType.CPT,
        )

    assert source.requested_sizes and max(source.requested_sizes) == 17
    assert not temporary.exists()


def test_schema_summary_has_bounded_field_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataset_files, "MAX_SCHEMA_FIELDS", 4)
    rows = b"".join(
        json.dumps(
            {"instruction": "Q", "output": "A", f"extra_{index}": index},
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for index in range(5)
    )

    with pytest.raises(ValueError, match="字段种类超过 4"):
        validate_and_store_jsonl(
            io.BytesIO(rows),
            tmp_path / ".too-many-fields.part",
            tmp_path / "too-many-fields.jsonl",
            DatasetType.SFT,
        )


@pytest.mark.asyncio
async def test_upload_close_failure_removes_atomically_stored_orphan(
    isolated_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    class CloseFailureUpload:
        filename = "close-failure.jsonl"
        file = io.BytesIO(b'{"instruction":"Q","output":"A"}\n')

        @staticmethod
        async def close() -> None:
            raise RuntimeError("模拟临时上传文件关闭失败")

    dataset_root = get_settings().dataset_root
    before = {path.name for path in dataset_root.glob("*.jsonl")}
    async with isolated_session_factory() as session:
        with pytest.raises(RuntimeError, match="关闭失败"):
            await upload_dataset(
                name="close-failure",
                dataset_type=DatasetType.SFT,
                version="v1.0.0",
                description=None,
                file=CloseFailureUpload(),  # type: ignore[arg-type]
                session=session,
            )
    assert {path.name for path in dataset_root.glob("*.jsonl")} == before
