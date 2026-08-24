import io
from pathlib import Path

import pytest

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
