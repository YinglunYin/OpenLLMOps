from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_soft_delete_migration():  # type: ignore[no-untyped-def]
    migration_path = (
        Path(__file__).resolve().parents[1] / "migrations/versions/d82f6a1e093b_soft_delete_model_assets.py"
    )
    spec = importlib.util.spec_from_file_location("soft_delete_migration", migration_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_soft_delete_migration_refuses_lossy_downgrade(monkeypatch: pytest.MonkeyPatch) -> None:
    migration = _load_soft_delete_migration()

    class BindWithTombstone:
        @staticmethod
        def scalar(_statement) -> int:  # type: ignore[no-untyped-def]
            return 1

    monkeypatch.setattr(migration.op, "get_bind", lambda: BindWithTombstone())
    with pytest.raises(RuntimeError, match="拒绝移除 deleted_at"):
        migration.downgrade()
