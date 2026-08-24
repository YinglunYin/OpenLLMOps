from __future__ import annotations

import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from openllmops_eval import builtin_benchmarks
from openllmops_eval.builtin_benchmarks import (
    BenchmarkPreparationError,
    LicenseAcceptanceRequired,
    prepare_builtin_benchmark,
)
from openllmops_eval.dataset import load_jsonl


def _write_ceval_csv(path: Path, *, answer: str = "B", explanation: str = "因为乙正确") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,question,A,B,C,D,answer,explanation\n"
        f"0,哪一项正确？,甲,乙,丙,丁,{answer},{explanation}\n",
        encoding="utf-8",
    )


def _cmmlu_csv(question: str, answer: str) -> str:
    # 官方 CMMLU CSV 保留了 pandas 导出的无名索引列，并使用首字母大写字段名。
    return f",Question,A,B,C,D,Answer\n0,{question},甲,乙,丙,丁,{answer}\n"


def test_prepare_ceval_directory_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "official-ceval"
    _write_ceval_csv(source / "dev" / "computer_network_dev.csv")
    _write_ceval_csv(
        source / "val" / "computer_network_val.csv",
        answer="A",
        explanation="",
    )

    first = prepare_builtin_benchmark(
        "ceval",
        tmp_path / "first",
        source=source,
        source_revision="synthetic-official-layout",
        splits=("dev", "val"),
        allow_partial=True,
    )
    second = prepare_builtin_benchmark(
        "ceval",
        tmp_path / "second",
        source=source,
        source_revision="synthetic-official-layout",
        splits=("val", "dev"),
        allow_partial=True,
    )

    assert first.jsonl_path.read_bytes() == second.jsonl_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    samples, fingerprint = load_jsonl(first.jsonl_path)
    assert fingerprint == first.jsonl_sha256
    assert [sample.sample_id for sample in samples] == [
        "ceval:dev:computer_network:1",
        "ceval:val:computer_network:1",
    ]
    assert samples[0].answers == ("B",)
    assert samples[0].metadata["original_id"] == "0"
    assert samples[0].metadata["explanation"] == "因为乙正确"

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["sha256_scope"] == "canonical_csv_tree"
    assert manifest["source"]["revision_verified"] is False
    assert manifest["output"]["record_count"] == 2
    assert manifest["output"]["split_counts"] == {"dev": 1, "val": 1}
    assert manifest["conversion"]["partial"] is True
    assert manifest["output"]["sha256"] == fingerprint


def test_prepare_cmmlu_uppercase_headers_from_zip(tmp_path: Path) -> None:
    source = tmp_path / "CMMLU-fixed.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "CMMLU-fixed/data/dev/chinese_history.csv",
            _cmmlu_csv("秦统一六国发生在哪一时期？", "C"),
        )
        archive.writestr(
            "CMMLU-fixed/data/test/chinese_history.csv",
            _cmmlu_csv("唐朝都城是哪座城市？", "A"),
        )

    result = prepare_builtin_benchmark(
        "cmmlu",
        tmp_path / "output",
        source=source,
        source_revision="fixed",
        allow_partial=True,
    )

    rows = [json.loads(line) for line in result.jsonl_path.read_text().splitlines()]
    assert [row["metadata"]["split"] for row in rows] == ["dev", "test"]
    assert rows[0]["answer"] == "C"
    assert rows[0]["choices"] == {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["sha256_scope"] == "archive_bytes"
    assert len(manifest["source"]["sha256"]) == 64
    assert len(manifest["source"]["content_sha256"]) == 64

    # 同一 CSV 从解压目录导入时，包装目录变化不应改变统一 JSONL 或内容指纹。
    directory = tmp_path / "CMMLU-directory"
    dev = directory / "data" / "dev" / "chinese_history.csv"
    test = directory / "data" / "test" / "chinese_history.csv"
    dev.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    dev.write_text(_cmmlu_csv("秦统一六国发生在哪一时期？", "C"), encoding="utf-8")
    test.write_text(_cmmlu_csv("唐朝都城是哪座城市？", "A"), encoding="utf-8")
    directory_result = prepare_builtin_benchmark(
        "cmmlu",
        tmp_path / "directory-output",
        source=directory,
        source_revision="fixed",
        allow_partial=True,
    )
    directory_manifest = json.loads(directory_result.manifest_path.read_text(encoding="utf-8"))
    assert result.jsonl_path.read_bytes() == directory_result.jsonl_path.read_bytes()
    assert manifest["source"]["content_sha256"] == directory_manifest["source"]["content_sha256"]


def test_incomplete_subjects_require_explicit_partial_flag(tmp_path: Path) -> None:
    source = tmp_path / "ceval"
    _write_ceval_csv(source / "dev" / "law_dev.csv")

    with pytest.raises(BenchmarkPreparationError, match="--allow-partial"):
        prepare_builtin_benchmark(
            "ceval",
            tmp_path / "output",
            source=source,
            source_revision="fixture",
            splits=("dev",),
        )


def test_partial_source_still_rejects_non_official_subject(tmp_path: Path) -> None:
    source = tmp_path / "cmmlu"
    csv_path = source / "dev" / "invented_subject.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(_cmmlu_csv("问题", "A"), encoding="utf-8")

    with pytest.raises(BenchmarkPreparationError, match="非官方科目名"):
        prepare_builtin_benchmark(
            "cmmlu",
            tmp_path / "output",
            source=source,
            source_revision="fixture",
            splits=("dev",),
            allow_partial=True,
        )


def test_zip_path_traversal_is_rejected_without_extracting(tmp_path: Path) -> None:
    source = tmp_path / "malicious.zip"
    escaped = tmp_path / "escape.csv"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("../escape.csv", _cmmlu_csv("问题", "A"))

    with pytest.raises(BenchmarkPreparationError, match="不安全路径"):
        prepare_builtin_benchmark(
            "cmmlu",
            tmp_path / "output",
            source=source,
            source_revision="untrusted",
            allow_partial=True,
        )
    assert not escaped.exists()


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("CMMLU/data/dev/law.csv")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(link, "../../outside.csv")

    with pytest.raises(BenchmarkPreparationError, match="符号链接"):
        prepare_builtin_benchmark(
            "cmmlu",
            tmp_path / "output",
            source=source,
            source_revision="untrusted",
            allow_partial=True,
        )


def test_tar_with_common_dot_prefix_is_supported(tmp_path: Path) -> None:
    source = tmp_path / "cmmlu.tar.gz"
    content = _cmmlu_csv("问题", "B").encode()
    member = tarfile.TarInfo("./data/dev/college_law.csv")
    member.size = len(content)
    with tarfile.open(source, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(content))

    result = prepare_builtin_benchmark(
        "cmmlu",
        tmp_path / "output",
        source=source,
        source_revision="fixture",
        splits=("dev",),
        allow_partial=True,
    )
    row = json.loads(result.jsonl_path.read_text(encoding="utf-8"))
    assert row["id"] == "cmmlu:dev:college_law:1"


def test_pickle_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "data.pkl"
    source.write_bytes(b"not-even-a-pickle")

    with pytest.raises(BenchmarkPreparationError, match="pickle"):
        prepare_builtin_benchmark(
            "ceval",
            tmp_path / "output",
            source=source,
            source_revision="untrusted",
            allow_partial=True,
        )


def test_online_requires_license_acceptance_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> str:
        nonlocal called
        called = True
        raise AssertionError("不应发起网络请求")

    monkeypatch.setattr(builtin_benchmarks, "_download", fail_if_called)
    with pytest.raises(LicenseAcceptanceRequired, match="显式接受"):
        prepare_builtin_benchmark("ceval", tmp_path / "output", online=True)
    assert called is False


def test_unlabeled_ceval_test_split_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "ceval"
    _write_ceval_csv(source / "test" / "law_test.csv", answer="")

    with pytest.raises(BenchmarkPreparationError, match="test split 未发布答案"):
        prepare_builtin_benchmark(
            "ceval",
            tmp_path / "output",
            source=source,
            source_revision="fixture",
            splits=("test",),
            allow_partial=True,
        )


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "cmmlu"
    csv_path = source / "dev" / "college_law.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(_cmmlu_csv("问题", "D"), encoding="utf-8")
    output = tmp_path / "output"
    prepare_builtin_benchmark(
        "cmmlu",
        output,
        source=source,
        source_revision="fixture",
        splits=("dev",),
        allow_partial=True,
    )

    with pytest.raises(BenchmarkPreparationError, match="--overwrite"):
        prepare_builtin_benchmark(
            "cmmlu",
            output,
            source=source,
            source_revision="fixture",
            splits=("dev",),
            allow_partial=True,
        )
