import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from openllmops_agent.evaluation_runtime import (
    DatasetSource,
    EvaluationInputError,
    load_dataset_manifest_summary,
    load_pair_report_metadata,
    prepare_evaluation_workspace,
    strict_existing_path,
)


def _roots(tmp_path: Path) -> dict[str, Path]:
    roots = {
        "dataset_root": tmp_path / "datasets",
        "evaluation_dataset_root": tmp_path / "evaluation-datasets",
        "evaluation_output_root": tmp_path / "evaluation-output",
        "runtime_root": tmp_path / "runtime",
    }
    for path in roots.values():
        path.mkdir()
    return roots


def _row(sample_id: str, question: str) -> str:
    return json.dumps(
        {
            "id": sample_id,
            "question": question,
            "choices": {"A": "否", "B": "是"},
            "answer": "B",
        },
        ensure_ascii=False,
    )


def _pair_report(dataset_sha256: str) -> dict:
    baseline = {
        "dataset_sha256": dataset_sha256,
        "model_name": "baseline",
        "template": "base",
        "total": 2,
        "correct": 1,
        "invalid": 0,
        "accuracy_percent": 50.0,
        "average_latency_ms": 10.0,
        "categories": [
            {
                "category": "domain/default",
                "total": 2,
                "correct": 1,
                "invalid": 0,
                "accuracy_percent": 50.0,
            }
        ],
        "sample_ids": ["domain:1", "domain:2"],
    }
    candidate = {
        **baseline,
        "model_name": "candidate",
        "template": "instruct",
        "correct": 2,
        "accuracy_percent": 100.0,
        "average_latency_ms": 12.0,
        "categories": [
            {
                "category": "domain/default",
                "total": 2,
                "correct": 2,
                "invalid": 0,
                "accuracy_percent": 100.0,
            }
        ],
    }
    return {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "dataset_sha256": dataset_sha256,
            "baseline_model": "baseline",
            "candidate_model": "candidate",
            "baseline_percent": 50.0,
            "candidate_percent": 100.0,
            "percentage_point_change": 50.0,
            "relative_change_percent": 100.0,
            "comparable": True,
            "reason": None,
            "category_changes": [
                {
                    "category": "domain/default",
                    "baseline_percent": 50.0,
                    "candidate_percent": 100.0,
                    "percentage_point_change": 50.0,
                }
            ],
        },
    }


def test_multiple_datasets_are_merged_deterministically_with_source_fingerprints(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    first = roots["dataset_root"] / "first.jsonl"
    second = roots["dataset_root"] / "second.jsonl"
    first.write_text(_row("1", "问题一") + "\n", encoding="utf-8")
    second.write_text(_row("2", "问题二") + "\n", encoding="utf-8")

    run_id = uuid4()
    workspace = prepare_evaluation_workspace(
        run_id=run_id,
        generation=1,
        # 输入逆序，产物仍必须按 name/path 确定性排序。
        sources=[DatasetSource("second", second), DatasetSource("first", first)],
        requested_output_path=roots["evaluation_output_root"] / str(run_id),
        **roots,
    )

    lines = [json.loads(line) for line in workspace.dataset_path.read_text().splitlines()]
    assert [line["id"] for line in lines] == ["first:1", "second:2"]
    assert [line["category"] for line in lines] == ["first/default", "second/default"]
    manifest = json.loads(workspace.dataset_manifest_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in manifest["sources"]] == ["first", "second"]
    assert manifest["combined"]["sha256"] == hashlib.sha256(workspace.dataset_path.read_bytes()).hexdigest()
    assert load_dataset_manifest_summary(workspace.dataset_manifest_path) == (
        manifest["combined"]["sha256"],
        2,
    )


def test_evaluation_paths_reject_symlinks_escape_and_reused_output(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    outside = tmp_path / "outside.jsonl"
    outside.write_text(_row("1", "outside") + "\n", encoding="utf-8")
    linked = roots["dataset_root"] / "linked.jsonl"
    linked.symlink_to(outside)

    with pytest.raises(EvaluationInputError, match="软链接"):
        strict_existing_path(linked, (roots["dataset_root"],), directory=False)

    run_id = uuid4()
    valid = roots["dataset_root"] / "valid.jsonl"
    valid.write_text(_row("1", "valid") + "\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="系统派生"):
        prepare_evaluation_workspace(
            run_id=run_id,
            generation=1,
            sources=[DatasetSource("domain", valid)],
            requested_output_path=tmp_path / str(run_id),
            **roots,
        )

    output = roots["evaluation_output_root"] / str(run_id)
    output.mkdir()
    (output / "old-result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="非空"):
        prepare_evaluation_workspace(
            run_id=run_id,
            generation=1,
            sources=[DatasetSource("domain", valid)],
            requested_output_path=output,
            **roots,
        )


def test_pair_report_is_sanitized_and_bound_to_manifest_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    digest = "a" * 64
    (output / "pair-report.json").write_text(
        json.dumps(_pair_report(digest), ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = load_pair_report_metadata(
        output,
        expected_dataset_sha256=digest,
        expected_total=2,
    )

    assert metadata["comparison"]["percentage_point_change"] == 50.0
    assert "sample_ids" not in metadata["metrics"]["baseline"]
    with pytest.raises(EvaluationInputError, match="指纹"):
        load_pair_report_metadata(output, expected_dataset_sha256="b" * 64, expected_total=2)


def test_pair_report_rejects_nonfinite_and_inconsistent_comparison(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / "pair-report.json"
    report_path.write_text('{"baseline":NaN,"candidate":{},"comparison":{}}', encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="非有限"):
        load_pair_report_metadata(output)

    report = _pair_report("c" * 64)
    report["comparison"]["percentage_point_change"] = 99.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="汇总不一致"):
        load_pair_report_metadata(output)


def test_pair_report_rejects_category_denominator_or_template_drift(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    report_path = output / "pair-report.json"
    report = _pair_report("d" * 64)
    report["baseline"].update(
        {
            "total": 3,
            "correct": 1,
            "accuracy_percent": 33.3333,
            "sample_ids": ["domain:1", "domain:2", "domain:3"],
            "categories": [
                {
                    "category": "domain/a",
                    "total": 2,
                    "correct": 1,
                    "invalid": 0,
                    "accuracy_percent": 50.0,
                },
                {
                    "category": "domain/b",
                    "total": 1,
                    "correct": 0,
                    "invalid": 0,
                    "accuracy_percent": 0.0,
                },
            ],
        }
    )
    report["candidate"].update(
        {
            "total": 3,
            "correct": 2,
            "accuracy_percent": 66.6667,
            "sample_ids": ["domain:1", "domain:2", "domain:3"],
            "categories": [
                {
                    "category": "domain/a",
                    "total": 1,
                    "correct": 1,
                    "invalid": 0,
                    "accuracy_percent": 100.0,
                },
                {
                    "category": "domain/b",
                    "total": 2,
                    "correct": 1,
                    "invalid": 0,
                    "accuracy_percent": 50.0,
                },
            ],
        }
    )
    report["comparison"].update(
        {
            "baseline_percent": 33.3333,
            "candidate_percent": 66.6667,
            "percentage_point_change": 33.3334,
            "relative_change_percent": 100.0003,
            "category_changes": [
                {
                    "category": "domain/a",
                    "baseline_percent": 50.0,
                    "candidate_percent": 100.0,
                    "percentage_point_change": 50.0,
                },
                {
                    "category": "domain/b",
                    "baseline_percent": 0.0,
                    "candidate_percent": 50.0,
                    "percentage_point_change": 50.0,
                },
            ],
        }
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="同名分类的样本数"):
        load_pair_report_metadata(output)

    valid = _pair_report("d" * 64)
    report_path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="baseline 模板"):
        load_pair_report_metadata(output, expected_base_template="instruct")


def test_merge_rejects_category_that_would_exceed_report_contract(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    dataset = roots["dataset_root"] / "long-category.jsonl"
    row = json.loads(_row("1", "question"))
    row["category"] = "x" * 192
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    run_id = uuid4()

    with pytest.raises(EvaluationInputError, match="category"):
        prepare_evaluation_workspace(
            run_id=run_id,
            generation=1,
            sources=[DatasetSource("s" * 64, dataset)],
            requested_output_path=roots["evaluation_output_root"] / str(run_id),
            **roots,
        )
