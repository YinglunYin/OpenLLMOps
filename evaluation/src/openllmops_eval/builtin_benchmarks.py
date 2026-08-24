"""安全、可复现地准备 C-Eval 与 CMMLU 内置评测集。

这里只读取官方数据文件，不导入或执行上游仓库中的任何 Python 代码。在线来源固定到
不可变 revision 和制品 SHA-256；离线来源则记录管理员声明的 revision 与实际内容指纹。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx

BenchmarkName = Literal["ceval", "cmmlu"]
SourceMode = Literal["online", "offline"]

LICENSE_ID = "CC-BY-NC-SA-4.0"
SPLIT_ORDER = ("dev", "val", "test")
CHOICE_LABELS = ("A", "B", "C", "D")

# 安全上限明显高于两个官方 CSV 数据集的真实规模，但能阻止压缩炸弹和异常输入。
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
MAX_CSV_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_CSV_BYTES = 256 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 500

# 科目名来自下方固定 revision 的官方目录。离线来源即使数量相同，也不能用任意文件名
# 冒充内置数据；新增官方 revision 若改变科目集合，应在代码审查中显式更新。
CEVAL_SUBJECTS = frozenset(
    {
        "accountant",
        "advanced_mathematics",
        "art_studies",
        "basic_medicine",
        "business_administration",
        "chinese_language_and_literature",
        "civil_servant",
        "clinical_medicine",
        "college_chemistry",
        "college_economics",
        "college_physics",
        "college_programming",
        "computer_architecture",
        "computer_network",
        "discrete_mathematics",
        "education_science",
        "electrical_engineer",
        "environmental_impact_assessment_engineer",
        "fire_engineer",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_chinese",
        "high_school_geography",
        "high_school_history",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "ideological_and_moral_cultivation",
        "law",
        "legal_professional",
        "logic",
        "mao_zedong_thought",
        "marxism",
        "metrology_engineer",
        "middle_school_biology",
        "middle_school_chemistry",
        "middle_school_geography",
        "middle_school_history",
        "middle_school_mathematics",
        "middle_school_physics",
        "middle_school_politics",
        "modern_chinese_history",
        "operating_system",
        "physician",
        "plant_protection",
        "probability_and_statistics",
        "professional_tour_guide",
        "sports_science",
        "tax_accountant",
        "teacher_qualification",
        "urban_and_rural_planner",
        "veterinary_medicine",
    }
)

CMMLU_SUBJECTS = frozenset(
    {
        "agronomy",
        "anatomy",
        "ancient_chinese",
        "arts",
        "astronomy",
        "business_ethics",
        "chinese_civil_service_exam",
        "chinese_driving_rule",
        "chinese_food_culture",
        "chinese_foreign_policy",
        "chinese_history",
        "chinese_literature",
        "chinese_teacher_qualification",
        "clinical_knowledge",
        "college_actuarial_science",
        "college_education",
        "college_engineering_hydrology",
        "college_law",
        "college_mathematics",
        "college_medical_statistics",
        "college_medicine",
        "computer_science",
        "computer_security",
        "conceptual_physics",
        "construction_project_management",
        "economics",
        "education",
        "electrical_engineering",
        "elementary_chinese",
        "elementary_commonsense",
        "elementary_information_and_technology",
        "elementary_mathematics",
        "ethnology",
        "food_science",
        "genetics",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_geography",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_politics",
        "human_sexuality",
        "international_law",
        "journalism",
        "jurisprudence",
        "legal_and_moral_basis",
        "logical",
        "machine_learning",
        "management",
        "marketing",
        "marxist_theory",
        "modern_chinese",
        "nutrition",
        "philosophy",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_study",
        "sociology",
        "sports_science",
        "traditional_chinese_medicine",
        "virology",
        "world_history",
        "world_religions",
    }
)


class BenchmarkPreparationError(ValueError):
    """可直接展示给管理员的数据准备错误。"""


class LicenseAcceptanceRequired(BenchmarkPreparationError):
    """在线下载前尚未显式接受非商业数据许可。"""


@dataclass(frozen=True)
class BenchmarkSpec:
    name: BenchmarkName
    title: str
    repository_url: str
    documentation_url: str
    license_url: str
    online_revision: str
    download_url: str
    artifact_sha256: str
    default_splits: tuple[str, ...]
    subjects: frozenset[str]


BENCHMARK_SPECS: dict[BenchmarkName, BenchmarkSpec] = {
    "ceval": BenchmarkSpec(
        name="ceval",
        title="C-Eval",
        repository_url="https://huggingface.co/datasets/ceval/ceval-exam",
        documentation_url="https://github.com/hkust-nlp/ceval",
        license_url=(
            "https://github.com/hkust-nlp/ceval/blob/"
            "cba65ae93bcf189149ced9f66ae0c958201faed9/LICENSE-DATA"
        ),
        online_revision="3923b519fd180e689d0961bf3a032ece929742f3",
        download_url=(
            "https://huggingface.co/datasets/ceval/ceval-exam/resolve/"
            "3923b519fd180e689d0961bf3a032ece929742f3/ceval-exam.zip"
        ),
        artifact_sha256="68786deeea68ff089c56563ee48fab8160da857b77b913437bb504d681fd8e20",
        # 该固定 CSV 快照的 test 不含答案，不能用于准确率计算。
        default_splits=("dev", "val"),
        subjects=CEVAL_SUBJECTS,
    ),
    "cmmlu": BenchmarkSpec(
        name="cmmlu",
        title="CMMLU",
        repository_url="https://github.com/haonan-li/CMMLU",
        documentation_url="https://github.com/haonan-li/CMMLU",
        license_url=(
            "https://github.com/haonan-li/CMMLU/blob/"
            "d6e7b716d8ac694f38969a6c0407437d1fded799/README.md#license"
        ),
        online_revision="d6e7b716d8ac694f38969a6c0407437d1fded799",
        download_url=(
            "https://github.com/haonan-li/CMMLU/archive/"
            "d6e7b716d8ac694f38969a6c0407437d1fded799.zip"
        ),
        artifact_sha256="154593336d5074d793ed990222876b83490b0aed97638a62618d1fe2da7c2cac",
        default_splits=("dev", "test"),
        subjects=CMMLU_SUBJECTS,
    ),
}


@dataclass(frozen=True)
class SourceCsv:
    """来源中的一个 CSV；logical_path 永远是已校验的 POSIX 相对路径。"""

    logical_path: str
    content: bytes
    root_hint: str | None = None


@dataclass(frozen=True)
class PreparedBenchmark:
    jsonl_path: Path
    manifest_path: Path
    record_count: int
    jsonl_sha256: str


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_member_path(raw_name: str) -> str:
    """验证归档成员名，不依赖 extractall 的隐式路径处理。"""

    # `tar -cf archive.tar .` 常产生安全的 `./data/...` 前缀，先做有限规范化。
    while raw_name.startswith("./"):
        raw_name = raw_name[2:]
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise BenchmarkPreparationError(f"归档包含不安全路径: {raw_name!r}")
    raw_parts = raw_name.split("/")
    path = PurePosixPath(raw_name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise BenchmarkPreparationError(f"归档包含不安全路径: {raw_name!r}")
    if re.match(r"^[A-Za-z]:", path.parts[0]):
        raise BenchmarkPreparationError(f"归档包含 Windows 绝对路径: {raw_name!r}")
    return path.as_posix()


def _check_csv_size(logical_path: str, size: int, total: int) -> int:
    if size > MAX_CSV_FILE_BYTES:
        raise BenchmarkPreparationError(f"CSV 文件过大: {logical_path}")
    total += size
    if total > MAX_TOTAL_CSV_BYTES:
        raise BenchmarkPreparationError("CSV 解压后总大小超过安全限制")
    return total


def _read_zip(path: Path) -> list[SourceCsv]:
    files: list[SourceCsv] = []
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise BenchmarkPreparationError("归档成员数量超过安全限制")
            for member in members:
                raw_name = member.filename.rstrip("/")
                if member.is_dir() and raw_name in {"", "."}:
                    continue
                logical_path = _validate_member_path(raw_name)
                unix_mode = member.external_attr >> 16
                if member.flag_bits & 0x1:
                    raise BenchmarkPreparationError(f"不支持加密归档成员: {logical_path}")
                if unix_mode and stat.S_ISLNK(unix_mode):
                    raise BenchmarkPreparationError(f"归档中不允许符号链接: {logical_path}")
                if member.is_dir():
                    continue
                file_type = stat.S_IFMT(unix_mode)
                if file_type and not stat.S_ISREG(unix_mode):
                    raise BenchmarkPreparationError(f"归档中包含特殊文件: {logical_path}")
                if not logical_path.lower().endswith(".csv"):
                    continue
                total = _check_csv_size(logical_path, member.file_size, total)
                has_impossible_size = member.file_size > 0 and member.compress_size == 0
                has_excessive_ratio = (
                    member.compress_size > 0
                    and member.file_size / member.compress_size > MAX_ZIP_COMPRESSION_RATIO
                )
                if has_impossible_size or has_excessive_ratio:
                    raise BenchmarkPreparationError(f"CSV 压缩比异常: {logical_path}")
                with archive.open(member, "r") as source:
                    content = source.read(MAX_CSV_FILE_BYTES + 1)
                if len(content) != member.file_size or len(content) > MAX_CSV_FILE_BYTES:
                    raise BenchmarkPreparationError(f"CSV 文件大小异常: {logical_path}")
                files.append(SourceCsv(logical_path, content))
    except (zipfile.BadZipFile, OSError) as exc:
        raise BenchmarkPreparationError(f"无法读取 ZIP 归档: {path.name}") from exc
    return files


def _read_tar(path: Path) -> list[SourceCsv]:
    files: list[SourceCsv] = []
    total = 0
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise BenchmarkPreparationError("归档成员数量超过安全限制")
            for member in members:
                raw_name = member.name.rstrip("/")
                if member.isdir() and raw_name in {"", "."}:
                    continue
                logical_path = _validate_member_path(raw_name)
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise BenchmarkPreparationError(f"归档中包含链接或特殊文件: {logical_path}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise BenchmarkPreparationError(f"归档中包含未知成员类型: {logical_path}")
                if not logical_path.lower().endswith(".csv"):
                    continue
                total = _check_csv_size(logical_path, member.size, total)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BenchmarkPreparationError(f"无法读取归档成员: {logical_path}")
                content = extracted.read(MAX_CSV_FILE_BYTES + 1)
                if len(content) != member.size or len(content) > MAX_CSV_FILE_BYTES:
                    raise BenchmarkPreparationError(f"CSV 文件大小异常: {logical_path}")
                files.append(SourceCsv(logical_path, content))
    except (tarfile.TarError, OSError) as exc:
        raise BenchmarkPreparationError(f"无法读取 TAR 归档: {path.name}") from exc
    return files


def _read_directory(root: Path) -> list[SourceCsv]:
    files: list[SourceCsv] = []
    total = 0
    seen_entries = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as exc:
            raise BenchmarkPreparationError(f"无法读取来源目录: {current}") from exc
        for entry in entries:
            seen_entries += 1
            if seen_entries > MAX_ARCHIVE_MEMBERS:
                raise BenchmarkPreparationError("来源目录条目数量超过安全限制")
            if entry.is_symlink():
                raise BenchmarkPreparationError(f"来源目录中不允许符号链接: {entry.name}")
            entry_path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                stack.append(entry_path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise BenchmarkPreparationError(f"来源目录中包含特殊文件: {entry.name}")
            relative = entry_path.relative_to(root).as_posix()
            if not relative.lower().endswith(".csv"):
                continue
            size = entry.stat(follow_symlinks=False).st_size
            total = _check_csv_size(relative, size, total)
            try:
                content = entry_path.read_bytes()
            except OSError as exc:
                raise BenchmarkPreparationError(f"无法读取 CSV: {relative}") from exc
            if len(content) != size:
                raise BenchmarkPreparationError(f"读取期间 CSV 大小发生变化: {relative}")
            files.append(SourceCsv(relative, content, root.name.lower()))
    return files


def _read_source(path: Path) -> tuple[list[SourceCsv], str, str]:
    """返回 CSV、来源指纹及指纹范围；归档从不落地解压。"""

    if not path.exists():
        raise BenchmarkPreparationError(f"来源不存在: {path}")
    if path.is_symlink():
        raise BenchmarkPreparationError("来源路径不能是符号链接")
    if path.is_dir():
        files = _read_directory(path)
        return files, _canonical_csv_sha256(files), "canonical_csv_tree"
    if not path.is_file():
        raise BenchmarkPreparationError("来源必须是普通文件或目录")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        raise BenchmarkPreparationError("不接受 pickle 来源；请提供官方 CSV 目录或 ZIP/TAR")
    source_sha256 = _sha256_file(path)
    if zipfile.is_zipfile(path):
        return _read_zip(path), source_sha256, "archive_bytes"
    try:
        is_tar = tarfile.is_tarfile(path)
    except OSError as exc:
        raise BenchmarkPreparationError(f"无法识别来源文件: {path.name}") from exc
    if is_tar:
        return _read_tar(path), source_sha256, "archive_bytes"
    raise BenchmarkPreparationError("来源文件必须是 ZIP、TAR、TAR.GZ 或 TGZ 归档")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkPreparationError(f"无法读取来源文件: {path.name}") from exc
    return digest.hexdigest()


def _canonical_csv_sha256(files: list[SourceCsv]) -> str:
    """对“逻辑路径 + 原始字节”计算确定性 CSV 树指纹。"""

    digest = hashlib.sha256()
    for source in sorted(files, key=lambda item: item.logical_path):
        name = source.logical_path.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(source.content).to_bytes(8, "big"))
        digest.update(source.content)
    return digest.hexdigest()


def _canonical_source_path(spec: BenchmarkSpec, split: str, category: str) -> str:
    filename = f"{category}_{split}.csv" if spec.name == "ceval" else f"{category}.csv"
    return f"{split}/{filename}"


def _benchmark_content_sha256(spec: BenchmarkSpec, files: list[SourceCsv]) -> str:
    """去除仓库顶层包装目录，便于比较目录与归档中的同一官方数据。"""

    normalized: list[SourceCsv] = []
    seen: set[str] = set()
    for source in files:
        split = _detect_split(source)
        if split is None:
            continue
        category = _category(spec, source, split)
        if category is None:
            continue
        logical_path = _canonical_source_path(spec, split, category)
        if logical_path in seen:
            raise BenchmarkPreparationError(f"规范化后的 CSV 路径重复: {logical_path}")
        seen.add(logical_path)
        normalized.append(SourceCsv(logical_path, source.content))
    return _canonical_csv_sha256(normalized)


def _download(spec: BenchmarkSpec, target: Path) -> str:
    digest = hashlib.sha256()
    downloaded = 0
    headers = {"User-Agent": "OpenLLMOps-Evaluation/0.1"}
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client,
            client.stream("GET", spec.download_url) as response,
        ):
            response.raise_for_status()
            with target.open("wb") as output:
                for chunk in response.iter_bytes():
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise BenchmarkPreparationError("在线制品超过下载大小限制")
                    digest.update(chunk)
                    output.write(chunk)
    except (httpx.HTTPError, OSError) as exc:
        raise BenchmarkPreparationError(f"下载 {spec.title} 官方制品失败") from exc
    actual = digest.hexdigest()
    if actual != spec.artifact_sha256:
        raise BenchmarkPreparationError(
            f"官方制品 SHA-256 不匹配：期望 {spec.artifact_sha256}，实际 {actual}"
        )
    return actual


def _detect_split(source: SourceCsv) -> str | None:
    parts = [part.lower() for part in PurePosixPath(source.logical_path).parts[:-1]]
    if source.root_hint in SPLIT_ORDER:
        parts.insert(0, source.root_hint)
    matches = [part for part in parts if part in SPLIT_ORDER]
    if not matches:
        return None
    if len(set(matches)) > 1:
        raise BenchmarkPreparationError(f"CSV 路径包含多个 split: {source.logical_path}")
    return matches[-1]


def _category(spec: BenchmarkSpec, source: SourceCsv, split: str) -> str | None:
    stem = PurePosixPath(source.logical_path).stem
    if spec.name == "ceval":
        suffix = f"_{split}"
        if not stem.lower().endswith(suffix):
            return None
        stem = stem[: -len(suffix)]
    category = stem.strip().lower()
    if not category or not re.fullmatch(r"[a-z0-9_\-]+", category):
        raise BenchmarkPreparationError(f"无法识别科目名: {source.logical_path}")
    return category


def _normalized_row(raw: dict[str | None, str | list[str] | None]) -> dict[str, str]:
    if None in raw:
        raise BenchmarkPreparationError("CSV 行的列数多于表头")
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        assert key is not None
        normalized_key = key.lstrip("\ufeff").strip().lower()
        if not normalized_key:
            # CMMLU 官方 CSV 的第一列是 pandas 导出的无名索引列。
            continue
        if normalized_key in normalized:
            raise BenchmarkPreparationError(f"CSV 包含重复字段: {key}")
        if not isinstance(value, str):
            raise BenchmarkPreparationError(f"CSV 字段缺失: {key}")
        normalized[normalized_key] = value.strip()
    return normalized


def _parse_csv(
    spec: BenchmarkSpec,
    source: SourceCsv,
    split: str,
    category: str,
) -> list[dict[str, object]]:
    canonical_source_path = _canonical_source_path(spec, split, category)
    try:
        text = source.content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BenchmarkPreparationError(f"CSV 不是有效 UTF-8: {source.logical_path}") from exc
    if "\x00" in text:
        raise BenchmarkPreparationError(f"CSV 包含 NUL 字节: {source.logical_path}")

    records: list[dict[str, object]] = []
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames is None:
            raise BenchmarkPreparationError(f"CSV 缺少表头: {source.logical_path}")
        for source_row, raw_row in enumerate(reader, start=2):
            row = _normalized_row(raw_row)
            question = row.get("question", "")
            answer = row.get("answer", "").upper()
            choices = {label: row.get(label.lower(), "") for label in CHOICE_LABELS}
            if not question:
                raise BenchmarkPreparationError(
                    f"{source.logical_path}:{source_row} 缺少有效 question"
                )
            if any(not value for value in choices.values()):
                raise BenchmarkPreparationError(
                    f"{source.logical_path}:{source_row} 缺少 A/B/C/D 选项"
                )
            if answer not in CHOICE_LABELS:
                detail = (
                    "；该固定 C-Eval CSV 的 test split 未发布答案"
                    if (spec.name == "ceval" and split == "test")
                    else ""
                )
                raise BenchmarkPreparationError(
                    f"{source.logical_path}:{source_row} 缺少有效 answer{detail}"
                )
            metadata: dict[str, object] = {
                "benchmark": spec.name,
                "source_file": canonical_source_path,
                "source_row": source_row,
                "split": split,
            }
            if original_id := row.get("id", ""):
                metadata["original_id"] = original_id
            if explanation := row.get("explanation", ""):
                metadata["explanation"] = explanation
            records.append(
                {
                    "answer": answer,
                    "category": category,
                    "choices": choices,
                    "id": f"{spec.name}:{split}:{category}:{source_row - 1}",
                    "metadata": metadata,
                    "question": question,
                    "task_type": "multiple_choice",
                }
            )
    except csv.Error as exc:
        raise BenchmarkPreparationError(f"CSV 格式错误: {source.logical_path}") from exc
    if not records:
        raise BenchmarkPreparationError(f"CSV 不含样本: {source.logical_path}")
    return records


def _normalize_splits(spec: BenchmarkSpec, splits: tuple[str, ...] | None) -> tuple[str, ...]:
    selected = set(splits or spec.default_splits)
    invalid = selected.difference(SPLIT_ORDER)
    if invalid:
        raise BenchmarkPreparationError(f"不支持的 split: {', '.join(sorted(invalid))}")
    if not selected:
        raise BenchmarkPreparationError("至少选择一个 split")
    return tuple(split for split in SPLIT_ORDER if split in selected)


def _convert(
    spec: BenchmarkSpec,
    files: list[SourceCsv],
    splits: tuple[str, ...],
    allow_partial: bool,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int], int, bool]:
    selected_files: list[tuple[str, str, SourceCsv]] = []
    seen: set[tuple[str, str]] = set()
    category_sets: dict[str, set[str]] = {split: set() for split in splits}
    for source in files:
        split = _detect_split(source)
        if split not in splits:
            continue
        category = _category(spec, source, split)
        if category is None:
            continue
        identity = (split, category)
        if identity in seen:
            raise BenchmarkPreparationError(f"split/科目 CSV 重复: {split}/{category}")
        seen.add(identity)
        category_sets[split].add(category)
        selected_files.append((split, category, source))

    if not selected_files:
        raise BenchmarkPreparationError("来源中没有找到所选 split 的官方 CSV")

    missing_splits = [split for split, categories in category_sets.items() if not categories]
    if missing_splits:
        raise BenchmarkPreparationError(f"所选 split 未找到任何 CSV: {', '.join(missing_splits)}")

    unknown_categories = set().union(*category_sets.values()).difference(spec.subjects)
    if unknown_categories:
        raise BenchmarkPreparationError(
            f"来源包含非官方科目名: {', '.join(sorted(unknown_categories))}"
        )

    partial = any(categories != spec.subjects for categories in category_sets.values())
    first_categories = category_sets[splits[0]]
    aligned = all(categories == first_categories for categories in category_sets.values())
    partial = partial or not aligned
    if partial and not allow_partial:
        detail = ", ".join(f"{split}={len(category_sets[split])}" for split in splits)
        raise BenchmarkPreparationError(
            f"科目集合不完整或 split 间不一致（期望每个 split {len(spec.subjects)} 科；{detail}）。"
            "如确为测试子集，请显式使用 --allow-partial"
        )

    order = {split: index for index, split in enumerate(SPLIT_ORDER)}
    selected_files.sort(key=lambda item: (order[item[0]], item[1], item[2].logical_path))
    records: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for split, category, source in selected_files:
        converted = _parse_csv(spec, source, split, category)
        records.extend(converted)
        split_counts[split] += len(converted)
        category_counts[category] += len(converted)
    return (
        records,
        dict(sorted(split_counts.items())),
        dict(sorted(category_counts.items())),
        len(selected_files),
        partial,
    )


def _jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    lines = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_atomic(path: Path, content: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(content)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise BenchmarkPreparationError(f"无法写入输出文件: {path}") from exc


def prepare_builtin_benchmark(
    benchmark: BenchmarkName,
    output_dir: Path,
    *,
    online: bool = False,
    source: Path | None = None,
    source_revision: str | None = None,
    accept_noncommercial_license: bool = False,
    splits: tuple[str, ...] | None = None,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> PreparedBenchmark:
    """从固定在线制品或管理员提供的官方来源生成 JSONL 与 manifest。"""

    if benchmark not in BENCHMARK_SPECS:
        raise BenchmarkPreparationError("benchmark 仅支持 ceval 或 cmmlu")
    spec = BENCHMARK_SPECS[benchmark]
    if online == (source is not None):
        raise BenchmarkPreparationError("必须且只能选择 online 或 source 之一")
    if online and not accept_noncommercial_license:
        raise LicenseAcceptanceRequired(
            f"{spec.title} 使用 {LICENSE_ID} 非商业许可；在线下载前必须显式接受许可"
        )
    if online and source_revision is not None:
        raise BenchmarkPreparationError("在线模式 revision 已固定，不能传入 source_revision")
    if source is not None and (source_revision is None or not source_revision.strip()):
        raise BenchmarkPreparationError("离线模式必须用 source_revision 记录管理员提供的版本")

    selected_splits = _normalize_splits(spec, splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{benchmark}.jsonl"
    manifest_path = output_dir / f"{benchmark}.manifest.json"
    if not overwrite and (jsonl_path.exists() or manifest_path.exists()):
        raise BenchmarkPreparationError("输出已存在；确认替换时请显式使用 --overwrite")

    mode: SourceMode
    input_name: str
    revision: str
    revision_verified: bool
    with tempfile.TemporaryDirectory(prefix="openllmops-benchmark-") as temporary_dir:
        if online:
            mode = "online"
            downloaded = Path(temporary_dir) / "official-artifact"
            source_sha256 = _download(spec, downloaded)
            files, reread_sha256, sha256_scope = _read_source(downloaded)
            if reread_sha256 != source_sha256:
                raise BenchmarkPreparationError("下载后读取制品时 SHA-256 发生变化")
            input_name = spec.download_url.rsplit("/", 1)[-1]
            revision = spec.online_revision
            revision_verified = True
        else:
            assert source is not None and source_revision is not None
            mode = "offline"
            files, source_sha256, sha256_scope = _read_source(source)
            input_name = source.name
            revision = source_revision.strip()
            revision_verified = False

    content_sha256 = _benchmark_content_sha256(spec, files)
    records, split_counts, category_counts, source_file_count, partial = _convert(
        spec, files, selected_splits, allow_partial
    )
    jsonl_content = _jsonl_bytes(records)
    jsonl_sha256 = _sha256_bytes(jsonl_content)
    manifest = {
        "benchmark": benchmark,
        "conversion": {
            "allow_partial": allow_partial,
            "format": "openllmops-eval-jsonl-v1",
            "partial": partial,
            "source_file_count": source_file_count,
            "splits": list(selected_splits),
        },
        "dataset_license": {
            "accepted_noncommercial_by_cli": accept_noncommercial_license,
            "id": LICENSE_ID,
            "url": spec.license_url,
        },
        "output": {
            "category_counts": category_counts,
            "path": jsonl_path.name,
            "record_count": len(records),
            "sha256": jsonl_sha256,
            "split_counts": split_counts,
        },
        "schema_version": 1,
        "source": {
            "content_sha256": content_sha256,
            "input_name": input_name,
            "mode": mode,
            "repository": spec.repository_url,
            "revision": revision,
            "revision_verified": revision_verified,
            "sha256": source_sha256,
            "sha256_scope": sha256_scope,
            "url": spec.download_url if online else None,
        },
    }
    manifest_content = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_atomic(jsonl_path, jsonl_content)
    try:
        _write_atomic(manifest_path, manifest_content)
    except BenchmarkPreparationError:
        # 避免只留下无 manifest 的半成品；旧文件只会在 overwrite 明确开启时被替换。
        jsonl_path.unlink(missing_ok=True)
        raise
    return PreparedBenchmark(jsonl_path, manifest_path, len(records), jsonl_sha256)
