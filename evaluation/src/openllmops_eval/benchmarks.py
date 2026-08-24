"""把 C-Eval/CMMLU 常见 CSV 布局转换为平台统一 JSONL。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def convert_csv_directory(source_dir: Path, output_path: Path, benchmark: str) -> int:
    """转换本地已取得的数据，不在运行时隐式联网下载许可数据。"""

    if benchmark not in {"ceval", "cmmlu"}:
        raise ValueError("benchmark 仅支持 ceval 或 cmmlu")
    count = 0
    with output_path.open("w", encoding="utf-8") as target:
        for csv_path in sorted(source_dir.rglob("*.csv")):
            category = csv_path.stem
            with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
                for row_number, row in enumerate(csv.DictReader(source), start=1):
                    question = row.get("question", "").strip()
                    answer = row.get("answer", "").strip().upper()
                    if not question or answer not in {"A", "B", "C", "D"}:
                        raise ValueError(f"{csv_path}:{row_number} 缺少有效 question/answer")
                    record = {
                        "id": f"{benchmark}:{category}:{row_number}",
                        "task_type": "multiple_choice",
                        "category": category,
                        "question": question,
                        "choices": {label: row.get(label, "") for label in ("A", "B", "C", "D")},
                        "answer": answer,
                        "metadata": {"benchmark": benchmark, "source_file": csv_path.name},
                    }
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    if count == 0:
        raise ValueError("来源目录中没有可转换的 CSV 样本")
    return count

