#!/usr/bin/env python3
"""按受限、确定的 dotenv 语义读取单个值，供 POSIX preflight 共用。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
INTERPOLATION_PATTERN = re.compile(r"\$(?:\{|[A-Za-z_])")


def read_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"环境文件第 {line_number} 行缺少等号")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"环境文件第 {line_number} 行变量名无效")
        value = raw_value.strip()
        single_quoted = len(value) >= 2 and value[0] == value[-1] == "'"
        double_quoted = len(value) >= 2 and value[0] == value[-1] == '"'
        if single_quoted or double_quoted:
            value = value[1:-1]
        else:
            # Compose 只在 # 前有空白时把它视为行尾注释。
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if not single_quoted and INTERPOLATION_PATTERN.search(value):
            raise ValueError(f"{key} 禁止使用 dotenv/shell 变量插值，请写入最终字面值")
        values[key] = value
    return values


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("用法：read-env-value.py <deploy/.env> <KEY>")
    try:
        value = read_values(Path(sys.argv[1])).get(sys.argv[2], "")
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"环境文件解析失败：{exc}") from exc
    print(value)


if __name__ == "__main__":
    main()
