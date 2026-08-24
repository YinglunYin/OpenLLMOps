"""动态模型导入容器入口。"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .importer import ImportRequest, ModelImporter, ModelSource


def _required_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"请求字段 {name} 必须是非空路径")
    return Path(value)


def _load_request(path: Path) -> tuple[ImportRequest, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("导入请求必须是 JSON 对象")
    token: str | None = None
    token_file = payload.get("access_token_file")
    if token_file is not None:
        # 密钥通过只读 secret 文件注入，不能出现在命令行、请求日志或结果文件里。
        token = _required_path(token_file, "access_token_file").read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("访问令牌文件为空")
    request = ImportRequest(
        import_id=uuid.UUID(str(payload["import_id"])),
        source=ModelSource(str(payload["source"])),
        repository=str(payload["repository"]) if payload.get("repository") else None,
        revision=str(payload["revision"]) if payload.get("revision") else None,
        source_directory=(
            _required_path(payload["source_directory"], "source_directory")
            if payload.get("source_directory")
            else None
        ),
        access_token=token,
    )
    return request, payload


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenLLMOps 模型导入执行器")
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--inbox-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args()

    request, _ = _load_request(args.request_file)
    importer = ModelImporter(
        inbox_root=args.inbox_root,
        staging_root=args.staging_root,
        store_root=args.store_root,
    )

    def progress(stage: str, completed: int, total: int | None) -> None:
        # 每行独立 JSON，Node Agent 可实时转发且不会因半行日志造成解析歧义。
        print(
            json.dumps(
                {"event": "progress", "stage": stage, "completed": completed, "total": total},
                ensure_ascii=False,
            ),
            flush=True,
        )

    final, manifest = importer.run(
        request,
        progress=progress,
        cancelled=(lambda: args.cancel_file.exists()) if args.cancel_file else None,
    )
    _write_json_atomically(
        args.result_file,
        {
            "import_id": str(request.import_id),
            "status": "ready",
            "local_path": str(final),
            "manifest": manifest.as_dict(),
        },
    )


if __name__ == "__main__":
    main()
