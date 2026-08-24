import io
import json
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.api.routes import datasets as dataset_routes
from app.api.routes import openai_gateway
from app.core.config import get_settings
from app.core.database import AsyncSessionFactory
from app.models import Deployment
from app.models.enums import DeploymentState


def create_ready_model(
    seed_model_asset: Callable[..., dict[str, str]],
    suffix: str = "base",
    *,
    kind: str = "instruct",
) -> dict[str, str]:
    model_path = get_settings().model_root / str(uuid.uuid4())
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    (model_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
    return seed_model_asset(
        model_path,
        kind=kind,
        name=f"Qwen demo {suffix}",
    )


def mark_deployment_running(client: TestClient, deployment_id: str, internal_url: str) -> None:
    """测试专用：模拟 Reconciler 已把 vLLM 实例协调到 running。"""

    async def persist() -> None:
        async with AsyncSessionFactory() as session, session.begin():
            deployment = await session.get(Deployment, uuid.UUID(deployment_id))
            assert deployment is not None
            deployment.actual_state = DeploymentState.RUNNING
            deployment.internal_url = internal_url

    assert client.portal is not None
    client.portal.call(persist)


def test_health_capabilities_and_removed_responses(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["status"] == "ready"

    capabilities = client.get("/api/v1/system/capabilities").json()
    assert capabilities["gpu_policy"] == "exclusive_non_preemptive"
    assert "/v1/responses" not in capabilities["openai_endpoints"]
    assert "/v1/responses" not in client.get("/openapi.json").json()["paths"]
    removed = client.post("/v1/responses", json={"model": "missing"})
    assert removed.status_code == 404
    assert "error" in removed.json() and "detail" not in removed.json()
    missing = client.post("/v1/completions", json={"model": "missing", "prompt": "hello"})
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "model_not_found"
    invalid = client.post("/v1/completions", json={})
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    # 模型与数据集只能经过受控导入/上传管线，不能直接伪造 READY 记录。
    assert client.post("/api/v1/model-assets", json={}).status_code == 405
    assert client.post("/api/v1/datasets", json={}).status_code == 405


def test_openai_gateway_routes_generation_embedding_and_streaming(
    client: TestClient,
    seed_model_asset: Callable[..., dict[str, str]],
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    generation_model = create_ready_model(seed_model_asset, "gateway-generation")
    embedding_model = create_ready_model(seed_model_asset, "gateway-embedding", kind="embedding")
    generation_alias = f"generation-{uuid.uuid4()}"
    embedding_alias = f"embedding-{uuid.uuid4()}"

    def create_deployment(model: dict[str, str], alias: str, task_type: str) -> dict:
        response = client.post(
            "/api/v1/deployments",
            json={
                "name": f"gateway-{uuid.uuid4()}",
                "served_model_name": alias,
                "model_asset_id": model["id"],
                "task_type": task_type,
                "gpu_ids": [0],
                "tensor_parallel_size": 1,
            },
        )
        assert response.status_code == 201, response.text
        return response.json()

    generation = create_deployment(generation_model, generation_alias, "generate")
    embedding = create_deployment(embedding_model, embedding_alias, "embedding")
    mark_deployment_running(client, generation["id"], "http://generation-vllm:8000")
    mark_deployment_running(client, embedding["id"], "http://embedding-vllm:8000")

    observed: list[tuple[str, dict, str | None]] = []

    class UpstreamEventStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield b"data: [DONE]\n\n"

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed.append((request.url.path, payload, request.headers.get("authorization")))
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"index": 0, "embedding": [0.1, 0.2]}]},
                headers={"x-request-id": "embedding-request"},
            )
        if payload.get("stream"):
            return httpx.Response(
                200,
                stream=UpstreamEventStream(),
                headers={"content-type": "text/event-stream", "x-request-id": "stream-request"},
            )
        return httpx.Response(
            200,
            json={"id": "cmpl-test", "object": "text_completion", "choices": [{"text": "ok"}]},
            headers={"x-request-id": "completion-request"},
        )

    monkeypatch.setattr(
        openai_gateway,
        "_build_proxy_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler)),
    )

    completion = client.post(
        "/v1/completions",
        json={"model": generation_alias, "prompt": "hello"},
    )
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"] == "ok"
    # 对外 request id 由网关统一生成，不能被上游同名响应头覆盖。
    assert completion.headers["x-request-id"] != "completion-request"

    stream = client.post(
        "/v1/chat/completions",
        json={"model": generation_alias, "messages": [{"role": "user", "content": "hello"}], "stream": True},
    )
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in stream.text

    embeddings = client.post(
        "/v1/embeddings",
        json={"model": embedding_alias, "input": ["hello"]},
    )
    assert embeddings.status_code == 200
    assert embeddings.json()["data"][0]["embedding"] == [0.1, 0.2]

    # 同名部署只允许其任务类型对应的端点，避免把生成请求误投到 Embedding 进程。
    assert (
        client.post(
            "/v1/completions",
            json={"model": embedding_alias, "prompt": "wrong type"},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/embeddings",
            json={"model": generation_alias, "input": "wrong type"},
        ).status_code
        == 404
    )
    assert [item[0] for item in observed] == [
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
    ]


def test_model_asset_and_deployment_lifecycle(
    client: TestClient,
    seed_model_asset: Callable[..., dict[str, str]],
) -> None:
    model = create_ready_model(seed_model_asset, "deployment")
    response = client.post(
        "/api/v1/deployments",
        json={
            "name": f"deployment-{uuid.uuid4()}",
            "served_model_name": f"qwen-demo-{uuid.uuid4()}",
            "model_asset_id": model["id"],
            "task_type": "generate",
            "gpu_ids": [0, 1],
            "tensor_parallel_size": 2,
            "simplified_config": {"max_model_len": 4096},
            "vllm_args": {"gpu_memory_utilization": 0.9},
        },
    )
    assert response.status_code == 201, response.text
    deployment = response.json()
    assert deployment["desired_state"] == "running"
    assert deployment["actual_state"] == "queued"
    assert deployment["queued_at"] is not None
    assert deployment["health_status"] is None
    assert deployment["started_at"] is None
    assert deployment["error_message"] is None
    # 控制面响应只公开网关语义，不暴露容器网络地址或内部监听端口。
    assert "internal_url" not in deployment and "port" not in deployment

    start = client.post(f"/api/v1/deployments/{deployment['id']}/start")
    assert start.status_code == 200
    assert start.json()["actual_state"] == "queued"
    assert client.patch(f"/api/v1/deployments/{deployment['id']}", json={"port": 8101}).status_code == 422

    stop = client.post(f"/api/v1/deployments/{deployment['id']}/stop")
    assert stop.status_code == 200
    assert stop.json()["actual_state"] == "stopped"
    forbidden_arg = client.patch(
        f"/api/v1/deployments/{deployment['id']}",
        json={"vllm_args": {"trust_remote_code": True}},
    )
    assert forbidden_arg.status_code == 422

    invalid_simple = client.post(
        "/api/v1/deployments",
        json={
            "name": f"invalid-memory-{uuid.uuid4()}",
            "served_model_name": f"invalid-memory-{uuid.uuid4()}",
            "model_asset_id": model["id"],
            "task_type": "generate",
            "gpu_ids": [0],
            "tensor_parallel_size": 1,
            "simplified_config": {"gpu_memory_utilization": 0.99},
        },
    )
    assert invalid_simple.status_code == 422

    base_payload = {
        "model_asset_id": model["id"],
        "task_type": "generate",
        "gpu_ids": [0],
        "tensor_parallel_size": 1,
    }
    for alias in ("中文模型", "qwen demo", "bad\nname"):
        rejected_alias = client.post(
            "/api/v1/deployments",
            json={
                **base_payload,
                "name": f"invalid-alias-{uuid.uuid4()}",
                "served_model_name": alias,
            },
        )
        assert rejected_alias.status_code == 422

    for invalid_args in (
        {"made_up_flag": True},
        {"max_num_seqs": "64"},
        {"gpu_memory_utilization": 1.0},
    ):
        rejected_args = client.post(
            "/api/v1/deployments",
            json={
                **base_payload,
                "name": f"invalid-args-{uuid.uuid4()}",
                "served_model_name": f"invalid-args-{uuid.uuid4()}",
                "vllm_args": invalid_args,
            },
        )
        assert rejected_args.status_code == 422

    valid_alias = client.post(
        "/api/v1/deployments",
        json={
            **base_payload,
            "name": f"valid-alias-{uuid.uuid4()}",
            "served_model_name": "qwen2.5/demo:7b",
            "vllm_args": {"enable_prefix_caching": True, "max_num_seqs": 64},
        },
    )
    assert valid_alias.status_code == 201

    # 即使部署已经停止，历史记录仍引用该版本；删除模型必须显式拒绝，
    # 不能依赖数据库外键报出难以理解的 500。
    referenced_delete = client.delete(f"/api/v1/model-assets/{model['id']}")
    assert referenced_delete.status_code == 409
    assert "引用" in referenced_delete.json()["detail"]


def test_model_asset_soft_delete_hides_record_but_keeps_files(
    client: TestClient,
    seed_model_asset: Callable[..., dict[str, str]],
) -> None:
    model = create_ready_model(seed_model_asset, f"soft-delete-{uuid.uuid4()}")
    model_path = get_settings().model_root / Path(model["local_path"]).name
    assert model_path.exists()

    # 资产状态、路径等受控字段不能通过 PATCH 绕开导入校验。
    rejected_patch = client.patch(
        f"/api/v1/model-assets/{model['id']}",
        json={"status": "ready", "local_path": "/tmp/forged"},
    )
    assert rejected_patch.status_code == 422

    models_before = client.get("/api/v1/dashboard/summary").json()["models"]["total"]
    deleted = client.delete(f"/api/v1/model-assets/{model['id']}")
    assert deleted.status_code == 204
    assert model_path.exists()
    assert client.get(f"/api/v1/model-assets/{model['id']}").status_code == 404
    assert model["id"] not in {item["id"] for item in client.get("/api/v1/model-assets").json()}
    dashboard = client.get("/api/v1/dashboard/summary").json()
    assert dashboard["models"]["total"] == models_before - 1
    assert model["id"] not in {item["resource_id"] for item in dashboard["recent_activity"]}
    create_with_deleted = client.post(
        "/api/v1/deployments",
        json={
            "name": f"deleted-model-{uuid.uuid4()}",
            "served_model_name": f"deleted-model-{uuid.uuid4()}",
            "model_asset_id": model["id"],
            "task_type": "generate",
            "gpu_ids": [0],
            "tensor_parallel_size": 1,
        },
    )
    assert create_with_deleted.status_code == 404
    assert client.delete(f"/api/v1/model-assets/{model['id']}").status_code == 404


def test_dataset_upload_preview_and_training_queue(
    client: TestClient,
    seed_model_asset: Callable[..., dict[str, str]],
) -> None:
    payload = b'{"instruction":"say hello","output":"hello"}\n{"instruction":"2+2","output":"4"}\n'
    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": "demo-sft", "dataset_type": "sft", "version": "v2.1.0"},
        files={"file": ("demo.jsonl", io.BytesIO(payload), "application/jsonl")},
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["record_count"] == 2
    assert dataset["status"] == "ready"
    assert dataset["schema_summary"]["version"] == "v2.1.0"
    preview = client.get(f"/api/v1/datasets/{dataset['id']}/preview?limit=1")
    assert preview.status_code == 200
    assert len(preview.json()) == 1

    model = create_ready_model(seed_model_asset, "training")
    training = client.post(
        "/api/v1/training-jobs",
        json={
            "name": f"sft-{uuid.uuid4()}",
            "model_asset_id": model["id"],
            "dataset_id": dataset["id"],
            "stage": "sft",
            "algorithm": "qlora",
            "gpu_ids": [0],
            "training_config": {"num_train_epochs": 1, "template": "qwen"},
        },
    )
    assert training.status_code == 201, training.text
    assert training.json()["actual_state"] == "queued"
    assert training.json()["output_dir"] == str(get_settings().checkpoint_root / training.json()["id"])

    terminated = client.post(f"/api/v1/training-jobs/{training.json()['id']}/terminate")
    assert terminated.status_code == 200
    assert terminated.json()["actual_state"] == "canceled"

    invalid_cpt = client.post(
        "/api/v1/training-jobs",
        json={
            "name": f"cpt-{uuid.uuid4()}",
            "model_asset_id": model["id"],
            "dataset_id": dataset["id"],
            "stage": "cpt",
            "algorithm": "freeze",
            "gpu_ids": [0],
        },
    )
    assert invalid_cpt.status_code == 422


def test_dataset_upload_rejects_unsafe_filename_and_removes_unregistered_file(
    client: TestClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    payload = b'{"instruction":"Q","output":"A"}\n'
    too_long = client.post(
        "/api/v1/datasets/upload",
        data={"name": "unsafe-name", "dataset_type": "sft"},
        files={"file": (f"{'x' * 250}.jsonl", io.BytesIO(payload), "application/jsonl")},
    )
    assert too_long.status_code == 422

    before = {path.name for path in get_settings().dataset_root.glob("*.jsonl")}

    async def reject_registration(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise dataset_routes.HTTPException(status_code=409, detail="模拟数据库登记失败")

    monkeypatch.setattr(dataset_routes, "commit_or_conflict", reject_registration)
    failed = client.post(
        "/api/v1/datasets/upload",
        data={"name": "orphan-cleanup", "dataset_type": "sft"},
        files={"file": ("valid.jsonl", io.BytesIO(payload), "application/jsonl")},
    )
    assert failed.status_code == 409
    assert {path.name for path in get_settings().dataset_root.glob("*.jsonl")} == before


def test_api_key_plaintext_is_only_returned_once(client: TestClient) -> None:
    name = f"playground-{uuid.uuid4()}"
    created = client.post("/api/v1/api-keys", json={"name": name})
    assert created.status_code == 201, created.text
    assert created.json()["key"].startswith("ollm_")

    listed = client.get("/api/v1/api-keys").json()
    matching = next(item for item in listed if item["name"] == name)
    assert "key" not in matching
    assert "key_hash" not in matching


def test_evaluation_rejects_embedding_models(
    client: TestClient,
    seed_model_asset: Callable[..., dict[str, str]],
) -> None:
    embedding = create_ready_model(
        seed_model_asset,
        f"embedding-{uuid.uuid4()}",
        kind="embedding",
    )
    candidate = create_ready_model(seed_model_asset, "evaluation-candidate")

    response = client.post(
        "/api/v1/evaluation-runs",
        json={
            "name": f"evaluation-{uuid.uuid4()}",
            "base_model_asset_id": embedding["id"],
            "candidate_model_asset_id": candidate["id"],
            "builtin_datasets": ["ceval"],
            "gpu_ids": [0],
        },
    )
    assert response.status_code == 422
    assert "Embedding" in response.json()["detail"]


def test_admin_and_issued_api_key_authentication(client: TestClient) -> None:
    settings = get_settings()
    previous_enabled = settings.auth_enabled
    previous_admin_key = settings.admin_api_key
    settings.auth_enabled = True
    settings.admin_api_key = "test-admin-key"
    try:
        assert client.get("/api/v1/system/capabilities").status_code == 401
        openai_unauthorized = client.post(
            "/v1/completions",
            json={"model": "not-deployed", "prompt": "hello"},
        )
        assert openai_unauthorized.status_code == 401
        assert openai_unauthorized.json()["error"]["type"] == "authentication_error"
        admin_headers = {settings.api_key_header: "test-admin-key"}
        assert client.get("/api/v1/system/capabilities", headers=admin_headers).status_code == 200

        created = client.post(
            "/api/v1/api-keys",
            headers=admin_headers,
            json={"name": f"authenticated-{uuid.uuid4()}"},
        )
        assert created.status_code == 201, created.text
        issued_headers = {settings.api_key_header: created.json()["key"]}
        # 普通推理 Key 不能进入管理面，但仍可通过 OpenAI 网关鉴权。
        assert client.get("/api/v1/system/capabilities", headers=issued_headers).status_code == 401
        assert (
            client.post(
                "/v1/completions",
                headers=issued_headers,
                json={"model": "not-deployed", "prompt": "hello"},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/v1/api-keys",
                headers=issued_headers,
                json={"name": f"forbidden-{uuid.uuid4()}"},
            ).status_code
            == 401
        )
    finally:
        settings.auth_enabled = previous_enabled
        settings.admin_api_key = previous_admin_key
