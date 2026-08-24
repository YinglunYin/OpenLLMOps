import io
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import get_settings


def create_ready_model(client: TestClient, suffix: str = "base") -> dict:
    response = client.post(
        "/api/v1/model-assets",
        json={
            "name": f"Qwen demo {suffix}",
            "source_type": "manual",
            "local_path": f"/srv/openllmops/models/{uuid.uuid4()}",
            "model_kind": "instruct",
            "status": "ready",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_health_capabilities_and_removed_responses(client: TestClient) -> None:
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["status"] == "ready"

    capabilities = client.get("/api/v1/system/capabilities").json()
    assert capabilities["gpu_policy"] == "exclusive_non_preemptive"
    assert "/v1/responses" not in capabilities["openai_endpoints"]
    assert "/v1/responses" not in client.get("/openapi.json").json()["paths"]
    assert client.post("/v1/responses", json={"model": "missing"}).status_code == 404


def test_model_asset_and_deployment_lifecycle(client: TestClient) -> None:
    model = create_ready_model(client, "deployment")
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
    assert deployment["actual_state"] == "created"

    start = client.post(f"/api/v1/deployments/{deployment['id']}/start")
    assert start.status_code == 200
    assert start.json()["actual_state"] == "queued"
    assert client.patch(f"/api/v1/deployments/{deployment['id']}", json={"port": 8101}).status_code == 409

    stop = client.post(f"/api/v1/deployments/{deployment['id']}/stop")
    assert stop.status_code == 200
    assert stop.json()["actual_state"] == "stopped"
    forbidden_arg = client.patch(
        f"/api/v1/deployments/{deployment['id']}",
        json={"vllm_args": {"trust_remote_code": True}},
    )
    assert forbidden_arg.status_code == 422


def test_dataset_upload_preview_and_training_queue(
    client: TestClient,
    test_root: Path,
) -> None:
    payload = (
        b'{"instruction":"say hello","output":"hello"}\n'
        b'{"messages":[{"role":"user","content":"2+2"},'
        b'{"role":"assistant","content":"4"}]}\n'
    )
    response = client.post(
        "/api/v1/datasets/upload",
        data={"name": "demo-sft", "dataset_type": "sft"},
        files={"file": ("demo.jsonl", io.BytesIO(payload), "application/jsonl")},
    )
    assert response.status_code == 201, response.text
    dataset = response.json()
    assert dataset["record_count"] == 2
    assert dataset["status"] == "ready"
    preview = client.get(f"/api/v1/datasets/{dataset['id']}/preview?limit=1")
    assert preview.status_code == 200
    assert len(preview.json()) == 1

    model = create_ready_model(client, "training")
    output_dir = test_root / "checkpoints" / str(uuid.uuid4())
    training = client.post(
        "/api/v1/training-jobs",
        json={
            "name": f"sft-{uuid.uuid4()}",
            "model_asset_id": model["id"],
            "dataset_id": dataset["id"],
            "stage": "sft",
            "algorithm": "qlora",
            "gpu_ids": [0],
            "training_config": {"num_train_epochs": 1},
            "output_dir": str(output_dir),
        },
    )
    assert training.status_code == 201, training.text
    assert training.json()["actual_state"] == "queued"

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
            "output_dir": str(output_dir),
        },
    )
    assert invalid_cpt.status_code == 422


def test_api_key_plaintext_is_only_returned_once(client: TestClient) -> None:
    name = f"playground-{uuid.uuid4()}"
    created = client.post("/api/v1/api-keys", json={"name": name})
    assert created.status_code == 201, created.text
    assert created.json()["key"].startswith("ollm_")

    listed = client.get("/api/v1/api-keys").json()
    matching = next(item for item in listed if item["name"] == name)
    assert "key" not in matching
    assert "key_hash" not in matching


def test_evaluation_rejects_embedding_models(client: TestClient) -> None:
    embedding = client.post(
        "/api/v1/model-assets",
        json={
            "name": f"embedding-{uuid.uuid4()}",
            "source_type": "manual",
            "local_path": f"/srv/openllmops/models/{uuid.uuid4()}",
            "model_kind": "embedding",
            "status": "ready",
        },
    )
    assert embedding.status_code == 201, embedding.text
    candidate = create_ready_model(client, "evaluation-candidate")

    response = client.post(
        "/api/v1/evaluation-runs",
        json={
            "name": f"evaluation-{uuid.uuid4()}",
            "base_model_asset_id": embedding.json()["id"],
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
