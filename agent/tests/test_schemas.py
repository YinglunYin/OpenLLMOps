from uuid import uuid4

import pytest
from pydantic import ValidationError

from openllmops_agent.schemas import InferenceLaunchRequest


def test_inference_request_rejects_duplicate_gpu() -> None:
    with pytest.raises(ValidationError):
        InferenceLaunchRequest(
            deployment_id=uuid4(),
            image="registry/vllm:test",
            gpu_ids=[0, 0],
            model_path="/srv/openllmops/models/demo",
            served_model_name="demo",
        )


def test_inference_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InferenceLaunchRequest(
            deployment_id=uuid4(),
            image="registry/vllm:test",
            gpu_ids=[0],
            model_path="/srv/openllmops/models/demo",
            served_model_name="demo",
            privileged=True,
        )
