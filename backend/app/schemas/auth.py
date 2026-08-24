from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: SecretStr = Field(min_length=1, max_length=1024)


class AdminIdentityRead(BaseModel):
    username: str
    auth_method: Literal["session", "bootstrap_key", "disabled"]
    expires_at: datetime | None = None
    csrf_token: str | None = None
