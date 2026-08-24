#!/usr/bin/env python3
"""在 GitHub Smart HTTP 不可达时，通过官方 Git Data API 发布一个本地提交。

脚本只在远端分支尚不存在时创建引用，并要求 GitHub 返回的 tree/commit SHA
与本地完全一致。GitHub 不允许向完全空仓库写 Git Data 对象时，脚本会先创建
一个可追溯的引导提交，再用合并提交连接本地历史，全程不做强制覆盖。
凭证从 Git Credential Helper 读取，仅保存在当前进程内且不会打印。
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import os
import subprocess
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class Identity:
    name: str
    email: str
    date: str


@dataclass(frozen=True, slots=True)
class CommitMetadata:
    tree: str
    parents: tuple[str, ...]
    author: Identity
    committer: Identity
    message: str


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], text=text)


def credential() -> str:
    completed = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    token = values.get("password")
    if not token:
        raise SystemExit("系统 Git Credential Helper 中没有 github.com 凭证")
    return token


def _identity(raw: str) -> Identity:
    # Git 原始格式为 “姓名 <邮箱> Unix秒 时区”；从右侧拆分可保留姓名中的空格。
    person, timestamp, offset = raw.rsplit(" ", 2)
    name, email_part = person.rsplit(" <", 1)
    email = email_part.removesuffix(">")
    sign = 1 if offset[0] == "+" else -1
    timezone = dt.timezone(
        sign * dt.timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5]))
    )
    date = dt.datetime.fromtimestamp(int(timestamp), timezone).isoformat()
    return Identity(name=name, email=email, date=date)


def commit_metadata(commit: str) -> CommitMetadata:
    raw = git("cat-file", "-p", commit)
    assert isinstance(raw, str)
    headers, message = raw.split("\n\n", 1)
    values: dict[str, list[str]] = {}
    for line in headers.splitlines():
        key, value = line.split(" ", 1)
        values.setdefault(key, []).append(value)
    if "gpgsig" in values or "encoding" in values:
        raise SystemExit("API 回退推送暂不支持签名或非 UTF-8 提交")
    return CommitMetadata(
        tree=values["tree"][0],
        parents=tuple(values.get("parent", [])),
        author=_identity(values["author"][0]),
        committer=_identity(values["committer"][0]),
        message=message,
    )


def tracked_blobs(commit: str) -> list[tuple[str, str, str]]:
    raw = git("ls-tree", "-r", "-z", commit, text=False)
    assert isinstance(raw, bytes)
    entries: list[tuple[str, str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, path_bytes = item.split(b"\t", 1)
        mode, object_type, sha = metadata.decode().split(" ")
        if object_type != "blob":
            raise SystemExit(f"不支持的 Git 对象类型: {object_type}")
        entries.append((mode, sha, path_bytes.decode("utf-8")))
    return entries


def request(client: httpx.Client, method: str, path: str, **kwargs: object) -> dict:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        # 只展示 GitHub 的非敏感错误摘要，绝不输出请求头。
        message = response.json().get("message", response.text[:300])
        raise SystemExit(f"GitHub API {method} {path} 失败 ({response.status_code}): {message}")
    return response.json()


def identity_payload(identity: Identity) -> dict[str, str]:
    return {"name": identity.name, "email": identity.email, "date": identity.date}


def create_local_commit(metadata: CommitMetadata) -> str:
    """用结构化元数据写入本地 Git 对象；不会移动分支或工作区。"""

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": metadata.author.name,
            "GIT_AUTHOR_EMAIL": metadata.author.email,
            "GIT_AUTHOR_DATE": metadata.author.date,
            "GIT_COMMITTER_NAME": metadata.committer.name,
            "GIT_COMMITTER_EMAIL": metadata.committer.email,
            "GIT_COMMITTER_DATE": metadata.committer.date,
        }
    )
    command = ["git", "commit-tree", metadata.tree]
    for parent in metadata.parents:
        command.extend(["-p", parent])
    completed = subprocess.run(
        command,
        input=metadata.message,
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return completed.stdout.strip()


def materialize_bootstrap_commit(
    bootstrap: dict,
    *,
    content: bytes,
    filename: str,
    author_timezone_hint: str,
    committer_timezone_hint: str,
) -> str:
    """把 Contents API 的单文件引导提交重建到本地对象库。"""

    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        input=content,
        capture_output=True,
        check=True,
    ).stdout.decode().strip()
    expected_blob = bootstrap["content"]["sha"]
    if blob != expected_blob:
        raise SystemExit(f"引导 Blob SHA 校验失败，本地 {blob}，远端 {expected_blob}")

    tree = subprocess.run(
        ["git", "mktree"],
        input=f"100644 blob {blob}\t{filename}\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    commit = bootstrap["commit"]
    expected_tree = commit["tree"]["sha"]
    if tree != expected_tree:
        raise SystemExit(f"引导 Tree SHA 校验失败，本地 {tree}，远端 {expected_tree}")

    metadata = CommitMetadata(
        tree=tree,
        parents=tuple(parent["sha"] for parent in commit["parents"]),
        author=Identity(
            name=commit["author"]["name"],
            email=commit["author"]["email"],
            date=_date_with_original_timezone(commit["author"]["date"], author_timezone_hint),
        ),
        committer=Identity(
            name=commit["committer"]["name"],
            email=commit["committer"]["email"],
            date=_date_with_original_timezone(
                commit["committer"]["date"], committer_timezone_hint
            ),
        ),
        message=commit["message"],
    )
    local_bootstrap = create_local_commit(metadata)
    if local_bootstrap != commit["sha"]:
        raise SystemExit(
            f"引导 Commit SHA 校验失败，本地 {local_bootstrap}，远端 {commit['sha']}"
        )
    return local_bootstrap


def _date_with_original_timezone(api_date: str, hint_date: str) -> str:
    """GitHub API 把时间统一显示为 UTC，但提交对象仍保留调用方时区。"""

    instant = dt.datetime.fromisoformat(api_date.replace("Z", "+00:00"))
    hint_timezone = dt.datetime.fromisoformat(hint_date).tzinfo
    if hint_timezone is None:
        raise SystemExit("本地提交时间缺少时区")
    return instant.astimezone(hint_timezone).isoformat()


def upload_commit(
    client: httpx.Client, api_root: str, commit: str, metadata: CommitMetadata
) -> None:
    result = request(
        client,
        "POST",
        f"{api_root}/git/commits",
        json={
            "message": metadata.message,
            "tree": metadata.tree,
            "parents": list(metadata.parents),
            "author": identity_payload(metadata.author),
            "committer": identity_payload(metadata.committer),
        },
    )
    if result["sha"] != commit:
        raise SystemExit(f"Commit SHA 校验失败，本地 {commit}，远端 {result['sha']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", help="owner/name")
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--remote-name",
        default="origin",
        help="发布成功后更新对应的本地 remote-tracking ref",
    )
    parser.add_argument(
        "--create-private",
        action="store_true",
        help="目标不存在时，在当前凭证账号下创建同名私有仓库",
    )
    args = parser.parse_args()

    local_commit = str(git("rev-parse", args.commit)).strip()
    metadata = commit_metadata(local_commit)
    owner_repo = args.repository.strip("/")
    api_root = f"/repos/{owner_repo}"
    headers = {
        "Authorization": f"Bearer {credential()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=120) as client:
        repository_response = client.get(api_root)
        if repository_response.status_code == 404 and args.create_private:
            current_user = request(client, "GET", "/user")["login"]
            owner, repository_name = owner_repo.split("/", 1)
            if current_user.casefold() != owner.casefold():
                raise SystemExit(
                    f"当前凭证属于 {current_user}，不能在账号 {owner} 下创建仓库"
                )
            repository = request(
                client,
                "POST",
                "/user/repos",
                json={
                    "name": repository_name,
                    "description": "单机多卡 NVIDIA 大模型服务综合管理系统",
                    "private": True,
                    "auto_init": False,
                },
            )
            if repository["full_name"].casefold() != owner_repo.casefold() or not repository["private"]:
                raise SystemExit("仓库创建结果不符合预期，已停止上传")
            print(f"已创建私有仓库 {repository['full_name']}")
        elif repository_response.status_code >= 400:
            message = repository_response.json().get("message", "unknown error")
            raise SystemExit(
                f"读取仓库失败 ({repository_response.status_code}): {message}"
            )
        ref_response = client.get(f"{api_root}/git/ref/heads/{args.branch}")
        bootstrap_filename = ".openllmops-bootstrap"
        bootstrap_content = b"OpenLLMOps API bootstrap\n"
        bootstrap_commit: str | None = None
        existing_remote_commit: str | None = None
        if ref_response.status_code == 200:
            remote_commit = ref_response.json()["object"]["sha"]
            if remote_commit == local_commit:
                print(f"远端 {args.branch} 已指向 {local_commit}")
                return
            is_fast_forward = (
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", remote_commit, local_commit],
                    check=False,
                ).returncode
                == 0
            )
            if is_fast_forward:
                existing_remote_commit = remote_commit
            else:
                remote_api_commit = request(
                    client, "GET", f"{api_root}/git/commits/{remote_commit}"
                )
                remote_tree = request(
                    client,
                    "GET",
                    f"{api_root}/git/trees/{remote_api_commit['tree']['sha']}?recursive=1",
                )["tree"]
                is_our_bootstrap = (
                    remote_api_commit["message"]
                    == "chore: bootstrap repository for Git Data API"
                    and len(remote_tree) == 1
                    and remote_tree[0]["path"] == bootstrap_filename
                    and remote_tree[0]["type"] == "blob"
                )
                if not is_our_bootstrap:
                    raise SystemExit(
                        f"远端分支已存在且指向 {remote_commit}，为避免覆盖已停止"
                    )
                bootstrap_commit = materialize_bootstrap_commit(
                    {
                        "content": {"sha": remote_tree[0]["sha"]},
                        "commit": remote_api_commit,
                    },
                    content=bootstrap_content,
                    filename=bootstrap_filename,
                    author_timezone_hint=metadata.author.date,
                    committer_timezone_hint=metadata.committer.date,
                )
        repository_is_empty = (
            ref_response.status_code == 409
            and ref_response.json().get("message") == "Git Repository is empty."
        )
        if ref_response.status_code not in {200, 404} and not repository_is_empty:
            raise SystemExit(f"读取远端分支失败: HTTP {ref_response.status_code}")

        # GitHub 对完全空仓库的 Git Data API 返回 409。先用 Contents API 建立
        # 一次最小提交，随后把它作为第二父提交合并，确保没有历史被强制覆盖。
        if repository_is_empty:
            bootstrap_result = request(
                client,
                "PUT",
                f"{api_root}/contents/{bootstrap_filename}",
                json={
                    "message": "chore: bootstrap repository for Git Data API",
                    "content": base64.b64encode(bootstrap_content).decode(),
                    "branch": args.branch,
                    "author": {
                        "name": metadata.author.name,
                        "email": metadata.author.email,
                    },
                    "committer": {
                        "name": metadata.committer.name,
                        "email": metadata.committer.email,
                    },
                },
            )
            bootstrap_sha = bootstrap_result["commit"]["sha"]
            bootstrap_api_commit = request(
                client, "GET", f"{api_root}/git/commits/{bootstrap_sha}"
            )
            bootstrap_result["commit"] = bootstrap_api_commit
            bootstrap_commit = materialize_bootstrap_commit(
                bootstrap_result,
                content=bootstrap_content,
                filename=bootstrap_filename,
                author_timezone_hint=metadata.author.date,
                committer_timezone_hint=metadata.committer.date,
            )
            print(f"已创建 Git Data API 引导提交 {bootstrap_commit}")

        tree_entries: list[dict[str, str]] = []
        uploaded: set[str] = set()
        for mode, sha, path in tracked_blobs(local_commit):
            if sha not in uploaded:
                content = git("cat-file", "blob", sha, text=False)
                assert isinstance(content, bytes)
                result = request(
                    client,
                    "POST",
                    f"{api_root}/git/blobs",
                    json={"content": base64.b64encode(content).decode(), "encoding": "base64"},
                )
                if result["sha"] != sha:
                    raise SystemExit(f"Blob SHA 校验失败: {path}")
                uploaded.add(sha)
            tree_entries.append({"path": path, "mode": mode, "type": "blob", "sha": sha})

        tree = request(
            client,
            "POST",
            f"{api_root}/git/trees",
            json={"tree": tree_entries},
        )["sha"]
        if tree != metadata.tree:
            raise SystemExit(f"Tree SHA 校验失败，本地 {metadata.tree}，远端 {tree}")

        upload_commit(client, api_root, local_commit, metadata)

        published_commit = local_commit
        if bootstrap_commit:
            now = dt.datetime.now().astimezone().replace(microsecond=0).isoformat()
            merge_metadata = CommitMetadata(
                tree=metadata.tree,
                parents=(local_commit, bootstrap_commit),
                author=Identity(metadata.author.name, metadata.author.email, now),
                committer=Identity(metadata.committer.name, metadata.committer.email, now),
                message="chore: connect GitHub API bootstrap history\n",
            )
            published_commit = create_local_commit(merge_metadata)
            upload_commit(client, api_root, published_commit, merge_metadata)
            request(
                client,
                "PATCH",
                f"{api_root}/git/refs/heads/{args.branch}",
                json={"sha": published_commit, "force": False},
            )
            current_head = str(git("rev-parse", "HEAD")).strip()
            if current_head != local_commit:
                raise SystemExit("本地 HEAD 已变化，未移动本地分支")
            subprocess.run(
                [
                    "git",
                    "update-ref",
                    f"refs/heads/{args.branch}",
                    published_commit,
                    local_commit,
                ],
                check=True,
            )
        elif existing_remote_commit:
            request(
                client,
                "PATCH",
                f"{api_root}/git/refs/heads/{args.branch}",
                json={"sha": local_commit, "force": False},
            )
        else:
            request(
                client,
                "POST",
                f"{api_root}/git/refs",
                json={"ref": f"refs/heads/{args.branch}", "sha": local_commit},
            )
        try:
            remote_url = str(git("remote", "get-url", args.remote_name)).strip().casefold()
        except subprocess.CalledProcessError:
            remote_url = ""
        normalized_repository = owner_repo.casefold()
        if normalized_repository in remote_url:
            subprocess.run(
                [
                    "git",
                    "update-ref",
                    f"refs/remotes/{args.remote_name}/{args.branch}",
                    published_commit,
                ],
                check=True,
            )
        print(f"已发布 {owner_repo}:{args.branch} -> {published_commit}")


if __name__ == "__main__":
    main()
