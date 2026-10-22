"""Backends de almacenamiento: local, S3 (boto3) y OneDrive (MSAL + Microsoft Graph)."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

from ds_project.config import StorageCfg

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


class StorageBackend:
    def upload_file(self, local_path: PathLike, key: str) -> None:
        raise NotImplementedError

    def download_file(self, key: str, local_path: PathLike) -> Path:
        raise NotImplementedError

    def list_files(self, prefix: str = "") -> List[str]:
        raise NotImplementedError

    def upload_directory(self, local_dir: PathLike, prefix: str) -> int:
        local_dir = Path(local_dir)
        count = 0
        for f in sorted(p for p in local_dir.rglob("*") if p.is_file()):
            key = "/".join(filter(None, [prefix.strip("/"), f.relative_to(local_dir).as_posix()]))
            self.upload_file(f, key)
            count += 1
        return count


class LocalStorage(StorageBackend):
    def __init__(self, root: PathLike) -> None:
        self.root = Path(root)

    def upload_file(self, local_path: PathLike, key: str) -> None:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def download_file(self, key: str, local_path: PathLike) -> Path:
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / key, dest)
        return dest

    def list_files(self, prefix: str = "") -> List[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = "", client=None) -> None:
        if not bucket:
            raise ValueError("storage.s3_bucket es obligatorio")
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return "/".join(filter(None, [self.prefix, key.lstrip("/")]))

    def upload_file(self, local_path: PathLike, key: str) -> None:
        self.client.upload_file(str(local_path), self.bucket, self._key(key))

    def download_file(self, key: str, local_path: PathLike) -> Path:
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(dest))
        return dest

    def list_files(self, prefix: str = "") -> List[str]:
        full = self._key(prefix)
        keys: List[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if k.endswith("/"):
                    continue
                keys.append(k[len(self.prefix) :].lstrip("/") if self.prefix else k)
        return keys


class OneDriveStorage(StorageBackend):
    GRAPH = "https://graph.microsoft.com/v1.0"
    SIMPLE_UPLOAD_LIMIT = 4 * 1024 * 1024
    CHUNK = 320 * 1024 * 10  # múltiplo de 320 KiB

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        user_id: str,
        root: str = "",
    ) -> None:
        if not all([client_id, client_secret, tenant_id, user_id]):
            raise ValueError("Faltan credenciales ONEDRIVE_* en el entorno")
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.root = root.strip("/")
        self._app = None

    def _token(self) -> str:
        import msal

        if self._app is None:
            self._app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=f"https://login.microsoftonline.com/{self.tenant_id}",
                client_credential=self.client_secret,
            )
        result = self._app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if "access_token" not in result:
            raise RuntimeError(
                f"Error de autenticación OneDrive: {result.get('error_description')}"
            )
        return result["access_token"]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token()}"}

    def _path(self, key: str) -> str:
        return "/".join(filter(None, [self.root, key.strip("/")]))

    def _item_url(self, key: str) -> str:
        return f"{self.GRAPH}/users/{self.user_id}/drive/root:/{self._path(key)}:"

    def upload_file(self, local_path: PathLike, key: str) -> None:
        import requests

        local_path = Path(local_path)
        size = local_path.stat().st_size
        if size <= self.SIMPLE_UPLOAD_LIMIT:
            r = requests.put(
                f"{self._item_url(key)}/content",
                headers=self._headers(),
                data=local_path.read_bytes(),
                timeout=120,
            )
            r.raise_for_status()
            return
        session = requests.post(
            f"{self._item_url(key)}/createUploadSession",
            headers=self._headers(),
            json={"item": {"@microsoft.graph.conflictBehavior": "replace"}},
            timeout=60,
        )
        session.raise_for_status()
        upload_url = session.json()["uploadUrl"]
        with local_path.open("rb") as fh:
            start = 0
            while start < size:
                chunk = fh.read(self.CHUNK)
                end = start + len(chunk) - 1
                r = requests.put(
                    upload_url,
                    headers={"Content-Range": f"bytes {start}-{end}/{size}"},
                    data=chunk,
                    timeout=300,
                )
                r.raise_for_status()
                start = end + 1

    def download_file(self, key: str, local_path: PathLike) -> Path:
        import requests

        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(
            f"{self._item_url(key)}/content",
            headers=self._headers(),
            stream=True,
            timeout=300,
        ) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for block in r.iter_content(chunk_size=1 << 20):
                    fh.write(block)
        return dest

    def list_files(self, prefix: str = "") -> List[str]:
        import requests

        results: List[str] = []
        stack = [prefix.strip("/")]
        while stack:
            folder = stack.pop()
            path = self._path(folder)
            url: Optional[str] = (
                f"{self.GRAPH}/users/{self.user_id}/drive/root:/{path}:/children"
                if path
                else f"{self.GRAPH}/users/{self.user_id}/drive/root/children"
            )
            while url:
                r = requests.get(url, headers=self._headers(), timeout=60)
                r.raise_for_status()
                payload = r.json()
                for item in payload.get("value", []):
                    rel = "/".join(filter(None, [folder, item["name"]]))
                    if "folder" in item:
                        stack.append(rel)
                    else:
                        results.append(rel)
                url = payload.get("@odata.nextLink")
        return sorted(results)


def get_storage(cfg: StorageCfg) -> Optional[StorageBackend]:
    if cfg.backend == "none":
        return None
    if cfg.backend == "local":
        return LocalStorage(cfg.local_root)
    if cfg.backend == "s3":
        return S3Storage(cfg.s3_bucket, cfg.s3_prefix)
    if cfg.backend == "onedrive":
        return OneDriveStorage(
            client_id=os.getenv("ONEDRIVE_CLIENT_ID", ""),
            client_secret=os.getenv("ONEDRIVE_CLIENT_SECRET", ""),
            tenant_id=os.getenv("ONEDRIVE_TENANT_ID", ""),
            user_id=os.getenv("ONEDRIVE_USER_ID", ""),
            root=cfg.onedrive_root,
        )
    raise ValueError(f"Backend desconocido: {cfg.backend}")
