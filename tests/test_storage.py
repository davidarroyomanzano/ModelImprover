from unittest.mock import MagicMock, patch

import pytest

from ds_project.config import StorageCfg
from ds_project.utils import storage as st
from ds_project.utils.storage import (
    LocalStorage,
    OneDriveStorage,
    S3Storage,
    StorageBackend,
    get_storage,
)

pytest.importorskip("requests")
GRAPH = OneDriveStorage.GRAPH
ONEDRIVE_VARS = (
    "ONEDRIVE_CLIENT_ID",
    "ONEDRIVE_CLIENT_SECRET",
    "ONEDRIVE_TENANT_ID",
    "ONEDRIVE_USER_ID",
)


# ---------- Base / Local ----------
class _Recorder(StorageBackend):
    def __init__(self):
        self.keys = []

    def upload_file(self, local_path, key):
        self.keys.append(key)


def test_upload_directory_keys(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub" / "b.txt").write_text("b")
    rec = _Recorder()
    assert rec.upload_directory(tmp_path, "/models/") == 2
    assert rec.keys == ["models/a.txt", "models/sub/b.txt"]
    rec2 = _Recorder()
    rec2.upload_directory(tmp_path, "")
    assert rec2.keys == ["a.txt", "sub/b.txt"]


def test_base_not_implemented():
    with pytest.raises(NotImplementedError):
        StorageBackend().list_files()


def test_local_roundtrip(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "x.wav").write_bytes(b"123")
    store = LocalStorage(tmp_path / "remote")
    assert store.list_files() == []
    assert store.upload_directory(src, "raw") == 1
    assert store.list_files("raw") == ["raw/a/x.wav"]
    assert store.list_files("nope") == []
    assert store.download_file("raw/a/x.wav", tmp_path / "dl" / "x.wav").read_bytes() == b"123"


# ---------- S3 ----------
def _s3(prefix="pre"):
    client = MagicMock()
    return S3Storage("bucket", prefix, client=client), client


def test_s3_requires_bucket():
    with pytest.raises(ValueError):
        S3Storage("", client=MagicMock())


def test_s3_upload_download(tmp_path):
    s, c = _s3("/pre/")
    s.upload_file(tmp_path / "f.wav", "a/f.wav")
    c.upload_file.assert_called_once_with(str(tmp_path / "f.wav"), "bucket", "pre/a/f.wav")
    dest = s.download_file("/a/f.wav", tmp_path / "x" / "y" / "f.wav")
    assert dest.parent.is_dir()
    c.download_file.assert_called_once_with("bucket", "pre/a/f.wav", str(dest))


def test_s3_list_files():
    s, c = _s3()
    paginator = c.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "pre/raw/"}, {"Key": "pre/raw/a.wav"}]},
        {},
    ]
    assert s.list_files("raw") == ["raw/a.wav"]
    c.get_paginator.assert_called_once_with("list_objects_v2")
    paginator.paginate.assert_called_once_with(Bucket="bucket", Prefix="pre/raw")


def test_s3_list_without_prefix():
    s, c = _s3("")
    c.get_paginator.return_value.paginate.return_value = [{"Contents": [{"Key": "raw/a.wav"}]}]
    assert s.list_files("raw") == ["raw/a.wav"]


# ---------- OneDrive ----------
@pytest.fixture
def msal(fake_module):
    m = fake_module("msal")
    m.ConfidentialClientApplication.return_value.acquire_token_for_client.return_value = {
        "access_token": "tok"
    }
    return m


@pytest.fixture
def onedrive(msal):
    return OneDriveStorage("cid", "sec", "ten", "uid", root="/root/")


def _resp(payload=None):
    r = MagicMock()
    r.json.return_value = payload or {}
    return r


def test_onedrive_missing_credentials():
    with pytest.raises(ValueError):
        OneDriveStorage("cid", "", "ten", "uid")


def test_onedrive_token_cached(onedrive, msal):
    assert onedrive._headers() == {"Authorization": "Bearer tok"}
    onedrive._headers()
    msal.ConfidentialClientApplication.assert_called_once_with(
        "cid",
        authority="https://login.microsoftonline.com/ten",
        client_credential="sec",
    )


def test_onedrive_auth_error(onedrive, msal):
    msal.ConfidentialClientApplication.return_value.acquire_token_for_client.return_value = {
        "error_description": "bad secret"
    }
    with pytest.raises(RuntimeError, match="bad secret"):
        onedrive._headers()


def test_onedrive_simple_upload(onedrive, tmp_path):
    f = tmp_path / "x.wav"
    f.write_bytes(b"abc")
    with patch("requests.put") as put:
        onedrive.upload_file(f, "raw/x.wav")
    assert put.call_args.args[0] == f"{GRAPH}/users/uid/drive/root:/root/raw/x.wav:/content"
    assert put.call_args.kwargs["headers"] == {"Authorization": "Bearer tok"}
    assert put.call_args.kwargs["data"] == b"abc"
    put.return_value.raise_for_status.assert_called_once()


def test_onedrive_chunked_upload(onedrive, tmp_path):
    onedrive.SIMPLE_UPLOAD_LIMIT = 4
    onedrive.CHUNK = 4
    f = tmp_path / "big.bin"
    f.write_bytes(b"0123456789")
    with patch("requests.post", return_value=_resp({"uploadUrl": "https://up"})) as post, patch(
        "requests.put"
    ) as put:
        onedrive.upload_file(f, "big.bin")
    assert post.call_args.args[0].endswith("/root/big.bin:/createUploadSession")
    assert [c.args[0] for c in put.call_args_list] == ["https://up"] * 3
    assert [c.kwargs["headers"]["Content-Range"] for c in put.call_args_list] == [
        "bytes 0-3/10",
        "bytes 4-7/10",
        "bytes 8-9/10",
    ]
    assert b"".join(c.kwargs["data"] for c in put.call_args_list) == b"0123456789"


def test_onedrive_download(onedrive, tmp_path):
    resp = MagicMock()
    resp.__enter__.return_value = resp
    resp.iter_content.return_value = [b"ab", b"c"]
    with patch("requests.get", return_value=resp) as get:
        out = onedrive.download_file("raw/x.wav", tmp_path / "d" / "x.wav")
    assert out.read_bytes() == b"abc"
    assert get.call_args.kwargs["stream"] is True
    resp.raise_for_status.assert_called_once()


def test_onedrive_list_recursive_with_pagination(onedrive):
    pages = [
        _resp(
            {
                "value": [{"name": "a", "folder": {}}, {"name": "f1.wav"}],
                "@odata.nextLink": "https://next",
            }
        ),
        _resp({"value": [{"name": "f2.wav"}]}),
        _resp({"value": [{"name": "x.wav"}]}),
    ]
    with patch("requests.get", side_effect=pages) as get:
        out = onedrive.list_files("raw")
    assert out == ["raw/a/x.wav", "raw/f1.wav", "raw/f2.wav"]
    urls = [c.args[0] for c in get.call_args_list]
    assert urls[0] == f"{GRAPH}/users/uid/drive/root:/root/raw:/children"
    assert urls[1] == "https://next"
    assert urls[2] == f"{GRAPH}/users/uid/drive/root:/root/raw/a:/children"


def test_onedrive_list_drive_root(msal):
    store = OneDriveStorage("cid", "sec", "ten", "uid")
    with patch("requests.get", return_value=_resp({"value": []})) as get:
        assert store.list_files() == []
    assert get.call_args.args[0] == f"{GRAPH}/users/uid/drive/root/children"


# ---------- Factoría ----------
def test_get_storage_none_and_local(tmp_path):
    assert get_storage(StorageCfg()) is None
    store = get_storage(StorageCfg(backend="local", local_root=str(tmp_path)))
    assert isinstance(store, LocalStorage) and store.root == tmp_path


def test_get_storage_s3():
    with patch.object(st, "S3Storage") as s3:
        get_storage(StorageCfg(backend="s3", s3_bucket="b", s3_prefix="p"))
    s3.assert_called_once_with("b", "p")


def test_get_storage_onedrive(monkeypatch):
    for var in ONEDRIVE_VARS:
        monkeypatch.setenv(var, "v")
    store = get_storage(StorageCfg(backend="onedrive", onedrive_root="r"))
    assert isinstance(store, OneDriveStorage) and store.root == "r"


def test_get_storage_onedrive_missing_env(monkeypatch):
    for var in ONEDRIVE_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError):
        get_storage(StorageCfg(backend="onedrive"))


def test_get_storage_unknown():
    cfg = StorageCfg()
    cfg.backend = "ftp"
    with pytest.raises(ValueError):
        get_storage(cfg)
