"""Object storage for uploaded source images.

Images never go into Postgres; the database keeps only the storage key. Two
backends implement the same small interface:

* ``LocalObjectStorage`` writes under the upload directory. It is the default,
  so the backend still runs with no cloud configuration at all.
* ``SupabaseObjectStorage`` talks to a private Supabase Storage bucket over its
  REST API with the standard library, mirroring how Model B reaches OpenRouter.
  The service-role key stays on the server; browsers get short-lived signed URLs.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol


class ObjectStorageError(RuntimeError):
    pass


class ObjectStorage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, keys: list[str]) -> None: ...

    def signed_url(self, key: str, expires_in_s: int) -> str | None: ...


def source_image_key(session_id: str, extension: str) -> str:
    return f"sessions/{session_id}/source{extension}"


class LocalObjectStorage:
    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ObjectStorageError("Storage key escapes the storage root.")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        try:
            return self.path_for(key).read_bytes()
        except OSError as error:
            raise ObjectStorageError("Stored object is not available.") from error

    def delete(self, keys: list[str]) -> None:
        for key in keys:
            try:
                self.path_for(key).unlink(missing_ok=True)
            except (OSError, ObjectStorageError):
                pass

    def signed_url(self, key: str, expires_in_s: int) -> str | None:
        return None


class SupabaseObjectStorage:
    def __init__(self, url: str, service_role_key: str, bucket: str, timeout_s: float = 20.0) -> None:
        self._base = url.rstrip("/") + "/storage/v1"
        self._key = service_role_key
        self._bucket = bucket
        self._timeout_s = timeout_s

    def _request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
        request = urllib.request.Request(
            f"{self._base}{path}",
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self._key}", "apikey": self._key, **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # Never echo the response body: it can quote request headers back.
            raise ObjectStorageError(f"Supabase Storage {method} failed with HTTP {error.code}.") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ObjectStorageError(f"Supabase Storage {method} could not be reached.") from error

    def _object_path(self, key: str) -> str:
        return f"/object/{self._bucket}/{urllib.parse.quote(key)}"

    def ensure_bucket(self) -> None:
        """Create the private bucket if it does not exist yet."""
        try:
            self._request("GET", f"/bucket/{self._bucket}")
        except ObjectStorageError:
            payload = json.dumps({"id": self._bucket, "name": self._bucket, "public": False}).encode()
            self._request("POST", "/bucket", payload, {"Content-Type": "application/json"})

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._request("POST", self._object_path(key), data, {"Content-Type": content_type, "x-upsert": "true"})

    def get(self, key: str) -> bytes:
        return self._request("GET", self._object_path(key))

    def delete(self, keys: list[str]) -> None:
        if keys:
            payload = json.dumps({"prefixes": keys}).encode()
            self._request("DELETE", f"/object/{self._bucket}", payload, {"Content-Type": "application/json"})

    def signed_url(self, key: str, expires_in_s: int) -> str | None:
        payload = json.dumps({"expiresIn": expires_in_s}).encode()
        body = self._request(
            "POST", f"/object/sign/{self._bucket}/{urllib.parse.quote(key)}", payload, {"Content-Type": "application/json"},
        )
        signed = json.loads(body or b"{}").get("signedURL")
        return f"{self._base}{signed}" if signed else None
