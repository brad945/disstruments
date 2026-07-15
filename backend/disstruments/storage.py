"""Storage interface. Local FS now; S3/R2 adapter at Phase 1 (design doc §12)."""
from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from .config import settings


class Storage(ABC):
    @abstractmethod
    def save(self, key: str, src: Path) -> int: ...
    @abstractmethod
    def path(self, key: str) -> Path: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalStorage(Storage):
    def __init__(self, root: Path | None = None):
        self.root = (root or settings.data_dir) / "objects"
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError("path traversal")
        return p

    def save(self, key: str, src: Path) -> int:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return dst.stat().st_size

    def path(self, key: str) -> Path:
        return self._p(key)

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)


storage: Storage = None  # set in main.create_app / tests


def init_storage(root: Path | None = None) -> Storage:
    global storage
    storage = LocalStorage(root)
    return storage
