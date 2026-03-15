"""Dataset version control — reproducible dataset tracking for backtests.

Manages versioning of datasets so that every backtest, signal generation,
and model training run can reference the exact dataset version it used.
Metadata and checksums are stored in SQLite; actual data files remain
in-place on the filesystem.
"""

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatasetVersionControl:
    """SQLite-backed dataset version registry for reproducibility.

    Tracks dataset versions by computing SHA-256 checksums and storing
    metadata (path, row counts, date ranges, etc.) alongside each
    registered version.  Supports tagging and point-in-time snapshots.

    Usage:
        dvc = DatasetVersionControl(config={"chunk_size": 8192})
        vid = dvc.register_dataset("sp500_daily", "/data/sp500.parquet",
                                   metadata={"rows": 504_000, "source": "polygon"})
        dvc.tag_version("sp500_daily", vid, "production")
        info = dvc.get_by_tag("sp500_daily", "production")
        assert dvc.validate_integrity("sp500_daily", vid)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        db_path: str = ":memory:",
    ):
        """Initialize dataset version control.

        Args:
            config: Optional configuration dict.  Recognised keys:
                - chunk_size (int): read chunk size for checksum (default 65536).
            db_path: Path to SQLite database. Use ":memory:" for testing.
        """
        self._config = config or {}
        self._chunk_size: int = self._config.get("chunk_size", 65_536)

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        logger.info("DatasetVersionControl initialized with db_path=%s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create tables and indexes if they do not exist."""
        cur = self._conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                path        TEXT NOT NULL,
                checksum    TEXT NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_tags (
                name        TEXT NOT NULL,
                version_id  TEXT NOT NULL,
                tag         TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (name, tag),
                FOREIGN KEY (version_id) REFERENCES dataset_versions(version_id)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                version_id  TEXT NOT NULL,
                description TEXT,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES dataset_versions(version_id)
            )
            """
        )

        # Indexes for common query patterns
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_name "
            "ON dataset_versions (name, created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_tags_name_tag "
            "ON dataset_tags (name, tag)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_name "
            "ON dataset_snapshots (name, created_at)"
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Core version operations
    # ------------------------------------------------------------------

    def register_dataset(
        self,
        name: str,
        path: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Register a dataset version and return its version_id.

        Computes a SHA-256 checksum of the file/directory at *path* and
        stores the version record.  If the exact same checksum already
        exists for this dataset name the record is still created (a new
        version_id) so that the caller always gets a unique handle.

        Args:
            name: Logical dataset name (e.g. "sp500_daily").
            path: Filesystem path to the dataset file or directory.
            metadata: Arbitrary metadata dict (row count, date range, etc.).

        Returns:
            A unique version_id string.
        """
        checksum = self.compute_checksum(path)
        version_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        self._conn.execute(
            """
            INSERT INTO dataset_versions
                (version_id, name, path, checksum, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (version_id, name, path, checksum, meta_json, now),
        )
        self._conn.commit()
        logger.info(
            "Registered dataset %s version=%s checksum=%s",
            name, version_id, checksum[:12],
        )
        return version_id

    def get_version(self, name: str, version_id: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset version.

        Args:
            name: Logical dataset name.
            version_id: The version identifier returned by register_dataset.

        Returns:
            Dict with keys: version_id, name, path, checksum, metadata,
            created_at.

        Raises:
            KeyError: If the version is not found.
        """
        row = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE name = ? AND version_id = ?",
            (name, version_id),
        ).fetchone()

        if row is None:
            raise KeyError(
                f"Version {version_id} not found for dataset '{name}'"
            )
        return self._row_to_dict(row)

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a dataset, oldest first.

        Args:
            name: Logical dataset name.

        Returns:
            List of version dicts ordered by creation time ascending.
        """
        rows = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE name = ? ORDER BY created_at ASC",
            (name,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_latest_version(self, name: str) -> Dict[str, Any]:
        """Get the most recently registered version of a dataset.

        Args:
            name: Logical dataset name.

        Returns:
            Version dict for the latest version.

        Raises:
            KeyError: If no versions exist for the dataset.
        """
        row = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE name = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (name,),
        ).fetchone()

        if row is None:
            raise KeyError(f"No versions found for dataset '{name}'")
        return self._row_to_dict(row)

    # ------------------------------------------------------------------
    # Checksum / integrity
    # ------------------------------------------------------------------

    def compute_checksum(self, path: str) -> str:
        """Compute SHA-256 checksum of a file or directory.

        For a single file the checksum covers the entire file contents.
        For a directory the checksum is computed over sorted relative
        paths and their individual checksums, producing a deterministic
        hash of the whole tree.

        Args:
            path: Filesystem path to a file or directory.

        Returns:
            Hex-encoded SHA-256 digest string.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if p.is_file():
            return self._checksum_file(p)

        # Directory: deterministic hash of (relative_path, file_hash) pairs
        h = hashlib.sha256()
        for child in sorted(p.rglob("*")):
            if child.is_file():
                rel = child.relative_to(p).as_posix()
                file_hash = self._checksum_file(child)
                h.update(f"{rel}:{file_hash}\n".encode())
        return h.hexdigest()

    def _checksum_file(self, filepath: Path) -> str:
        """Compute SHA-256 of a single file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(self._chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def validate_integrity(self, name: str, version_id: str) -> bool:
        """Validate that a dataset has not been modified since registration.

        Re-computes the checksum of the file/directory at the stored path
        and compares it against the recorded checksum.

        Args:
            name: Logical dataset name.
            version_id: Version identifier.

        Returns:
            True if the current checksum matches the stored one.
        """
        version = self.get_version(name, version_id)
        stored_checksum = version["checksum"]
        try:
            current_checksum = self.compute_checksum(version["path"])
        except FileNotFoundError:
            logger.warning(
                "Integrity check failed — path missing: %s", version["path"]
            )
            return False

        valid = current_checksum == stored_checksum
        if not valid:
            logger.warning(
                "Integrity check FAILED for %s version=%s "
                "(stored=%s current=%s)",
                name, version_id, stored_checksum[:12], current_checksum[:12],
            )
        else:
            logger.debug(
                "Integrity check passed for %s version=%s", name, version_id
            )
        return valid

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def tag_version(self, name: str, version_id: str, tag: str) -> bool:
        """Tag a dataset version with a human-readable label.

        If the tag already exists for this dataset it is moved to the
        new version_id (upsert).

        Args:
            name: Logical dataset name.
            version_id: Version to tag.
            tag: Tag label (e.g. "production", "backtest-2024Q1").

        Returns:
            True if the tag was successfully applied.
        """
        # Verify the version exists
        self.get_version(name, version_id)

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO dataset_tags (name, version_id, tag, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name, tag)
            DO UPDATE SET version_id = excluded.version_id,
                          created_at = excluded.created_at
            """,
            (name, version_id, tag, now),
        )
        self._conn.commit()
        logger.info("Tagged %s version=%s as '%s'", name, version_id, tag)
        return True

    def get_by_tag(self, name: str, tag: str) -> Dict[str, Any]:
        """Get the dataset version associated with a tag.

        Args:
            name: Logical dataset name.
            tag: Tag label.

        Returns:
            Version dict for the tagged version.

        Raises:
            KeyError: If the tag is not found.
        """
        row = self._conn.execute(
            "SELECT version_id FROM dataset_tags WHERE name = ? AND tag = ?",
            (name, tag),
        ).fetchone()

        if row is None:
            raise KeyError(f"Tag '{tag}' not found for dataset '{name}'")

        return self.get_version(name, row["version_id"])

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def create_snapshot(self, name: str, description: str = "") -> str:
        """Create a point-in-time snapshot record of the latest version.

        A snapshot captures the current latest version_id so that future
        look-ups can retrieve the exact state at this moment, even after
        new versions are registered.

        Args:
            name: Logical dataset name.
            description: Human-readable description of the snapshot purpose.

        Returns:
            A unique snapshot_id string.

        Raises:
            KeyError: If no versions exist for the dataset.
        """
        latest = self.get_latest_version(name)
        snapshot_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO dataset_snapshots
                (snapshot_id, name, version_id, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (snapshot_id, name, latest["version_id"], description, now),
        )
        self._conn.commit()
        logger.info(
            "Created snapshot %s for %s (version=%s): %s",
            snapshot_id, name, latest["version_id"], description,
        )
        return snapshot_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict, deserializing metadata."""
        d = dict(row)
        if "metadata" in d and d["metadata"] is not None:
            d["metadata"] = json.loads(d["metadata"])
        return d

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("DatasetVersionControl connection closed")

    def __enter__(self) -> "DatasetVersionControl":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
