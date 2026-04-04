"""State persistence layer — SQLite-backed key-value store.

Provides durable storage for components that currently keep state
in-memory only: approval workflow, deployment registry, kill switch
peak NAV, alerting history, order state.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StateStore:
    """SQLite-backed key-value store with namespace isolation.

    Each component stores its state in a namespace (table partition).
    Values are JSON-serialized for flexibility.

    Usage:
        store = StateStore("/path/to/state.db")
        store.save("kill_switch", "peak_nav", 1_050_000.0)
        val = store.load("kill_switch", "peak_nav")
        store.save("approval", "strategy_alpha", {"status": "APPROVED", "days": 130})
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize state store.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for testing.
        """
        if db_path != ":memory:":
            parent = Path(db_path).parent
            parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state (
                namespace TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._conn.commit()
        logger.debug("StateStore initialized with db_path=%s", db_path)

    def save(self, namespace: str, key: str, value: Any) -> None:
        """Save a value. Overwrites if key exists.

        Values are JSON-serialized. Supports: dict, list, str, int, float, bool, None.
        """
        serialized = json.dumps(value)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO state (namespace, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, key)
            DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (namespace, key, serialized, now),
        )
        self._conn.commit()
        logger.debug("Saved %s/%s", namespace, key)

    def load(self, namespace: str, key: str, default: Any = None) -> Any:
        """Load a value. Returns default if not found."""
        cursor = self._conn.execute(
            "SELECT value FROM state WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        row = cursor.fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a key. Returns True if key existed."""
        cursor = self._conn.execute(
            "DELETE FROM state WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_keys(self, namespace: str) -> List[str]:
        """List all keys in a namespace."""
        cursor = self._conn.execute(
            "SELECT key FROM state WHERE namespace = ? ORDER BY key",
            (namespace,),
        )
        return [row[0] for row in cursor.fetchall()]

    def load_all(self, namespace: str) -> Dict[str, Any]:
        """Load all key-value pairs in a namespace."""
        cursor = self._conn.execute(
            "SELECT key, value FROM state WHERE namespace = ? ORDER BY key",
            (namespace,),
        )
        return {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

    def clear_namespace(self, namespace: str) -> int:
        """Delete all keys in a namespace. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM state WHERE namespace = ?",
            (namespace,),
        )
        self._conn.commit()
        return cursor.rowcount

    def checkpoint(self) -> None:
        """Force a WAL checkpoint to keep the WAL file from growing unbounded.

        Should be called periodically (e.g. every hour) during long-running
        sessions to prevent the WAL file from growing to many times the
        size of the main database.
        """
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("WAL checkpoint completed")
        except Exception:
            logger.warning("WAL checkpoint failed", exc_info=True)

    def close(self) -> None:
        """Close the database connection."""
        self.checkpoint()
        self._conn.close()
        logger.debug("StateStore connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
