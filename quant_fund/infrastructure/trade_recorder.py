"""Trade recorder — persistent audit log for compliance.

Records every order submitted, fill received, kill switch check,
and approval state transition. Append-only SQLite storage with
indexed timestamps for efficient querying.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TradeRecorder:
    """Append-only trade and event recorder for compliance audit trail.

    Records:
    - Orders submitted (timestamp, strategy_id, ticker, side, qty, type, algo)
    - Fills received (fill_price, commission, slippage)
    - Kill switch checks (NAV, peak_nav, drawdown, triggered)
    - Approval state transitions (from_state, to_state, actor, reason)

    Usage:
        recorder = TradeRecorder("/path/to/trades.db")
        recorder.record_order(order_dict)
        recorder.record_fill(fill_dict)
        recorder.record_kill_switch_check(nav=950000, peak_nav=1000000, drawdown=0.05, triggered=False)
        recorder.record_approval_transition(strategy_id="alpha", from_state="UNDER_REVIEW", to_state="APPROVED", actor="risk_team", reason="Passed all checks")
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize trade recorder.

        Args:
            db_path: Path to SQLite database. Use ":memory:" for testing.
        """
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("TradeRecorder initialized with db_path=%s", db_path)

    def _create_tables(self) -> None:
        """Create tables and indexes if they do not exist."""
        cur = self._conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy_id TEXT,
                ticker TEXT,
                side TEXT,
                quantity INTEGER,
                order_type TEXT,
                algo TEXT,
                limit_price REAL,
                metadata TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                order_id TEXT,
                ticker TEXT,
                side TEXT,
                quantity INTEGER,
                fill_price REAL,
                commission REAL,
                slippage_bps REAL,
                metadata TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kill_switch_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy_id TEXT,
                nav REAL,
                peak_nav REAL,
                drawdown REAL,
                triggered INTEGER
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS approval_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy_id TEXT,
                from_state TEXT,
                to_state TEXT,
                actor TEXT,
                reason TEXT
            )
            """
        )

        # Indexes for efficient timestamp-based queries
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders (timestamp)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills (timestamp)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_kill_switch_checks_timestamp "
            "ON kill_switch_checks (timestamp)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_approval_transitions_timestamp "
            "ON approval_transitions (timestamp)"
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # Record methods
    # ------------------------------------------------------------------

    def record_order(
        self,
        timestamp: Optional[str] = None,
        strategy_id: str = "",
        ticker: str = "",
        side: str = "",
        quantity: int = 0,
        order_type: str = "market",
        algo: str = "direct",
        limit_price: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Record an order submission. Returns row ID."""
        ts = timestamp or pd.Timestamp.now().isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        cur = self._conn.execute(
            """
            INSERT INTO orders
                (timestamp, strategy_id, ticker, side, quantity,
                 order_type, algo, limit_price, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, strategy_id, ticker, side, quantity,
             order_type, algo, limit_price, meta_json),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        logger.debug("Recorded order id=%d ticker=%s side=%s qty=%d",
                      row_id, ticker, side, quantity)
        return row_id

    def record_fill(
        self,
        timestamp: Optional[str] = None,
        order_id: str = "",
        ticker: str = "",
        side: str = "",
        quantity: int = 0,
        fill_price: float = 0.0,
        commission: float = 0.0,
        slippage_bps: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> int:
        """Record a fill. Returns row ID."""
        ts = timestamp or pd.Timestamp.now().isoformat()
        meta_json = json.dumps(metadata) if metadata else None

        cur = self._conn.execute(
            """
            INSERT INTO fills
                (timestamp, order_id, ticker, side, quantity,
                 fill_price, commission, slippage_bps, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, order_id, ticker, side, quantity,
             fill_price, commission, slippage_bps, meta_json),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        logger.debug("Recorded fill id=%d ticker=%s price=%.4f",
                      row_id, ticker, fill_price)
        return row_id

    def record_kill_switch_check(
        self,
        strategy_id: str = "",
        nav: float = 0.0,
        peak_nav: float = 0.0,
        drawdown: float = 0.0,
        triggered: bool = False,
        timestamp: Optional[str] = None,
    ) -> int:
        """Record a kill switch check. Returns row ID."""
        ts = timestamp or pd.Timestamp.now().isoformat()

        cur = self._conn.execute(
            """
            INSERT INTO kill_switch_checks
                (timestamp, strategy_id, nav, peak_nav, drawdown, triggered)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, strategy_id, nav, peak_nav, drawdown, int(triggered)),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        logger.debug("Recorded kill switch check id=%d triggered=%s",
                      row_id, triggered)
        return row_id

    def record_approval_transition(
        self,
        strategy_id: str = "",
        from_state: str = "",
        to_state: str = "",
        actor: str = "",
        reason: str = "",
        timestamp: Optional[str] = None,
    ) -> int:
        """Record an approval state transition. Returns row ID."""
        ts = timestamp or pd.Timestamp.now().isoformat()

        cur = self._conn.execute(
            """
            INSERT INTO approval_transitions
                (timestamp, strategy_id, from_state, to_state, actor, reason)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, strategy_id, from_state, to_state, actor, reason),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        logger.debug("Recorded approval transition id=%d %s -> %s",
                      row_id, from_state, to_state)
        return row_id

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_orders(
        self,
        strategy_id: Optional[str] = None,
        ticker: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Query recorded orders."""
        query = "SELECT * FROM orders WHERE 1=1"
        params: list = []

        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker)
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_fills(
        self,
        ticker: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Query recorded fills."""
        query = "SELECT * FROM fills WHERE 1=1"
        params: list = []

        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker)
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_kill_switch_checks(
        self,
        strategy_id: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Query recorded kill switch checks."""
        query = "SELECT * FROM kill_switch_checks WHERE 1=1"
        params: list = []

        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["triggered"] = bool(d["triggered"])
            results.append(d)
        return results

    def get_approval_transitions(
        self,
        strategy_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[dict]:
        """Query recorded approval transitions."""
        query = "SELECT * FROM approval_transitions WHERE 1=1"
        params: list = []

        if strategy_id is not None:
            query += " AND strategy_id = ?"
            params.append(strategy_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def get_order_count(self, strategy_id: Optional[str] = None) -> int:
        """Get total order count."""
        if strategy_id is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM orders WHERE strategy_id = ?",
                (strategy_id,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM orders").fetchone()
        return row[0]

    def get_fill_count(self, ticker: Optional[str] = None) -> int:
        """Get total fill count."""
        if ticker is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM fills WHERE ticker = ?",
                (ticker,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM fills").fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
        logger.info("TradeRecorder connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
