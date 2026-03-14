"""Persistent store for computed embeddings.

Stores embeddings keyed by (model_id, date, ticker). Embeddings are
treated as features — all point-in-time rules apply.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EmbeddingFeatureStore:
    """In-memory persistent store for model embeddings.

    Keyed by (model_id, date, ticker). Supports retrieval with
    point-in-time filtering.
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._store: Dict[str, pd.DataFrame] = {}  # model_id -> DataFrame

    def store(
        self,
        model_id: str,
        date: pd.Timestamp,
        embeddings: pd.DataFrame,
    ) -> None:
        """Store embeddings for a model at a specific date.

        Args:
            model_id: Identifier for the model that produced these embeddings.
            date: Date of computation (point-in-time).
            embeddings: DataFrame indexed by ticker with embedding columns.
        """
        emb = embeddings.copy()
        emb["_date"] = date
        emb["_model_id"] = model_id

        if model_id not in self._store:
            self._store[model_id] = emb
        else:
            self._store[model_id] = pd.concat(
                [self._store[model_id], emb], ignore_index=False
            )

    def retrieve(
        self,
        model_id: str,
        as_of: pd.Timestamp,
        tickers: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve the most recent embeddings before as_of.

        Args:
            model_id: Model identifier.
            as_of: Point-in-time boundary. Returns embeddings from the
                most recent date strictly before as_of.
            tickers: Optional list of tickers to filter.

        Returns:
            DataFrame indexed by ticker with embedding columns,
            or None if no data available.
        """
        if model_id not in self._store:
            return None

        df = self._store[model_id]
        valid = df[df["_date"] < as_of]

        if valid.empty:
            return None

        latest_date = valid["_date"].max()
        latest = valid[valid["_date"] == latest_date].copy()

        if tickers is not None:
            latest = latest[latest.index.isin(tickers)]

        result = latest.drop(columns=["_date", "_model_id"], errors="ignore")
        return result if not result.empty else None

    def get_available_dates(self, model_id: str) -> List[pd.Timestamp]:
        """Get all dates for which embeddings are stored."""
        if model_id not in self._store:
            return []
        return sorted(self._store[model_id]["_date"].unique())

    def get_model_ids(self) -> List[str]:
        """Get all model IDs with stored embeddings."""
        return list(self._store.keys())

    def clear(self, model_id: Optional[str] = None) -> None:
        """Clear stored embeddings."""
        if model_id:
            self._store.pop(model_id, None)
        else:
            self._store.clear()
