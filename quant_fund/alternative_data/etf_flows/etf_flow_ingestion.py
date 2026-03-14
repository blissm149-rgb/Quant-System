"""ETF flow data ingestion."""

import logging
from typing import List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ETFFlowIngestion:
    """Ingests ETF flow data for sector rotation and flow surprise signals."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._data_source = cfg.get("etf_flows", "factset")

    @classmethod
    def from_config_file(cls, config_path: str) -> "ETFFlowIngestion":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("data_sources", {}))

    def load_flows(
        self,
        etf_tickers: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load ETF flow data.

        Returns:
            DataFrame with columns: etf_ticker, date, net_flow_usd,
            aum, net_flow_pct.
        """
        logger.info("Loading ETF flows for %d ETFs from %s to %s",
                     len(etf_tickers), start_date, end_date)
        return pd.DataFrame(columns=[
            "etf_ticker", "date", "net_flow_usd", "aum", "net_flow_pct",
        ])
