"""Implied volatility surface builder.

Constructs a smooth IV surface from option chain data using
interpolation across strikes and expiries.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImpliedVolatilitySurfaceBuilder:
    """Builds a smooth IV surface from discrete option chain data.

    The surface is parameterised by (moneyness, time-to-expiry) and
    provides interpolated IV at arbitrary points.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._interpolation_method = cfg.get("iv_interpolation", "linear")

    def build_surface(
        self,
        chain_df: pd.DataFrame,
        spot_price: float,
        as_of: pd.Timestamp,
    ) -> dict:
        """Build an IV surface from options chain data.

        Args:
            chain_df: Options chain DataFrame with strike, expiry, implied_vol.
            spot_price: Current stock price.
            as_of: Valuation date.

        Returns:
            Dict with surface data: moneyness grid, tte grid, iv grid.
        """
        if chain_df.empty or "implied_vol" not in chain_df.columns:
            return {"moneyness": [], "tte": [], "iv": []}

        chain = chain_df.dropna(subset=["implied_vol", "strike", "expiry"])
        if chain.empty:
            return {"moneyness": [], "tte": [], "iv": []}

        # Compute moneyness = strike / spot
        chain = chain.copy()
        chain["moneyness"] = chain["strike"] / spot_price
        chain["tte"] = (pd.to_datetime(chain["expiry"]) - as_of).dt.days / 365.0
        chain = chain[chain["tte"] > 0]

        return {
            "moneyness": chain["moneyness"].values.tolist(),
            "tte": chain["tte"].values.tolist(),
            "iv": chain["implied_vol"].values.tolist(),
        }

    def get_atm_iv(self, surface: dict) -> float:
        """Extract at-the-money implied volatility from the surface."""
        if not surface["moneyness"]:
            return np.nan

        moneyness = np.array(surface["moneyness"])
        ivs = np.array(surface["iv"])

        # Find closest to ATM (moneyness = 1.0)
        atm_idx = np.argmin(np.abs(moneyness - 1.0))
        return float(ivs[atm_idx])

    def get_skew(self, surface: dict, delta_low: float = 0.75, delta_high: float = 1.25) -> float:
        """Compute IV skew: 25-delta put IV minus 25-delta call IV.

        Approximated using moneyness bins.
        """
        if not surface["moneyness"]:
            return np.nan

        moneyness = np.array(surface["moneyness"])
        ivs = np.array(surface["iv"])

        put_mask = moneyness <= delta_low
        call_mask = moneyness >= delta_high

        if not put_mask.any() or not call_mask.any():
            return np.nan

        put_iv = ivs[put_mask].mean()
        call_iv = ivs[call_mask].mean()
        return float(put_iv - call_iv)
