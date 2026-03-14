"""Cross-sectional feature normalisation.

Applied after feature computation, before features enter the portfolio optimizer.
Supports z-score, rank, and percentile normalisation with configurable
winsorisation to handle outliers.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml


class FeatureNormalizer:
    """Normalises cross-sectional features for portfolio construction.

    All normalisation is cross-sectional (across tickers at a single point
    in time). For time-series normalisation, the mean and std must be computed
    using only historical data up to as_of — this is the caller's responsibility.
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._default_method = self._config.get("default_method", "zscore")
        self._default_winsorize_std = self._config.get("winsorize_std", 3.0)

    @classmethod
    def from_config_file(cls, config_path: str) -> "FeatureNormalizer":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config=config.get("normalization", {}))

    def normalize(
        self,
        raw_scores: pd.Series,
        method: Optional[str] = None,
        winsorize_std: Optional[float] = None,
    ) -> pd.Series:
        """Normalise a cross-sectional feature series.

        Args:
            raw_scores: Feature values indexed by ticker.
            method: Normalisation method — "zscore", "rank", or "percentile".
                Defaults to config or "zscore".
            winsorize_std: Number of standard deviations for winsorisation
                before z-scoring. Set to None or 0 to skip. Default: 3.0.

        Returns:
            Normalised Series indexed by ticker.
        """
        method = method or self._default_method
        winsorize_std = (
            winsorize_std if winsorize_std is not None else self._default_winsorize_std
        )

        if method == "zscore":
            return self._zscore_normalize(raw_scores, winsorize_std)
        elif method == "rank":
            return self._rank_normalize(raw_scores)
        elif method == "percentile":
            return self._percentile_normalize(raw_scores)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def normalize_dataframe(
        self,
        feature_matrix: pd.DataFrame,
        method: Optional[str] = None,
        winsorize_std: Optional[float] = None,
    ) -> pd.DataFrame:
        """Normalise each column of a feature matrix cross-sectionally.

        Args:
            feature_matrix: DataFrame indexed by ticker, columns are features.
            method: Normalisation method (applied to all columns).
            winsorize_std: Winsorisation threshold.

        Returns:
            Normalised DataFrame with same shape and index.
        """
        result = pd.DataFrame(index=feature_matrix.index)
        for col in feature_matrix.columns:
            result[col] = self.normalize(
                feature_matrix[col], method=method, winsorize_std=winsorize_std
            )
        return result

    def _zscore_normalize(
        self, scores: pd.Series, winsorize_std: float
    ) -> pd.Series:
        """Z-score normalisation with winsorisation.

        Winsorisation clips outliers before computing z-scores:
        1. Compute mean and std of the raw scores.
        2. Clip values outside [mean - winsorize_std*std, mean + winsorize_std*std].
        3. Recompute mean and std on clipped values.
        4. Return (clipped - mean) / std.
        """
        clean = scores.dropna()
        if len(clean) < 2:
            return pd.Series(0.0, index=scores.index)

        if winsorize_std and winsorize_std > 0:
            mean = clean.mean()
            std = clean.std()
            if std > 0:
                lower = mean - winsorize_std * std
                upper = mean + winsorize_std * std
                clipped = scores.clip(lower=lower, upper=upper)
            else:
                clipped = scores
        else:
            clipped = scores

        clean_clipped = clipped.dropna()
        mean = clean_clipped.mean()
        std = clean_clipped.std()

        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=scores.index)

        return (clipped - mean) / std

    def _rank_normalize(self, scores: pd.Series) -> pd.Series:
        """Rank normalisation producing a uniform distribution in [-1, 1].

        NaN values remain NaN. Non-NaN values are ranked and linearly
        mapped to [-1, 1].
        """
        ranked = scores.rank(method="average", na_option="keep")
        n_valid = scores.notna().sum()
        if n_valid < 2:
            return pd.Series(0.0, index=scores.index)
        # Map ranks from [1, n_valid] to [-1, 1]
        normalized = 2.0 * (ranked - 1) / (n_valid - 1) - 1.0
        return normalized

    def _percentile_normalize(self, scores: pd.Series) -> pd.Series:
        """Percentile normalisation producing values in [0, 1].

        NaN values remain NaN. Non-NaN values are mapped to their
        percentile rank.
        """
        ranked = scores.rank(method="average", na_option="keep", pct=True)
        return ranked
