"""Typed configuration dataclasses for core subsystems.

Replaces the ``cfg = config or {}; cfg.get("key", default)`` pattern
with structured dataclasses that provide IDE autocomplete, typo
prevention, and self-documenting defaults.

Usage:
    # From a raw dict (e.g. loaded from YAML):
    cfg = EngineConfig.from_dict(raw_dict)
    engine = TradingEngine(config=cfg)

    # Direct construction:
    cfg = EngineConfig(tick_interval_s=0.5, research_interval_s=1800)
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for TradingEngine."""

    tick_interval_s: float = 1.0
    convergence_interval_s: float = 5.0
    research_interval_s: float = 3600.0
    snapshot_interval_s: float = 60.0
    reconciliation_interval_s: float = 300.0
    health_check_interval_s: float = 60.0
    state_db_path: str = ":memory:"
    strategy_id: str = "default"
    max_impact_bps: float = 50.0
    synchronous: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "EngineConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KillSwitchConfig:
    """Configuration for portfolio kill switch."""

    drawdown_limit: float = 0.15
    initial_nav: float = 1_000_000.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "KillSwitchConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class DrawdownConfig:
    """Configuration for drawdown monitor."""

    initial_nav: float = 1_000_000.0
    drawdown_warning: float = 0.05
    drawdown_alert: float = 0.10
    drawdown_limit: float = 0.15

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "DrawdownConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class LeverageConfig:
    """Configuration for leverage controller."""

    max_leverage: float = 1.5

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "LeverageConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class ExposureConfig:
    """Configuration for exposure monitor."""

    max_leverage: float = 1.0
    max_sector_exposure: float = 0.40
    max_single_name_exposure: float = 0.10

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "ExposureConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class ResearchConfig:
    """Configuration for research runner."""

    universe: List[str] = field(default_factory=list)
    lookback_days: int = 252
    retrain_frequency: int = 24

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "ResearchConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class ConstraintConfig:
    """Configuration for portfolio constraint engine."""

    max_position_size: float = 0.10
    max_sector_exposure: float = 0.40
    max_leverage: float = 1.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "ConstraintConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class BrokerConfig:
    """Configuration for simulation broker."""

    initial_cash: float = 1_000_000.0
    half_spread_bps: float = 5.0
    market_impact_bps: float = 2.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "BrokerConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for PnL dashboard."""

    initial_nav: float = 1_000_000.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "DashboardConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class HealthMonitorConfig:
    """Configuration for system health monitor."""

    max_data_lag_s: float = 300.0

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "HealthMonitorConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class ModelHealthConfig:
    """Configuration for model health monitor."""

    event_log_path: str = "logs/health_events.jsonl"
    prediction_window: int = 500
    drift_z_threshold: float = 2.5
    staleness_hours: int = 48

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "ModelHealthConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass(frozen=True)
class LiveDataConfig:
    """Configuration for live data stream adapter."""

    polling_interval_sec: float = 60.0
    buffer_size: int = 1000

    @classmethod
    def from_dict(cls, d: Optional[dict] = None) -> "LiveDataConfig":
        if not d:
            return cls()
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
