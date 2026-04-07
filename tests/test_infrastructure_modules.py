"""Tests for infrastructure and data layer modules added in Phase 4.

Covers:
- LiveDataStreamAdapter (data_layer)
- DockerEnvironmentSetup (infrastructure)
- CICDPipelineManager (infrastructure)
- DatasetVersionControl (infrastructure)
- DisasterRecoveryManager (infrastructure)
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from quant_fund.data_layer.live_data_stream_adapter import (

    FeedProvider,
    LiveDataStreamAdapter,
    NoOpFeedProvider,
)
from quant_fund.infrastructure.ci_cd_pipeline_manager import CICDPipelineManager
from quant_fund.infrastructure.dataset_version_control import DatasetVersionControl
from quant_fund.infrastructure.disaster_recovery_manager import (
    BACKUPABLE_COMPONENTS,
    RECOVERY_SCENARIOS,
    DisasterRecoveryManager,
)

pytestmark = [pytest.mark.tier2]
from quant_fund.infrastructure.docker_environment_setup import (
    VALID_ENVIRONMENTS,
    DockerEnvironmentSetup,
)


# ── Helpers ────────────────────────────────────────────────────────


class StubFeedProvider(FeedProvider):
    """Test feed provider that returns deterministic data."""

    def __init__(self, tickers=None):
        self._tickers = tickers or ["AAPL", "MSFT"]
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def fetch_latest_bars(self, tickers: List[str]) -> pd.DataFrame:
        rows = []
        now = pd.Timestamp.now().normalize()
        for t in tickers:
            rows.append({
                "date": now,
                "ticker": t,
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 103.0,
                "volume": 1_000_000,
            })
        df = pd.DataFrame(rows).set_index(["date", "ticker"])
        return df

    def fetch_snapshot(
        self, tickers: List[str], fields: List[str]
    ) -> pd.DataFrame:
        rows = []
        now = pd.Timestamp.now().normalize()
        for t in tickers:
            row = {"date": now, "ticker": t}
            for f in fields:
                row[f] = 100.0
            rows.append(row)
        return pd.DataFrame(rows).set_index(["date", "ticker"])


# ═══════════════════════════════════════════════════════════════════
# LiveDataStreamAdapter tests
# ═══════════════════════════════════════════════════════════════════


class TestLiveDataStreamAdapter:
    """Tests for LiveDataStreamAdapter."""

    def test_init_defaults(self):
        adapter = LiveDataStreamAdapter()
        assert adapter.subscribed_tickers == []

    def test_subscribe_unsubscribe(self):
        adapter = LiveDataStreamAdapter()
        adapter.subscribe(["AAPL", "MSFT"])
        assert adapter.subscribed_tickers == ["AAPL", "MSFT"]
        adapter.unsubscribe(["AAPL"])
        assert adapter.subscribed_tickers == ["MSFT"]

    def test_noop_provider_returns_empty(self):
        adapter = LiveDataStreamAdapter()
        adapter.subscribe(["AAPL"])
        bars = adapter.get_latest_bar()
        assert bars.empty

    def test_stub_feed_get_latest_bar(self):
        feed = StubFeedProvider()
        adapter = LiveDataStreamAdapter(
            config={"live_feed": {"polling_interval_sec": 0.1, "buffer_size": 10}},
            feed_provider=feed,
        )
        adapter.subscribe(["AAPL", "MSFT"])
        adapter.connect()
        time.sleep(0.3)  # allow 1-2 poll cycles
        bars = adapter.get_latest_bar()
        adapter.disconnect()
        assert not bars.empty
        assert "close" in bars.columns

    def test_on_bar_callback(self):
        received = []
        feed = StubFeedProvider()
        adapter = LiveDataStreamAdapter(
            config={"live_feed": {"polling_interval_sec": 0.1}},
            feed_provider=feed,
        )
        adapter.subscribe(["AAPL"])
        adapter.on_bar(lambda df: received.append(df))
        adapter.connect()
        time.sleep(0.3)
        adapter.disconnect()
        assert len(received) > 0

    def test_context_manager(self):
        feed = StubFeedProvider()
        with LiveDataStreamAdapter(
            config={"live_feed": {"polling_interval_sec": 0.1}},
            feed_provider=feed,
        ) as adapter:
            assert feed.connected
        assert not feed.connected

    def test_get_snapshot_delegates_to_feed(self):
        feed = StubFeedProvider()
        adapter = LiveDataStreamAdapter(feed_provider=feed)
        adapter.subscribe(["AAPL"])
        snap = adapter.get_snapshot(["AAPL"], ["close", "volume"])
        assert not snap.empty
        assert "close" in snap.columns

    def test_get_buffered_bars_empty(self):
        adapter = LiveDataStreamAdapter()
        bars = adapter.get_buffered_bars()
        assert bars.empty

    def test_duplicate_subscribe(self):
        adapter = LiveDataStreamAdapter()
        adapter.subscribe(["AAPL", "AAPL"])
        assert adapter.subscribed_tickers == ["AAPL"]

    def test_multiindex_schema(self):
        feed = StubFeedProvider()
        adapter = LiveDataStreamAdapter(
            config={"live_feed": {"polling_interval_sec": 0.1}},
            feed_provider=feed,
        )
        adapter.subscribe(["AAPL"])
        adapter.connect()
        time.sleep(0.3)
        bars = adapter.get_latest_bar()
        adapter.disconnect()
        assert bars.index.names == ["date", "ticker"]


# ═══════════════════════════════════════════════════════════════════
# DockerEnvironmentSetup tests
# ═══════════════════════════════════════════════════════════════════


class TestDockerEnvironmentSetup:
    """Tests for DockerEnvironmentSetup."""

    @pytest.fixture
    def setup(self):
        return DockerEnvironmentSetup(config={
            "environment": "local_research",
            "timezone": "America/New_York",
            "base_currency": "USD",
            "log_level": "INFO",
        })

    def test_generate_dockerfile_all_environments(self, setup):
        for env in VALID_ENVIRONMENTS:
            dockerfile = setup.generate_dockerfile(env)
            assert "FROM" in dockerfile
            assert "COPY" in dockerfile

    def test_generate_dockerfile_invalid_environment(self, setup):
        with pytest.raises(ValueError):
            setup.generate_dockerfile("invalid_env")

    def test_compose_config_has_app_service(self, setup):
        compose = setup.generate_compose_config("local_research")
        assert "app" in compose["services"]

    def test_live_trading_has_monitoring_services(self, setup):
        compose = setup.generate_compose_config("live_trading")
        assert "prometheus" in compose["services"]
        assert "grafana" in compose["services"]

    def test_environment_variables_per_env(self, setup):
        for env in VALID_ENVIRONMENTS:
            env_vars = setup.get_environment_variables(env)
            assert env_vars["QUANTFUND_ENV"] == env
            assert "LOG_LEVEL" in env_vars

    def test_live_trading_env_vars(self, setup):
        env_vars = setup.get_environment_variables("live_trading")
        assert env_vars["KILL_SWITCH_ENABLED"] == "true"
        assert env_vars["BROKER_MODE"] == "live"

    def test_resource_limits_escalate(self, setup):
        research = setup.get_resource_limits("local_research")
        paper = setup.get_resource_limits("paper_trading")
        live = setup.get_resource_limits("live_trading")
        assert research["cpus"] < paper["cpus"] < live["cpus"]

    def test_validate_environment_valid(self, setup):
        valid, issues = setup.validate_environment("local_research")
        assert valid
        assert issues == []

    def test_validate_environment_invalid(self, setup):
        valid, issues = setup.validate_environment("invalid")
        assert not valid

    def test_required_services_increase(self, setup):
        research = setup.get_required_services("local_research")
        live = setup.get_required_services("live_trading")
        assert len(research) < len(live)

    def test_healthcheck_config(self, setup):
        hc = setup.generate_healthcheck_config()
        assert "test" in hc
        assert "interval" in hc
        assert "retries" in hc


# ═══════════════════════════════════════════════════════════════════
# CICDPipelineManager tests
# ═══════════════════════════════════════════════════════════════════


class TestCICDPipelineManager:
    """Tests for CICDPipelineManager."""

    @pytest.fixture
    def manager(self):
        return CICDPipelineManager()

    def test_generate_pipeline_config(self, manager):
        config = manager.generate_pipeline_config("dev")
        assert config["environment"] == "dev"
        assert len(config["stages"]) > 0

    def test_invalid_environment_raises(self, manager):
        with pytest.raises(ValueError):
            manager.generate_pipeline_config("production")

    def test_test_stages_mandatory(self, manager):
        stages = manager.get_test_stages()
        assert len(stages) == 3
        assert all(s["required"] for s in stages)
        names = [s["name"] for s in stages]
        assert "lookahead_bias_tests" in names

    def test_validation_stages(self, manager):
        stages = manager.get_validation_stages()
        assert len(stages) == 2
        names = [s["name"] for s in stages]
        assert "data_validation" in names
        assert "risk_checks" in names

    def test_live_deployment_has_gates(self, manager):
        stages = manager.get_deployment_stages("live")
        names = [s["name"] for s in stages]
        assert "paper_trading_validation" in names
        assert "risk_team_approval" in names

    def test_dev_deployment_no_gates(self, manager):
        stages = manager.get_deployment_stages("dev")
        names = [s["name"] for s in stages]
        assert "paper_trading_validation" not in names

    def test_validate_pipeline_all_pass(self, manager):
        results = {
            "unit_tests": "passed",
            "integration_tests": "passed",
            "lookahead_bias_tests": "passed",
            "data_validation": "passed",
            "risk_checks": "passed",
        }
        success, issues = manager.validate_pipeline_result(results)
        assert success
        assert issues == []

    def test_validate_pipeline_missing_stage(self, manager):
        results = {
            "unit_tests": "passed",
            "integration_tests": "passed",
            # missing lookahead_bias_tests
            "data_validation": "passed",
            "risk_checks": "passed",
        }
        success, issues = manager.validate_pipeline_result(results)
        assert not success

    def test_validate_pipeline_failed_test(self, manager):
        results = {
            "unit_tests": "failed",
            "integration_tests": "passed",
            "lookahead_bias_tests": "passed",
            "data_validation": "passed",
            "risk_checks": "passed",
        }
        success, issues = manager.validate_pipeline_result(results)
        assert not success

    def test_rollback_plan_live_blue_green(self, manager):
        plan = manager.get_rollback_plan("live")
        assert plan["strategy"] == "blue_green"

    def test_rollback_plan_dev_in_place(self, manager):
        plan = manager.get_rollback_plan("dev")
        assert plan["strategy"] == "in_place"

    def test_deployment_readiness_all_pass(self, manager):
        ready, blockers = manager.check_deployment_readiness(
            strategy_id="test_strat",
            approval_state={"risk_team": True},
            test_results={
                "unit": "passed",
                "integration": "passed",
                "lookahead_bias": "passed",
                "paper_trading": "passed",
            },
        )
        assert ready
        assert blockers == []

    def test_deployment_readiness_missing_risk_approval(self, manager):
        ready, blockers = manager.check_deployment_readiness(
            strategy_id="test_strat",
            approval_state={"risk_team": False},
            test_results={
                "unit": "passed",
                "integration": "passed",
                "lookahead_bias": "passed",
                "paper_trading": "passed",
            },
        )
        assert not ready
        assert any("risk team" in b.lower() for b in blockers)

    def test_deployment_readiness_missing_paper_trading(self, manager):
        ready, blockers = manager.check_deployment_readiness(
            strategy_id="test_strat",
            approval_state={"risk_team": True},
            test_results={
                "unit": "passed",
                "integration": "passed",
                "lookahead_bias": "passed",
            },
        )
        assert not ready


# ═══════════════════════════════════════════════════════════════════
# DatasetVersionControl tests
# ═══════════════════════════════════════════════════════════════════


class TestDatasetVersionControl:
    """Tests for DatasetVersionControl."""

    @pytest.fixture
    def dvc(self):
        return DatasetVersionControl()

    @pytest.fixture
    def temp_file(self, tmp_path):
        p = tmp_path / "test_data.csv"
        p.write_text("col1,col2\n1,2\n3,4\n")
        return str(p)

    @pytest.fixture
    def temp_dir(self, tmp_path):
        d = tmp_path / "dataset"
        d.mkdir()
        (d / "a.csv").write_text("x,y\n1,2\n")
        (d / "b.csv").write_text("a,b\n3,4\n")
        return str(d)

    def test_register_and_get_version(self, dvc, temp_file):
        vid = dvc.register_dataset("test_ds", temp_file, metadata={"rows": 2})
        version = dvc.get_version("test_ds", vid)
        assert version["name"] == "test_ds"
        assert version["metadata"]["rows"] == 2

    def test_list_versions(self, dvc, temp_file):
        v1 = dvc.register_dataset("ds", temp_file)
        v2 = dvc.register_dataset("ds", temp_file)
        versions = dvc.list_versions("ds")
        assert len(versions) == 2
        assert versions[0]["version_id"] == v1
        assert versions[1]["version_id"] == v2

    def test_get_latest_version(self, dvc, temp_file):
        dvc.register_dataset("ds", temp_file)
        v2 = dvc.register_dataset("ds", temp_file)
        latest = dvc.get_latest_version("ds")
        assert latest["version_id"] == v2

    def test_get_version_not_found(self, dvc):
        with pytest.raises(KeyError):
            dvc.get_version("missing", "abc")

    def test_compute_checksum_file(self, dvc, temp_file):
        checksum = dvc.compute_checksum(temp_file)
        assert len(checksum) == 64  # SHA-256 hex

    def test_compute_checksum_directory(self, dvc, temp_dir):
        checksum = dvc.compute_checksum(temp_dir)
        assert len(checksum) == 64

    def test_compute_checksum_deterministic(self, dvc, temp_file):
        c1 = dvc.compute_checksum(temp_file)
        c2 = dvc.compute_checksum(temp_file)
        assert c1 == c2

    def test_validate_integrity_pass(self, dvc, temp_file):
        vid = dvc.register_dataset("ds", temp_file)
        assert dvc.validate_integrity("ds", vid)

    def test_validate_integrity_fail_modified(self, dvc, temp_file):
        vid = dvc.register_dataset("ds", temp_file)
        # Modify the file
        with open(temp_file, "a") as f:
            f.write("5,6\n")
        assert not dvc.validate_integrity("ds", vid)

    def test_validate_integrity_fail_deleted(self, dvc, temp_file):
        vid = dvc.register_dataset("ds", temp_file)
        os.remove(temp_file)
        assert not dvc.validate_integrity("ds", vid)

    def test_tag_version(self, dvc, temp_file):
        vid = dvc.register_dataset("ds", temp_file)
        assert dvc.tag_version("ds", vid, "production")
        tagged = dvc.get_by_tag("ds", "production")
        assert tagged["version_id"] == vid

    def test_tag_version_upsert(self, dvc, temp_file):
        v1 = dvc.register_dataset("ds", temp_file)
        v2 = dvc.register_dataset("ds", temp_file)
        dvc.tag_version("ds", v1, "prod")
        dvc.tag_version("ds", v2, "prod")
        tagged = dvc.get_by_tag("ds", "prod")
        assert tagged["version_id"] == v2

    def test_get_by_tag_not_found(self, dvc):
        with pytest.raises(KeyError):
            dvc.get_by_tag("ds", "nonexistent")

    def test_create_snapshot(self, dvc, temp_file):
        dvc.register_dataset("ds", temp_file)
        snap_id = dvc.create_snapshot("ds", "test snapshot")
        assert len(snap_id) == 32  # uuid hex

    def test_context_manager(self, temp_file):
        with DatasetVersionControl() as dvc:
            vid = dvc.register_dataset("ds", temp_file)
            assert vid

    def test_checksum_missing_path(self, dvc):
        with pytest.raises(FileNotFoundError):
            dvc.compute_checksum("/nonexistent/path")


# ═══════════════════════════════════════════════════════════════════
# DisasterRecoveryManager tests
# ═══════════════════════════════════════════════════════════════════


class TestDisasterRecoveryManager:
    """Tests for DisasterRecoveryManager."""

    @pytest.fixture
    def dr(self):
        return DisasterRecoveryManager(config={
            "backup_dir": "/tmp/test_backups",
            "max_backups": 5,
            "failover_targets": {
                "state_store": "replica-store:5432",
                "trade_recorder": "replica-recorder:5432",
            },
            "backup_feed_providers": ["alpaca", "polygon"],
            "recovery_contacts": [{"name": "ops", "email": "ops@example.com"}],
        })

    def test_create_backup(self, dr):
        result = dr.create_backup(["state_store", "config"])
        assert "backup_id" in result
        assert result["components"] == ["state_store", "config"]

    def test_create_backup_empty_raises(self, dr):
        with pytest.raises(ValueError):
            dr.create_backup([])

    def test_create_backup_unknown_component_raises(self, dr):
        with pytest.raises(ValueError):
            dr.create_backup(["unknown_component"])

    def test_list_backups(self, dr):
        dr.create_backup(["config"])
        dr.create_backup(["state_store"])
        backups = dr.list_backups()
        assert len(backups) == 2
        # Most recent first
        assert backups[0]["components"] == ["state_store"]

    def test_validate_backup_valid(self, dr):
        result = dr.create_backup(["state_store"])
        valid, issues = dr.validate_backup(result["backup_id"])
        assert valid
        assert issues == []

    def test_validate_backup_not_found(self, dr):
        valid, issues = dr.validate_backup("nonexistent")
        assert not valid

    def test_retention_limit(self, dr):
        for i in range(7):
            dr.create_backup(["config"])
        assert len(dr.list_backups()) == 5

    def test_all_recovery_scenarios(self, dr):
        for scenario in RECOVERY_SCENARIOS:
            plan = dr.get_recovery_plan(scenario)
            assert plan["scenario"] == scenario
            assert "severity" in plan
            assert "steps" in plan
            assert len(plan["steps"]) > 0

    def test_recovery_plan_unknown_scenario(self, dr):
        with pytest.raises(ValueError):
            dr.get_recovery_plan("unknown_scenario")

    def test_database_corruption_plan(self, dr):
        plan = dr.get_recovery_plan("database_corruption")
        assert plan["severity"] == "critical"
        actions = [s["action"] for s in plan["steps"]]
        assert "halt_trading" in actions
        assert "restore_backup" in actions

    def test_broker_disconnect_plan(self, dr):
        plan = dr.get_recovery_plan("broker_disconnect")
        actions = [s["action"] for s in plan["steps"]]
        assert "halt_new_orders" in actions

    def test_kill_switch_plan(self, dr):
        plan = dr.get_recovery_plan("kill_switch_triggered")
        actions = [s["action"] for s in plan["steps"]]
        assert "reset_kill_switch" in actions
        assert "gradual_resume" in actions

    def test_data_feed_failure_with_backup(self, dr):
        plan = dr.get_recovery_plan("data_feed_failure")
        assert plan["estimated_recovery_minutes"] == 15

    def test_data_feed_failure_without_backup(self):
        dr = DisasterRecoveryManager(config={})
        plan = dr.get_recovery_plan("data_feed_failure")
        assert plan["estimated_recovery_minutes"] == 45

    def test_position_discrepancy_plan(self, dr):
        plan = dr.get_recovery_plan("position_discrepancy")
        actions = [s["action"] for s in plan["steps"]]
        assert "reconcile_positions" in actions

    def test_system_health_snapshot(self, dr):
        dr.create_backup(["state_store"])
        snapshot = dr.get_system_health_snapshot()
        assert "timestamp" in snapshot
        assert snapshot["backup_status"]["total_backups"] == 1

    def test_check_recovery_readiness_ready(self, dr):
        dr.create_backup(list(BACKUPABLE_COMPONENTS))
        ready, issues = dr.check_recovery_readiness()
        assert ready
        assert issues == []

    def test_check_recovery_readiness_no_backups(self):
        dr = DisasterRecoveryManager(config={
            "failover_targets": {
                "state_store": "x",
                "trade_recorder": "x",
            },
            "backup_feed_providers": ["alpaca"],
            "recovery_contacts": [{"name": "ops"}],
        })
        ready, issues = dr.check_recovery_readiness()
        assert not ready
        assert any("backup" in i.lower() for i in issues)

    def test_failover_config_configured(self, dr):
        config = dr.get_failover_config("state_store")
        assert config["configured"]
        assert config["failover_target"] == "replica-store:5432"

    def test_failover_config_not_configured(self, dr):
        config = dr.get_failover_config("config")
        assert not config["configured"]

    def test_failover_priority(self, dr):
        positions = dr.get_failover_config("positions")
        config = dr.get_failover_config("config")
        assert positions["priority"] < config["priority"]
