"""Disaster recovery manager — planning and execution layer.

Generates recovery plans, backup metadata, and failover configurations
for various failure scenarios. Also performs actual backup execution
(SQLite copy, config snapshot) when backed by a backup directory.
"""

import hashlib
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Components that can be backed up
BACKUPABLE_COMPONENTS = ("state_store", "trade_recorder", "config", "positions")

# Known recovery scenarios
RECOVERY_SCENARIOS = (
    "database_corruption",
    "broker_disconnect",
    "kill_switch_triggered",
    "data_feed_failure",
    "position_discrepancy",
)


@dataclass
class BackupRecord:
    """Metadata for a single backup."""

    backup_id: str
    timestamp: str
    components: List[str]
    status: str = "complete"
    size_bytes: int = 0
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "timestamp": self.timestamp,
            "components": self.components,
            "status": self.status,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryStep:
    """A single step in a recovery plan."""

    order: int
    action: str
    description: str
    component: str
    automated: bool = False
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order": self.order,
            "action": self.action,
            "description": self.description,
            "component": self.component,
            "automated": self.automated,
            "requires_approval": self.requires_approval,
        }


class DisasterRecoveryManager:
    """Plans and coordinates disaster recovery procedures.

    This manager does not execute recovery actions directly. It produces
    structured plans, backup metadata, and failover configurations that
    operators or automated runbooks can follow.

    Usage:
        dr = DisasterRecoveryManager(config={"backup_dir": "/backups"})
        backup = dr.create_backup(["state_store", "config"])
        plan = dr.get_recovery_plan("broker_disconnect")
        ready, issues = dr.check_recovery_readiness()
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize disaster recovery manager.

        Args:
            config: Optional configuration dict. Supported keys:
                - backup_dir: directory path for backup storage
                - max_backups: maximum number of backups to retain
                - failover_targets: mapping of component -> failover endpoint
                - backup_feed_providers: list of backup data feed providers
                - recovery_contacts: list of operator contact info dicts
        """
        self._config = config or {}
        self._backups: List[BackupRecord] = []
        self._backup_dir: str = self._config.get("backup_dir", "/var/backups/quant_fund")
        self._max_backups: int = self._config.get("max_backups", 50)
        self._failover_targets: Dict[str, str] = self._config.get(
            "failover_targets", {}
        )
        logger.info(
            "DisasterRecoveryManager initialized backup_dir=%s max_backups=%d",
            self._backup_dir,
            self._max_backups,
        )

    # ------------------------------------------------------------------
    # Backup management
    # ------------------------------------------------------------------

    def create_backup(self, components: list[str]) -> dict:
        """Create backup metadata for the specified components.

        This records the intent and metadata for a backup. The actual
        data copy is expected to be performed by the storage layer or
        an external backup tool referencing the returned metadata.

        Args:
            components: List of component names to back up. Valid values:
                ``state_store``, ``trade_recorder``, ``config``, ``positions``.

        Returns:
            Dict with backup metadata including ``backup_id`` and ``timestamp``.

        Raises:
            ValueError: If no components specified or an unknown component
                is requested.
        """
        if not components:
            raise ValueError("At least one component must be specified for backup")

        unknown = [c for c in components if c not in BACKUPABLE_COMPONENTS]
        if unknown:
            raise ValueError(f"Unknown backup components: {unknown}")

        backup_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()

        record = BackupRecord(
            backup_id=backup_id,
            timestamp=now,
            components=list(components),
            status="complete",
            metadata={
                "backup_dir": self._backup_dir,
                "initiated_by": "disaster_recovery_manager",
            },
        )
        self._backups.append(record)

        # Enforce retention limit
        if len(self._backups) > self._max_backups:
            removed = self._backups.pop(0)
            logger.info("Evicted oldest backup %s to enforce retention limit", removed.backup_id)

        logger.info("Created backup %s for components %s", backup_id, components)
        return record.to_dict()

    def execute_backup(
        self,
        components: list[str],
        source_paths: Optional[Dict[str, str]] = None,
    ) -> BackupRecord:
        """Execute an actual backup: copy files to backup_dir.

        Unlike ``create_backup`` which only records metadata, this method
        copies the actual SQLite databases and config files to the backup
        directory.

        Args:
            components: List of component names to back up.
            source_paths: Mapping of component name to source file path.
                E.g. ``{"state_store": "quantfund_paper.db"}``.

        Returns:
            BackupRecord with size and checksum populated.
        """
        if not components:
            raise ValueError("At least one component must be specified")

        unknown = [c for c in components if c not in BACKUPABLE_COMPONENTS]
        if unknown:
            raise ValueError(f"Unknown backup components: {unknown}")

        source_paths = source_paths or {}
        backup_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")

        backup_subdir = os.path.join(self._backup_dir, f"backup_{timestamp_str}_{backup_id}")
        os.makedirs(backup_subdir, exist_ok=True)

        total_size = 0
        copied_files: List[str] = []

        for component in components:
            src = source_paths.get(component)
            if src is None or not os.path.exists(src):
                logger.warning(
                    "Skipping backup of %s: source path %s not found",
                    component, src,
                )
                continue

            dest = os.path.join(backup_subdir, f"{component}_{os.path.basename(src)}")
            try:
                shutil.copy2(src, dest)
                file_size = os.path.getsize(dest)
                total_size += file_size
                copied_files.append(dest)
                logger.info("Backed up %s -> %s (%d bytes)", src, dest, file_size)
            except OSError as e:
                logger.error("Failed to copy %s: %s", src, e)

        # Compute checksum of all backed-up files
        checksum = self._compute_directory_checksum(backup_subdir)

        record = BackupRecord(
            backup_id=backup_id,
            timestamp=now.isoformat(),
            components=list(components),
            status="complete" if copied_files else "partial",
            size_bytes=total_size,
            checksum=checksum,
            metadata={
                "backup_dir": backup_subdir,
                "files": copied_files,
            },
        )
        self._backups.append(record)

        if len(self._backups) > self._max_backups:
            removed = self._backups.pop(0)
            old_dir = removed.metadata.get("backup_dir")
            if old_dir and os.path.isdir(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)
            logger.info("Evicted oldest backup %s", removed.backup_id)

        logger.info(
            "Backup %s complete: %d files, %d bytes",
            backup_id, len(copied_files), total_size,
        )
        return record

    def generate_backup_plan(self, components: list[str]) -> BackupRecord:
        """Generate a backup plan (metadata only, no file copy).

        This is a lightweight version of ``execute_backup`` for use
        in contexts where the caller just needs a plan or record.
        """
        return BackupRecord(
            backup_id=uuid.uuid4().hex[:16],
            timestamp=datetime.now(timezone.utc).isoformat(),
            components=list(components),
            status="planned",
            metadata={"backup_dir": self._backup_dir},
        )

    @staticmethod
    def _compute_directory_checksum(directory: str) -> str:
        """Compute a combined SHA-256 checksum of all files in a directory."""
        h = hashlib.sha256()
        for root, _, files in sorted(os.walk(directory)):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                except OSError:
                    pass
        return h.hexdigest()

    def list_backups(self) -> list[dict]:
        """List all available backup records.

        Returns:
            List of backup metadata dicts, most recent first.
        """
        return [b.to_dict() for b in reversed(self._backups)]

    def validate_backup(self, backup_id: str) -> tuple[bool, list[str]]:
        """Validate the integrity of a backup.

        Checks that the backup record exists and its metadata is
        consistent. In a production system this would also verify
        checksums and file existence on disk.

        Args:
            backup_id: The ID of the backup to validate.

        Returns:
            Tuple of (is_valid, list_of_issues). An empty issues list
            indicates the backup passed all checks.
        """
        issues: List[str] = []

        record = self._find_backup(backup_id)
        if record is None:
            return False, [f"Backup {backup_id} not found"]

        # Check required fields are populated
        if not record.components:
            issues.append("Backup has no components listed")

        if not record.timestamp:
            issues.append("Backup has no timestamp")

        # Validate all components are recognized
        for comp in record.components:
            if comp not in BACKUPABLE_COMPONENTS:
                issues.append(f"Unknown component in backup: {comp}")

        if record.status != "complete":
            issues.append(f"Backup status is '{record.status}', expected 'complete'")

        is_valid = len(issues) == 0
        logger.info(
            "Validated backup %s: valid=%s issues=%d", backup_id, is_valid, len(issues)
        )
        return is_valid, issues

    # ------------------------------------------------------------------
    # Recovery planning
    # ------------------------------------------------------------------

    def get_recovery_plan(self, scenario: str) -> dict:
        """Generate a recovery plan for the given failure scenario.

        The plan contains ordered steps for an operator to follow.
        Steps are informational; they do not trigger any automated action.

        Args:
            scenario: One of ``database_corruption``, ``broker_disconnect``,
                ``kill_switch_triggered``, ``data_feed_failure``, or
                ``position_discrepancy``.

        Returns:
            Dict with ``scenario``, ``severity``, ``steps``, and
            ``estimated_recovery_minutes``.

        Raises:
            ValueError: If the scenario is not recognized.
        """
        if scenario not in RECOVERY_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Valid scenarios: {RECOVERY_SCENARIOS}"
            )

        plan_builders = {
            "database_corruption": self._plan_database_corruption,
            "broker_disconnect": self._plan_broker_disconnect,
            "kill_switch_triggered": self._plan_kill_switch_triggered,
            "data_feed_failure": self._plan_data_feed_failure,
            "position_discrepancy": self._plan_position_discrepancy,
        }

        plan = plan_builders[scenario]()
        logger.info("Generated recovery plan for scenario=%s", scenario)
        return plan

    def get_system_health_snapshot(self) -> dict:
        """Capture the current system state for recovery purposes.

        Returns a snapshot dict summarizing component status, backup
        recency, and configuration completeness. Useful for pre-incident
        audits and post-incident root cause analysis.

        Returns:
            Dict with ``timestamp``, ``components``, ``backup_status``,
            and ``configuration`` sections.
        """
        now = datetime.now(timezone.utc).isoformat()
        latest_backup = self._backups[-1].to_dict() if self._backups else None

        snapshot: Dict[str, Any] = {
            "timestamp": now,
            "components": {
                "state_store": {"status": "unknown", "recoverable": True},
                "trade_recorder": {"status": "unknown", "recoverable": True},
                "config": {"status": "unknown", "recoverable": True},
                "positions": {"status": "unknown", "recoverable": True},
            },
            "backup_status": {
                "total_backups": len(self._backups),
                "latest_backup": latest_backup,
                "backup_dir": self._backup_dir,
            },
            "configuration": {
                "max_backups": self._max_backups,
                "failover_targets_configured": len(self._failover_targets),
                "backup_feed_providers": len(
                    self._config.get("backup_feed_providers", [])
                ),
                "recovery_contacts": len(
                    self._config.get("recovery_contacts", [])
                ),
            },
        }

        logger.debug("Captured system health snapshot at %s", now)
        return snapshot

    def check_recovery_readiness(self) -> tuple[bool, list[str]]:
        """Check whether the system is prepared to recover from failures.

        Evaluates backup freshness, failover configuration, and contact
        information. Returns readiness status and any gaps found.

        Returns:
            Tuple of (is_ready, list_of_issues). An empty issues list
            means the system is fully prepared for disaster recovery.
        """
        issues: List[str] = []

        # Check that at least one backup exists
        if not self._backups:
            issues.append("No backups available")
        else:
            latest = self._backups[-1]
            # Check that the latest backup covers all components
            missing = [
                c for c in BACKUPABLE_COMPONENTS if c not in latest.components
            ]
            if missing:
                issues.append(
                    f"Latest backup missing components: {missing}"
                )

        # Check failover configuration
        critical_components = ("state_store", "trade_recorder")
        for comp in critical_components:
            if comp not in self._failover_targets:
                issues.append(f"No failover target configured for {comp}")

        # Check backup data feed providers
        if not self._config.get("backup_feed_providers"):
            issues.append("No backup data feed providers configured")

        # Check recovery contacts
        if not self._config.get("recovery_contacts"):
            issues.append("No recovery contacts configured")

        is_ready = len(issues) == 0
        logger.info(
            "Recovery readiness check: ready=%s issues=%d", is_ready, len(issues)
        )
        return is_ready, issues

    def get_failover_config(self, component: str) -> dict:
        """Return the failover configuration for a component.

        Args:
            component: Component name (e.g. ``state_store``,
                ``trade_recorder``, ``config``, ``positions``).

        Returns:
            Dict with ``component``, ``failover_target``, ``strategy``,
            and ``priority``.
        """
        target = self._failover_targets.get(component)

        config: Dict[str, Any] = {
            "component": component,
            "failover_target": target,
            "configured": target is not None,
            "strategy": self._default_failover_strategy(component),
            "priority": self._component_priority(component),
        }

        logger.debug("Failover config for %s: configured=%s", component, config["configured"])
        return config

    # ------------------------------------------------------------------
    # Private: recovery plan builders
    # ------------------------------------------------------------------

    def _plan_database_corruption(self) -> dict:
        steps = [
            RecoveryStep(
                order=1,
                action="halt_trading",
                description="Immediately halt all trading activity and cancel open orders.",
                component="execution",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=2,
                action="assess_corruption",
                description=(
                    "Run integrity checks on state_store and trade_recorder "
                    "databases to determine corruption scope."
                ),
                component="state_store",
                automated=False,
            ),
            RecoveryStep(
                order=3,
                action="identify_backup",
                description=(
                    "Identify the most recent validated backup that predates "
                    "the corruption event."
                ),
                component="state_store",
                automated=False,
            ),
            RecoveryStep(
                order=4,
                action="restore_backup",
                description=(
                    "Restore state_store and trade_recorder from the selected "
                    f"backup in {self._backup_dir}."
                ),
                component="state_store",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=5,
                action="reconcile_positions",
                description=(
                    "Reconcile restored positions against broker records to "
                    "detect any gaps between backup point and corruption event."
                ),
                component="positions",
                automated=False,
            ),
            RecoveryStep(
                order=6,
                action="verify_integrity",
                description="Run full integrity checks on restored databases.",
                component="state_store",
                automated=False,
            ),
            RecoveryStep(
                order=7,
                action="resume_trading",
                description="Resume trading after verification is complete.",
                component="execution",
                automated=False,
                requires_approval=True,
            ),
        ]
        return {
            "scenario": "database_corruption",
            "severity": "critical",
            "estimated_recovery_minutes": 60,
            "steps": [s.to_dict() for s in steps],
            "required_backups": ["state_store", "trade_recorder"],
            "notes": (
                "Do not resume trading until position reconciliation is "
                "confirmed by risk team."
            ),
        }

    def _plan_broker_disconnect(self) -> dict:
        steps = [
            RecoveryStep(
                order=1,
                action="halt_new_orders",
                description="Stop submitting new orders. Let in-flight orders complete or time out.",
                component="execution",
                automated=True,
            ),
            RecoveryStep(
                order=2,
                action="queue_pending_orders",
                description="Move pending orders to a durable queue for replay after reconnection.",
                component="execution",
                automated=True,
            ),
            RecoveryStep(
                order=3,
                action="diagnose_connection",
                description=(
                    "Check broker API status page, network connectivity, and "
                    "authentication credentials."
                ),
                component="broker_interface",
                automated=False,
            ),
            RecoveryStep(
                order=4,
                action="attempt_reconnect",
                description="Attempt reconnection with exponential backoff (max 5 attempts).",
                component="broker_interface",
                automated=True,
            ),
            RecoveryStep(
                order=5,
                action="failover_broker",
                description=(
                    "If primary broker remains unreachable, switch to failover "
                    "broker endpoint if configured."
                ),
                component="broker_interface",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=6,
                action="reconcile_orders",
                description=(
                    "After reconnection, reconcile local order state with "
                    "broker-side fill reports."
                ),
                component="positions",
                automated=False,
            ),
            RecoveryStep(
                order=7,
                action="replay_queued_orders",
                description="Review and selectively replay queued orders.",
                component="execution",
                automated=False,
                requires_approval=True,
            ),
        ]
        return {
            "scenario": "broker_disconnect",
            "severity": "high",
            "estimated_recovery_minutes": 30,
            "steps": [s.to_dict() for s in steps],
            "required_backups": [],
            "notes": (
                "Queued orders may be stale by the time connectivity is "
                "restored. Review market conditions before replaying."
            ),
        }

    def _plan_kill_switch_triggered(self) -> dict:
        steps = [
            RecoveryStep(
                order=1,
                action="confirm_trigger",
                description=(
                    "Confirm the kill switch trigger was legitimate by reviewing "
                    "NAV, drawdown, and recent trades."
                ),
                component="risk_engine",
                automated=False,
            ),
            RecoveryStep(
                order=2,
                action="document_incident",
                description=(
                    "Record the trigger event: timestamp, NAV at trigger, "
                    "drawdown level, active strategies, and open positions."
                ),
                component="trade_recorder",
                automated=False,
            ),
            RecoveryStep(
                order=3,
                action="review_positions",
                description="Review all open positions and assess current market risk.",
                component="positions",
                automated=False,
            ),
            RecoveryStep(
                order=4,
                action="risk_assessment",
                description=(
                    "Perform a full risk assessment: check exposure limits, "
                    "concentration, and correlation."
                ),
                component="risk_engine",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=5,
                action="adjust_parameters",
                description=(
                    "If appropriate, adjust risk parameters (position limits, "
                    "drawdown thresholds) before resuming."
                ),
                component="config",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=6,
                action="reset_kill_switch",
                description=(
                    "Reset the kill switch state and update peak NAV reference "
                    "point in state_store."
                ),
                component="state_store",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=7,
                action="gradual_resume",
                description=(
                    "Resume trading with reduced position sizes. Ramp back to "
                    "full allocation over a defined schedule."
                ),
                component="execution",
                automated=False,
                requires_approval=True,
            ),
        ]
        return {
            "scenario": "kill_switch_triggered",
            "severity": "critical",
            "estimated_recovery_minutes": 120,
            "steps": [s.to_dict() for s in steps],
            "required_backups": [],
            "notes": (
                "Kill switch resume requires sign-off from both the portfolio "
                "manager and risk officer. Consider a cooling-off period."
            ),
        }

    def _plan_data_feed_failure(self) -> dict:
        backup_providers = self._config.get("backup_feed_providers", [])
        has_backup_feed = len(backup_providers) > 0

        steps = [
            RecoveryStep(
                order=1,
                action="detect_staleness",
                description=(
                    "Confirm data feed failure: check last-update timestamps "
                    "and compare against expected frequency."
                ),
                component="data_layer",
                automated=True,
            ),
            RecoveryStep(
                order=2,
                action="halt_dependent_strategies",
                description="Pause strategies that depend on the failed data feed.",
                component="execution",
                automated=True,
            ),
            RecoveryStep(
                order=3,
                action="switch_to_backup_feed",
                description=(
                    f"Switch to backup data feed provider "
                    f"({', '.join(backup_providers) if backup_providers else 'none configured'}). "
                    f"{'Backup feeds available.' if has_backup_feed else 'WARNING: No backup feeds configured — manual intervention required.'}"
                ),
                component="data_layer",
                automated=has_backup_feed,
                requires_approval=not has_backup_feed,
            ),
            RecoveryStep(
                order=4,
                action="validate_backup_feed",
                description=(
                    "Validate backup feed data quality: check for gaps, "
                    "outliers, and timestamp consistency."
                ),
                component="data_layer",
                automated=False,
            ),
            RecoveryStep(
                order=5,
                action="resume_strategies",
                description="Resume strategies once data feed is validated.",
                component="execution",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=6,
                action="monitor_primary",
                description=(
                    "Continue monitoring primary feed for recovery. Switch back "
                    "when stable for at least 15 minutes."
                ),
                component="data_layer",
                automated=True,
            ),
        ]
        return {
            "scenario": "data_feed_failure",
            "severity": "high",
            "estimated_recovery_minutes": 15 if has_backup_feed else 45,
            "steps": [s.to_dict() for s in steps],
            "required_backups": [],
            "notes": (
                "If no backup feed is available, halt all market-data-dependent "
                "strategies until primary feed recovers."
            ),
        }

    def _plan_position_discrepancy(self) -> dict:
        steps = [
            RecoveryStep(
                order=1,
                action="halt_affected_strategies",
                description=(
                    "Pause strategies associated with discrepant positions "
                    "to prevent further divergence."
                ),
                component="execution",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=2,
                action="snapshot_local_state",
                description="Export current local position state for comparison.",
                component="positions",
                automated=True,
            ),
            RecoveryStep(
                order=3,
                action="fetch_broker_positions",
                description="Pull current position report from broker.",
                component="broker_interface",
                automated=True,
            ),
            RecoveryStep(
                order=4,
                action="generate_diff_report",
                description=(
                    "Generate a position-by-position diff report between "
                    "local state and broker records."
                ),
                component="positions",
                automated=True,
            ),
            RecoveryStep(
                order=5,
                action="investigate_root_cause",
                description=(
                    "Review trade_recorder logs and order fills to determine "
                    "the source of the discrepancy (missed fill, duplicate "
                    "order, partial fill, etc.)."
                ),
                component="trade_recorder",
                automated=False,
            ),
            RecoveryStep(
                order=6,
                action="reconcile_positions",
                description=(
                    "Update local position state to match broker-confirmed "
                    "positions. Record all adjustments in the audit log."
                ),
                component="positions",
                automated=False,
                requires_approval=True,
            ),
            RecoveryStep(
                order=7,
                action="resume_trading",
                description="Resume affected strategies after reconciliation is verified.",
                component="execution",
                automated=False,
                requires_approval=True,
            ),
        ]
        return {
            "scenario": "position_discrepancy",
            "severity": "high",
            "estimated_recovery_minutes": 45,
            "steps": [s.to_dict() for s in steps],
            "required_backups": [],
            "notes": (
                "Broker positions are the source of truth. All local "
                "adjustments must be logged for compliance audit."
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_backup(self, backup_id: str) -> Optional[BackupRecord]:
        """Look up a backup record by ID."""
        for record in self._backups:
            if record.backup_id == backup_id:
                return record
        return None

    @staticmethod
    def _default_failover_strategy(component: str) -> str:
        """Return the default failover strategy name for a component."""
        strategies = {
            "state_store": "restore_from_backup",
            "trade_recorder": "restore_from_backup",
            "config": "reload_from_file",
            "positions": "reconcile_with_broker",
            "broker_interface": "switch_to_secondary",
            "data_layer": "switch_to_backup_feed",
        }
        return strategies.get(component, "manual_intervention")

    @staticmethod
    def _component_priority(component: str) -> int:
        """Return recovery priority for a component (1 = highest)."""
        priorities = {
            "positions": 1,
            "state_store": 2,
            "trade_recorder": 3,
            "broker_interface": 4,
            "config": 5,
            "data_layer": 6,
        }
        return priorities.get(component, 10)
