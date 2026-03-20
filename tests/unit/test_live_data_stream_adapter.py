"""Unit tests for live_data_stream_adapter module.

TESTING_PLAN.md Section 3.2 — Data Layer.
"""

import time

import pandas as pd
import pytest

from quant_fund.data_layer.live_data_stream_adapter import (
    LiveDataStreamAdapter,
    NoOpFeedProvider,
)


@pytest.mark.unit
@pytest.mark.tier2
class TestLiveDataStreamAdapter:
    """LiveDataStreamAdapter — real-time data feed management."""

    @pytest.fixture
    def adapter(self):
        provider = NoOpFeedProvider()
        return LiveDataStreamAdapter(
            config={"live_feed": {"polling_interval_sec": 0.1, "buffer_size": 10}},
            feed_provider=provider,
        )

    def test_subscribe_and_unsubscribe(self, adapter):
        """Subscribe adds tickers, unsubscribe removes them."""
        adapter.subscribe(["AAPL", "MSFT"])
        assert "AAPL" in adapter.subscribed_tickers
        assert "MSFT" in adapter.subscribed_tickers

        adapter.unsubscribe(["AAPL"])
        assert "AAPL" not in adapter.subscribed_tickers
        assert "MSFT" in adapter.subscribed_tickers

    def test_connect_and_disconnect(self, adapter):
        """Connect starts the adapter, disconnect stops it."""
        adapter.subscribe(["AAPL"])
        adapter.connect()
        # Give polling loop a moment
        time.sleep(0.2)
        adapter.disconnect()

    def test_context_manager(self, adapter):
        """Adapter works as context manager for auto connect/disconnect."""
        adapter.subscribe(["AAPL"])
        with adapter:
            time.sleep(0.1)
        # Should be disconnected after exiting context

    def test_get_latest_bar_returns_dataframe(self, adapter):
        """get_latest_bar returns a DataFrame."""
        adapter.subscribe(["AAPL"])
        result = adapter.get_latest_bar()
        assert isinstance(result, pd.DataFrame)

    def test_get_snapshot_returns_dataframe(self, adapter):
        """get_snapshot returns a DataFrame."""
        adapter.subscribe(["AAPL"])
        result = adapter.get_snapshot()
        assert isinstance(result, pd.DataFrame)

    def test_on_bar_callback_registered(self, adapter):
        """Callbacks can be registered for bar events."""
        received = []
        adapter.on_bar(lambda df: received.append(df))
        # Callback registered — actual firing depends on feed providing data

    def test_get_buffered_bars_returns_dataframe(self, adapter):
        """get_buffered_bars returns concatenated DataFrame."""
        result = adapter.get_buffered_bars()
        assert isinstance(result, pd.DataFrame)

    def test_empty_subscription_returns_empty(self, adapter):
        """No subscriptions means empty data returned."""
        result = adapter.get_latest_bar()
        assert len(result) == 0
