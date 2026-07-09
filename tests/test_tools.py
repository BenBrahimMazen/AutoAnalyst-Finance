"""Tests for the financial data tools (Step 3).

The first two tests hit the network (yfinance) and require internet access.
Marked with ``@pytest.mark.network`` so they can be deselected in offline CI.
"""

from __future__ import annotations

import pytest

from src.tools.financial_data import FinancialDataFetcher


def test_get_sector_peers_excludes_company() -> None:
    """get_sector_peers should map sectors and exclude the analyzed ticker."""
    fetcher = FinancialDataFetcher()
    peers = fetcher.get_sector_peers("Technology", "AAPL")
    assert "MSFT" in peers
    assert "GOOGL" in peers
    assert "AAPL" not in peers


def test_get_sector_peers_unknown_sector_defaults() -> None:
    """Unknown sectors fall back to broad index ETFs."""
    fetcher = FinancialDataFetcher()
    peers = fetcher.get_sector_peers("Totally Made Up Sector", "XYZ")
    assert peers == ["SPY", "QQQ", "DIA", "IWM"]


def test_get_macro_data_returns_all_keys_without_key() -> None:
    """get_macro_data must return all four macro keys even without FRED_API_KEY."""
    fetcher = FinancialDataFetcher()
    macro = fetcher.get_macro_data()
    for key in ("fed_funds_rate", "us_10y_yield", "cpi_yoy", "gdp_growth"):
        assert key in macro
        assert isinstance(macro[key], float)


@pytest.mark.network
def test_fetch_aapl() -> None:
    """End-to-end fetch of AAPL price + statements (requires network)."""
    fetcher = FinancialDataFetcher()
    price = fetcher.get_price_history("AAPL")
    assert "current_price" in price
    assert price["current_price"] > 0
    statements = fetcher.get_financial_statements("AAPL")
    assert "info" in statements


@pytest.mark.network
def test_fetch_invalid_ticker() -> None:
    """An invalid ticker returns an empty price dict instead of raising."""
    fetcher = FinancialDataFetcher()
    price = fetcher.get_price_history("INVALIDXXX")
    assert price == {}  # should not raise
