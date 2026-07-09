"""Financial data fetching tools (yfinance + FRED).

Every public method is defensive: it catches exceptions, logs a warning, and
returns an empty dict / sensible fallback rather than raising. The pipeline
must never crash because a single data source was unavailable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Approximate macro fallbacks used when FRED_API_KEY is not configured.
# These are intentionally conservative round numbers, not live data.
_FALLBACK_MACRO: dict[str, float] = {
    "fed_funds_rate": 4.5,
    "us_10y_yield": 4.2,
    "cpi_yoy": 2.9,
    "gdp_growth": 2.1,
}

# Sector -> representative competitor tickers.
_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["MSFT", "GOOGL", "META", "NVDA"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Industrials": ["HON", "GE", "MMM", "CAT"],
}
_DEFAULT_PEERS: list[str] = ["SPY", "QQQ", "DIA", "IWM"]


class FinancialDataFetcher:
    """Fetch price history, financial statements, macro data, and sector peers."""

    def __init__(self) -> None:
        """Initialize the fetcher (no network calls here)."""
        self._fred: Any = None

    # ------------------------------------------------------------------
    # Price history
    # ------------------------------------------------------------------
    def get_price_history(self, ticker: str) -> dict:
        """Fetch one year of daily OHLCV data via yfinance.

        Args:
            ticker: Stock symbol, e.g. ``"AAPL"`` or ``"BNP.PA"``.

        Returns:
            Dict with ``dates``, ``close``, ``volume``, ``high_52w``,
            ``low_52w``, ``current_price`` and ``ytd_return``. Returns an empty
            dict if the ticker cannot be resolved.
        """
        try:
            stock: Any = yf.Ticker(ticker)
            hist: pd.DataFrame = stock.history(period="1y")
            if hist is None or hist.empty:
                logger.warning("No price history for ticker %r", ticker)
                return {}

            close: pd.Series = hist["Close"].dropna()
            volume: pd.Series = hist["Volume"]
            if close.empty:
                logger.warning("Empty close series for ticker %r", ticker)
                return {}

            current_price: float = float(close.iloc[-1])
            high_52w: float = float(hist["High"].max())
            low_52w: float = float(hist["Low"].min())

            # Year-to-date return using close on/after Jan 1 of this year.
            # The yfinance index is timezone-aware (e.g. America/New_York), so
            # year_start must carry the same tz to be comparable (pandas 3.x
            # raises on tz-naive vs tz-aware comparisons).
            last_date = close.index[-1]
            year_start = pd.Timestamp(
                year=last_date.year, month=1, day=1, tz=close.index.tz
            )
            ytd_close = close[close.index >= year_start]
            first_price = float(ytd_close.iloc[0]) if not ytd_close.empty else current_price
            ytd_return: float = ((current_price - first_price) / first_price) * 100.0

            return {
                "dates": [d.strftime("%Y-%m-%d") for d in close.index],
                "close": [float(p) for p in close.tolist()],
                "volume": [int(v) for v in volume.reindex(close.index).fillna(0).tolist()],
                "high_52w": high_52w,
                "low_52w": low_52w,
                "current_price": current_price,
                "ytd_return": round(ytd_return, 2),
            }
        except Exception as exc:  # noqa: BLE001 - never let a data call crash the pipeline
            logger.warning("Failed to fetch price history for %r: %s", ticker, exc)
            return {}

    # ------------------------------------------------------------------
    # Financial statements
    # ------------------------------------------------------------------
    def get_financial_statements(self, ticker: str) -> dict:
        """Fetch quarterly financial statements and the ``info`` dict.

        Args:
            ticker: Stock symbol.

        Returns:
            Dict with ``income_stmt``, ``balance_sheet``, ``cash_flow`` (each a
            nested dict via ``DataFrame.to_dict()``) and ``info``. Missing
            statements become empty dicts. Never raises.
        """
        result: dict[str, Any] = {
            "income_stmt": {},
            "balance_sheet": {},
            "cash_flow": {},
            "info": {},
        }
        try:
            stock = yf.Ticker(ticker)

            def _to_dict(obj: Any) -> dict:
                """Safely convert a yfinance DataFrame to a JSON-serializable dict.

                yfinance statement columns are pandas ``Timestamp`` objects; if we
                leave them as dict keys, LangGraph's checkpointer (ormsgpack)
                raises ``TypeError: Dict key must a type serializable with
                OPT_NON_STR_KEYS``. We stringify the date columns to ISO date
                strings first so the whole state remains serializable.
                """
                try:
                    if obj is None or (hasattr(obj, "empty") and obj.empty):
                        return {}
                    df = obj.copy()
                    try:
                        df.columns = [
                            c.isoformat() if hasattr(c, "isoformat") else str(c)
                            for c in df.columns
                        ]
                    except Exception:  # noqa: BLE001 - columns may be unhashable/odd
                        pass
                    return df.to_dict()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not serialize statement for %r: %s", ticker, exc)
                    return {}

            result["income_stmt"] = _to_dict(stock.quarterly_income_stmt)
            result["balance_sheet"] = _to_dict(stock.quarterly_balance_sheet)
            result["cash_flow"] = _to_dict(stock.quarterly_cashflow)

            info: Any = stock.info
            if isinstance(info, dict):
                result["info"] = info
            elif info is None:
                logger.warning("No info dict returned for %r", ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch statements for %r: %s", ticker, exc)
        return result

    # ------------------------------------------------------------------
    # Macro data (FRED)
    # ------------------------------------------------------------------
    def get_macro_data(self) -> dict:
        """Fetch current macro indicators from FRED.

        Requires ``FRED_API_KEY``. If it is missing or FRED is unreachable,
        returns hardcoded approximate values and logs a warning.

        Returns:
            Dict with ``fed_funds_rate``, ``us_10y_yield``, ``cpi_yoy`` and
            ``gdp_growth``.
        """
        try:
            fred = self._get_fred()
            if fred is None:
                logger.warning("FRED_API_KEY missing — using hardcoded macro fallback.")
                return dict(_FALLBACK_MACRO)

            fed_funds = float(fred.get_series("FEDFUNDS").dropna().iloc[-1])
            ten_year = float(fred.get_series("GS10").dropna().iloc[-1])

            # CPI YoY: 12-month percent change of the index level.
            cpi_series = fred.get_series("CPIAUCSL").dropna().tail(13)
            if len(cpi_series) >= 2:
                cpi_yoy = float((cpi_series.iloc[-1] / cpi_series.iloc[0] - 1.0) * 100.0)
            else:
                cpi_yoy = _FALLBACK_MACRO["cpi_yoy"]

            # GDP growth series is already a quarterly % change (annualized).
            gdp_series = fred.get_series("A191RL1Q225SBEA").dropna()
            gdp_growth = float(gdp_series.iloc[-1]) if not gdp_series.empty else _FALLBACK_MACRO["gdp_growth"]

            return {
                "fed_funds_rate": round(fed_funds, 2),
                "us_10y_yield": round(ten_year, 2),
                "cpi_yoy": round(cpi_yoy, 2),
                "gdp_growth": round(gdp_growth, 2),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED fetch failed (%s) — using hardcoded macro fallback.", exc)
            return dict(_FALLBACK_MACRO)

    def _get_fred(self) -> Any:
        """Lazily build a ``Fred`` client, or ``None`` if no API key is set."""
        if self._fred is not None:
            return self._fred
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            return None
        try:
            from fredapi import Fred

            self._fred = Fred(api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize FRED client: %s", exc)
            self._fred = None
        return self._fred

    # ------------------------------------------------------------------
    # Sector peers
    # ------------------------------------------------------------------
    def get_sector_peers(self, sector: str, exclude_ticker: str) -> list[str]:
        """Return up to 4 competitor tickers for ``sector``.

        Args:
            sector: yfinance ``info["sector"]`` value.
            exclude_ticker: Ticker to remove from the result (the company itself).

        Returns:
            List of peer tickers with ``exclude_ticker`` filtered out.
        """
        peers: list[str] = _SECTOR_PEERS.get(sector, _DEFAULT_PEERS)
        excluded_upper = (exclude_ticker or "").upper()
        return [p for p in peers if p.upper() != excluded_upper]
