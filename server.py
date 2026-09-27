"""
XAUUSD Buy/Sell Probability Server
===================================
Connects to MetaTrader 5, fetches M1 OHLCV data for XAUUSD,
computes two indicator signals:

  Indicator 1 – EMA Crossover (EMA 20 / 50 / 200)
  Indicator 2 – SMA Crossover (SMA 8 / 19 / 50)

Combines the two indicators into a blended Buy % / Sell % and
exposes the results through a lightweight Flask REST API.

Endpoints:
  GET /api/signal      → current probabilities + latest candle data
  GET /api/history     → last N candles with indicator values
  GET /api/status      → MT5 connection health
"""

from __future__ import annotations

import math
import time
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    from flask import Flask, jsonify
    from flask_cors import CORS
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n  pip install flask flask-cors"
    )

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("[WARN] MetaTrader5 package not found – running in DEMO mode with synthetic data.")

# ─── Configuration ────────────────────────────────────────────────────────────

SYMBOL      = "XAUUSDm"    # auto-detected if wrong; common variants: XAUUSD, XAUUSDm, GOLD
TIMEFRAME   = None          # resolved below after mt5 import
FETCH_BARS  = 600           # bars loaded each live cycle (≥ EMA 200 warmup + display window)
REFRESH_SEC = 1             # live signal refresh interval (seconds)

# EMA periods
EMA_FAST   = 20
EMA_MID    = 50
EMA_SLOW   = 200

# SMA periods
SMA_FAST   = 8
SMA_MID    = 19
SMA_SLOW   = 50

# Weight given to each indicator in the final blended probability
EMA_WEIGHT = 0.5
SMA_WEIGHT = 0.5

# ─── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # allow the HTML dashboard to call the API from any origin

# ─── Shared state (updated by background thread) ──────────────────────────────

_lock   = threading.Lock()
_state: dict[str, Any] = {
    "connected": False,
    "last_update": None,
    "signal": {},
    "history": [],
    "error": None,
}

# ─── Indicator helpers ────────────────────────────────────────────────────────

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average using the standard smoothing factor."""
    result = np.full_like(prices, np.nan)
    k = 2.0 / (period + 1)
    # seed with SMA for the first valid value
    result[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)
    return result


def _sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    result = np.full_like(prices, np.nan)
    for i in range(period - 1, len(prices)):
        result[i] = np.mean(prices[i - period + 1 : i + 1])
    return result


# ─── Probability calculation ──────────────────────────────────────────────────

def _ema_score(close: np.ndarray) -> tuple[float, float, dict]:
    """
    Returns (buy_pct, sell_pct) based on EMA 20/50/200 alignment.

    Scoring logic (additive, each criterion worth 1 point):
      +BUY  if close > EMA20
      +BUY  if EMA20 > EMA50  (fast above mid)
      +BUY  if EMA50 > EMA200 (mid above slow – trend confirmation)
      +BUY  if close > EMA200 (price above long-term trend)
    Inverse conditions score SELL points.
    Neutral / mixed scenarios are split proportionally.
    """
    e20  = _ema(close, EMA_FAST)
    e50  = _ema(close, EMA_MID)
    e200 = _ema(close, EMA_SLOW)

    last_close = close[-1]
    last_e20   = e20[-1]
    last_e50   = e50[-1]
    last_e200  = e200[-1]

    if any(math.isnan(v) for v in [last_e20, last_e50, last_e200]):
        return 50.0, 50.0, {}

    # Momentum scoring: slope of EMAs
    slope_e20  = (e20[-1]  - e20[-3])  / 2 if not np.isnan(e20[-3])  else 0
    slope_e50  = (e50[-1]  - e50[-3])  / 2 if not np.isnan(e50[-3])  else 0
    slope_e200 = (e200[-1] - e200[-3]) / 2 if not np.isnan(e200[-3]) else 0

    buy_pts  = 0.0
    sell_pts = 0.0
    MAX_PTS  = 7.0  # maximum possible points

    # ── Alignment criteria ──
    buy_pts  += 1 if last_close > last_e20  else 0
    sell_pts += 1 if last_close < last_e20  else 0

    buy_pts  += 1 if last_e20 > last_e50   else 0
    sell_pts += 1 if last_e20 < last_e50   else 0

    buy_pts  += 1 if last_e50 > last_e200  else 0
    sell_pts += 1 if last_e50 < last_e200  else 0

    buy_pts  += 1 if last_close > last_e200 else 0
    sell_pts += 1 if last_close < last_e200 else 0

    # ── Slope / momentum criteria ──
    buy_pts  += 1 if slope_e20  > 0 else 0
    sell_pts += 1 if slope_e20  < 0 else 0

    buy_pts  += 1 if slope_e50  > 0 else 0
    sell_pts += 1 if slope_e50  < 0 else 0

    buy_pts  += 1 if slope_e200 > 0 else 0
    sell_pts += 1 if slope_e200 < 0 else 0

    total = buy_pts + sell_pts
    if total == 0:
        buy_pct = sell_pct = 50.0
    else:
        buy_pct  = round((buy_pts  / MAX_PTS) * 100, 2)
        sell_pct = round((sell_pts / MAX_PTS) * 100, 2)

    values = {
        "ema20":  round(last_e20,  4),
        "ema50":  round(last_e50,  4),
        "ema200": round(last_e200, 4),
        "slope_ema20":  round(slope_e20,  6),
        "slope_ema50":  round(slope_e50,  6),
        "slope_ema200": round(slope_e200, 6),
    }
    return buy_pct, sell_pct, values


def _sma_score(close: np.ndarray) -> tuple[float, float, dict]:
    """
    Returns (buy_pct, sell_pct) based on SMA 8 / 19 / 50 stack alignment.

    ── BULL (likely upward trend) ──────────────────────────────────────────
      SMA8  on TOP  (highest value)    ← price action pulls fast MA up first
      SMA19 in the middle
      SMA50 at BOTTOM (lowest value)
      All three lines sloping UPWARD
      Trigger: SMA8 recently crossed above SMA19 / SMA50 → up-move started

    ── BEAR (likely downward trend) ────────────────────────────────────────
      SMA50 on TOP  (highest value)    ← slow MA is still elevated
      SMA19 in the middle
      SMA8  at BOTTOM (lowest value)   ← fast MA dragged down first
      All three lines sloping DOWNWARD
      Trigger: SMA50 recently crossed above SMA19 / SMA8 → down-move started

    Scoring out of MAX_PTS = 10:
      ① Alignment   (0-4 pts)
         +4 full stack confirmed (8>19>50 bull  |  50>19>8 bear)
         +1 fast vs mid   only    (8>19 bull  |  8<19 bear)
         +1 mid  vs slow  only    (19>50 bull | 19<50 bear)
      ② Momentum    (0-3 pts)  one per MA slope pointing the right way
      ③ Cross bonus (0-3 pts)  decays with age, only fires when stack agrees
    """
    s8  = _sma(close, SMA_FAST)   # SMA 8
    s19 = _sma(close, SMA_MID)    # SMA 19
    s50 = _sma(close, SMA_SLOW)   # SMA 50

    last_s8  = s8[-1]
    last_s19 = s19[-1]
    last_s50 = s50[-1]

    if any(math.isnan(v) for v in [last_s8, last_s19, last_s50]):
        return 50.0, 50.0, {}

    # ── Slopes (3-bar span for noise resistance) ──────────────────────────
    slope_s8  = (s8[-1]  - s8[-3])  / 2 if not np.isnan(s8[-3])  else 0.0
    slope_s19 = (s19[-1] - s19[-3]) / 2 if not np.isnan(s19[-3]) else 0.0
    slope_s50 = (s50[-1] - s50[-3]) / 2 if not np.isnan(s50[-3]) else 0.0

    MAX_PTS  = 10.0
    buy_pts  = 0.0
    sell_pts = 0.0

    # ── ① Alignment ───────────────────────────────────────────────────────
    full_bull = (last_s8 > last_s19) and (last_s19 > last_s50)  # 8 top, 50 bottom
    full_bear = (last_s50 > last_s19) and (last_s19 > last_s8)  # 50 top, 8 bottom

    if full_bull:
        buy_pts += 4
    elif full_bear:
        sell_pts += 4
    else:
        # Partial stack: award individual legs
        if last_s8  > last_s19:  buy_pts  += 1   # fast above mid  → bull leg
        elif last_s8 < last_s19: sell_pts += 1   # fast below mid  → bear leg

        if last_s19 > last_s50:  buy_pts  += 1   # mid above slow  → bull leg
        elif last_s19 < last_s50: sell_pts += 1  # mid below slow  → bear leg

    # ── ② Momentum / slope ────────────────────────────────────────────────
    buy_pts  += 1 if slope_s8  > 0 else 0
    sell_pts += 1 if slope_s8  < 0 else 0

    buy_pts  += 1 if slope_s19 > 0 else 0
    sell_pts += 1 if slope_s19 < 0 else 0

    buy_pts  += 1 if slope_s50 > 0 else 0
    sell_pts += 1 if slope_s50 < 0 else 0

    # ── ③ Crossover trigger bonus (up to 3 pts) ───────────────────────────
    # Bull: SMA8 crossed ABOVE SMA19 or SMA50 (fast MA touches others upward)
    # Bear: SMA50 crossed ABOVE SMA19 or SMA8  (slow MA touched others, starting down-move)
    LOOKBACK = 10
    bull_cross_age = None
    bear_cross_age = None
    n = len(close)

    for lag in range(1, min(LOOKBACK + 1, n - 1)):
        i0 = n - 1 - lag   # bar BEFORE the potential cross
        i1 = n - lag        # bar AT / AFTER the cross

        if np.isnan(s8[i0]) or np.isnan(s19[i0]) or np.isnan(s50[i0]):
            continue

        # Bull: SMA8 crosses up through SMA19 or SMA50
        if bull_cross_age is None:
            c1 = (s8[i0] <= s19[i0]) and (s8[i1] > s19[i1])
            c2 = (s8[i0] <= s50[i0]) and (s8[i1] > s50[i1])
            if c1 or c2:
                bull_cross_age = lag

        # Bear: SMA50 crosses up through SMA19 or SMA8
        if bear_cross_age is None:
            c3 = (s50[i0] <= s19[i0]) and (s50[i1] > s19[i1])
            c4 = (s50[i0] <= s8[i0])  and (s50[i1] > s8[i1])
            if c3 or c4:
                bear_cross_age = lag

    # Bonus only fires when current stack alignment agrees with cross direction
    if bull_cross_age is not None and full_bull:
        if   bull_cross_age <= 2:  buy_pts += 3
        elif bull_cross_age <= 5:  buy_pts += 2
        else:                      buy_pts += 1

    if bear_cross_age is not None and full_bear:
        if   bear_cross_age <= 2:  sell_pts += 3
        elif bear_cross_age <= 5:  sell_pts += 2
        else:                      sell_pts += 1

    # ── Convert to percentage (cap at 100%) ───────────────────────────────
    unused = max(0.0, MAX_PTS - buy_pts - sell_pts)
    buy_pct  = round(min(((buy_pts + unused / 2) / MAX_PTS) * 100, 100.0), 2)
    sell_pct = round(min(((sell_pts + unused / 2) / MAX_PTS) * 100, 100.0), 2)

    values = {
        "sma8":  round(last_s8,  4),
        "sma19": round(last_s19, 4),
        "sma50": round(last_s50, 4),
        "slope_sma8":  round(slope_s8,  6),
        "slope_sma19": round(slope_s19, 6),
        "slope_sma50": round(slope_s50, 6),
        "stack":               ("BULL (8>19>50)" if full_bull else
                                "BEAR (50>19>8)" if full_bear else "MIXED"),
        "bull_cross_bars_ago": bull_cross_age,
        "bear_cross_bars_ago": bear_cross_age,
    }
    return buy_pct, sell_pct, values


def _blend(ema_buy, ema_sell, sma_buy, sma_sell) -> tuple[float, float]:
    """Weighted average of both indicator signals, clamped to [0, 100]."""
    blended_buy  = round(EMA_WEIGHT * ema_buy  + SMA_WEIGHT * sma_buy,  2)
    blended_sell = round(EMA_WEIGHT * ema_sell + SMA_WEIGHT * sma_sell, 2)
    return blended_buy, blended_sell


def _bias_label(buy_pct: float, sell_pct: float) -> str:
    diff = buy_pct - sell_pct
    if   diff >  40: return "STRONG BUY"
    elif diff >  15: return "BUY"
    elif diff >   5: return "WEAK BUY"
    elif diff <  -40: return "STRONG SELL"
    elif diff <  -15: return "SELL"
    elif diff <   -5: return "WEAK SELL"
    else:             return "NEUTRAL"

# ─── MT5 data fetch ───────────────────────────────────────────────────────────

def _fetch_and_compute() -> None:
    """Pull latest bars from MT5 (or generate demo data), compute signals."""
    global _state

    use_demo = False
    if MT5_AVAILABLE:
        tf    = mt5.TIMEFRAME_M1
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, FETCH_BARS)
        if rates is None or len(rates) == 0:
            print(f"[WARN] MT5 returned no data for {SYMBOL} – using demo data (market may be closed).")
            use_demo = True
        else:
            close  = np.array([r["close"]        for r in rates], dtype=float)
            high   = np.array([r["high"]          for r in rates], dtype=float)
            low    = np.array([r["low"]           for r in rates], dtype=float)
            volume = np.array([r["tick_volume"]   for r in rates], dtype=float)
            times  = [datetime.fromtimestamp(r["time"], tz=timezone.utc).isoformat()
                      for r in rates]
    else:
        use_demo = True

    if use_demo:
        # ── Demo / synthetic data (MT5 not available or market closed) ────
        np.random.seed(int(time.time()) % 1000)
        n      = FETCH_BARS
        base   = 2320.0
        walk   = np.cumsum(np.random.randn(n) * 0.5)
        close  = base + walk
        high   = close + np.abs(np.random.randn(n) * 0.4)
        low    = close - np.abs(np.random.randn(n) * 0.4)
        volume = np.random.randint(50, 500, n).astype(float)
        now    = datetime.now(tz=timezone.utc)
        times  = [datetime.fromtimestamp(
                      now.timestamp() - (n - i) * 60, tz=timezone.utc
                  ).isoformat() for i in range(n)]

    # ── Compute indicators ────────────────────────────────────────────────
    ema_buy, ema_sell, ema_vals = _ema_score(close)
    sma_buy, sma_sell, sma_vals = _sma_score(close)
    blend_buy, blend_sell       = _blend(ema_buy, ema_sell, sma_buy, sma_sell)
    label                       = _bias_label(blend_buy, blend_sell)

    # ── Build history (last 100 candles with indicator values) ────────────
    e20_arr  = _ema(close, EMA_FAST)
    e50_arr  = _ema(close, EMA_MID)
    e200_arr = _ema(close, EMA_SLOW)
    s8_arr   = _sma(close, SMA_FAST)
    s19_arr  = _sma(close, SMA_MID)
    s50_arr  = _sma(close, SMA_SLOW)

    history = []
    for i in range(max(0, len(close) - 100), len(close)):
        def _f(v):
            return None if np.isnan(v) else round(float(v), 4)
        history.append({
            "time":   times[i],
            "open":   round(float(close[i]), 4),   # approx; MT5 open not stored here
            "high":   round(float(high[i]), 4),
            "low":    round(float(low[i]), 4),
            "close":  round(float(close[i]), 4),
            "volume": int(volume[i]),
            "ema20":  _f(e20_arr[i]),
            "ema50":  _f(e50_arr[i]),
            "ema200": _f(e200_arr[i]),
            "sma8":   _f(s8_arr[i]),
            "sma19":  _f(s19_arr[i]),
            "sma50":  _f(s50_arr[i]),
        })

    signal = {
        "symbol":       SYMBOL,
        "timeframe":    "M1",
        "price":        round(float(close[-1]), 4),
        "label":        label,
        # Blended
        "buy_pct":      blend_buy,
        "sell_pct":     blend_sell,
        # EMA indicator
        "ema_buy_pct":  ema_buy,
        "ema_sell_pct": ema_sell,
        "ema_values":   ema_vals,
        # SMA indicator
        "sma_buy_pct":  sma_buy,
        "sma_sell_pct": sma_sell,
        "sma_values":   sma_vals,
        "demo_mode":    use_demo or not MT5_AVAILABLE,
    }

    with _lock:
        _state["connected"]   = True
        _state["last_update"] = datetime.now(tz=timezone.utc).isoformat()
        _state["signal"]      = signal
        _state["history"]     = history
        _state["error"]       = None


# ─── Background refresh thread ────────────────────────────────────────────────

def _background_loop():
    # Connect to MT5 once in the thread
    if MT5_AVAILABLE:
        if not mt5.initialize():
            with _lock:
                _state["error"] = "MT5 initialize() failed – is MetaTrader 5 running?"
            print("[ERROR] MT5 initialize() failed.")
        else:
            print(f"[INFO] MT5 connected. Terminal: {mt5.terminal_info().name}")
            # Try to select the configured symbol; if it fails, auto-detect
            # broker variants (XAUUSDm, XAUUSD., GOLD, etc.)
            global SYMBOL
            if not mt5.symbol_select(SYMBOL, True):
                print(f"[WARN] Symbol '{SYMBOL}' not found – searching broker symbols…")
                all_symbols = mt5.symbols_get()
                candidates  = [s.name for s in (all_symbols or [])
                               if "XAUUSD" in s.name.upper() or "GOLD" in s.name.upper()]
                print(f"[INFO] Gold candidates: {candidates}")
                selected = False
                for cand in candidates:
                    if mt5.symbol_select(cand, True):
                        SYMBOL = cand
                        print(f"[INFO] Using symbol: {SYMBOL}")
                        selected = True
                        break
                if not selected:
                    print("[WARN] No gold symbol found. Will use demo data.")

    while True:
        try:
            _fetch_and_compute()
        except Exception as exc:
            with _lock:
                _state["error"] = str(exc)
            print(f"[ERROR] {exc}")
        time.sleep(REFRESH_SEC)


# ─── API Routes ───────────────────────────────────────────────────────────────

import os as _os
from flask import send_from_directory as _send_from_directory

_HERE = _os.path.dirname(_os.path.abspath(__file__))

@app.route("/")
def serve_dashboard():
    """Serve the index.html dashboard."""
    return _send_from_directory(_HERE, "index.html")


@app.route("/api/signal")
def api_signal():
    with _lock:
        return jsonify({
            "ok":          _state["connected"] or not MT5_AVAILABLE,
            "last_update": _state["last_update"],
            "error":       _state["error"],
            "data":        _state["signal"],
        })


@app.route("/api/history")
def api_history():
    with _lock:
        return jsonify({
            "ok":    True,
            "count": len(_state["history"]),
            "data":  _state["history"],
        })


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "mt5_available": MT5_AVAILABLE,
            "connected":     _state["connected"],
            "demo_mode":     not MT5_AVAILABLE,
            "last_update":   _state["last_update"],
            "error":         _state["error"],
            "symbol":        SYMBOL,
            "timeframe":     "M1",
            "ema_periods":   [EMA_FAST, EMA_MID, EMA_SLOW],
            "sma_periods":   [SMA_FAST, SMA_MID, SMA_SLOW],
            "refresh_sec":   REFRESH_SEC,
        })



# ─── Backtest Route ───────────────────────────────────────────────────────────

def _run_backtest() -> dict:
    """
    Fetch ~30 days of H1 data, compute EMA+SMA signals on a rolling window
    for every bar, and evaluate forward-return win rates.

    Returns a dict with:
      bars        – list of per-bar signal snapshots (for charting)
      stats       – win-rate summary at 3h / 6h / 24h look-forward
      meta        – timeframe, period, bar count
    """
    WARMUP  = 210   # need at least EMA-200 worth of bars before first signal
    DAYS    = 30
    H1_BARS = DAYS * 24 + WARMUP   # H1 bars ≈ 720 + 210 warmup

    # ── Fetch H1 data ─────────────────────────────────────────────────────
    use_demo_bt = False
    if MT5_AVAILABLE:
        rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, H1_BARS)
        if rates is None or len(rates) == 0:
            use_demo_bt = True
        else:
            close  = np.array([r["close"]  for r in rates], dtype=float)
            high   = np.array([r["high"]   for r in rates], dtype=float)
            low    = np.array([r["low"]    for r in rates], dtype=float)
            times  = [datetime.fromtimestamp(r["time"], tz=timezone.utc).isoformat()
                      for r in rates]
    else:
        use_demo_bt = True

    if use_demo_bt:
        n       = H1_BARS
        base    = 2300.0
        np.random.seed(42)
        close   = base + np.cumsum(np.random.randn(n) * 1.2)
        high    = close + np.abs(np.random.randn(n) * 0.8)
        low     = close - np.abs(np.random.randn(n) * 0.8)
        now     = datetime.now(tz=timezone.utc)
        times   = [datetime.fromtimestamp(
                       now.timestamp() - (n - i) * 3600, tz=timezone.utc
                   ).isoformat() for i in range(n)]

    # ── Pre-compute all MAs across full array ─────────────────────────────
    e20  = _ema(close, EMA_FAST)
    e50  = _ema(close, EMA_MID)
    e200 = _ema(close, EMA_SLOW)
    s8   = _sma(close, SMA_FAST)
    s19  = _sma(close, SMA_MID)
    s50  = _sma(close, SMA_SLOW)

    def _safe(v):
        return None if np.isnan(v) else round(float(v), 3)

    # ── Rolling signal for each bar (starting after warmup) ───────────────
    bars = []
    for i in range(WARMUP, len(close)):
        # ── EMA alignment score (simplified, no slope window edge guards) ──
        lc   = close[i]
        le20 = e20[i]; le50 = e50[i]; le200 = e200[i]
        ls8  = s8[i];  ls19 = s19[i]; ls50  = s50[i]

        if any(math.isnan(v) for v in [le20, le50, le200, ls8, ls19, ls50]):
            continue

        # EMA slope (3-bar)
        sl_e20  = (e20[i]  - e20[i-2])  / 2
        sl_e50  = (e50[i]  - e50[i-2])  / 2
        sl_e200 = (e200[i] - e200[i-2]) / 2

        eb = 0.0; es = 0.0
        eb += 1 if lc > le20  else 0; es += 1 if lc < le20  else 0
        eb += 1 if le20 > le50  else 0; es += 1 if le20 < le50  else 0
        eb += 1 if le50 > le200 else 0; es += 1 if le50 < le200 else 0
        eb += 1 if lc > le200  else 0; es += 1 if lc < le200  else 0
        eb += 1 if sl_e20  > 0 else 0; es += 1 if sl_e20  < 0 else 0
        eb += 1 if sl_e50  > 0 else 0; es += 1 if sl_e50  < 0 else 0
        eb += 1 if sl_e200 > 0 else 0; es += 1 if sl_e200 < 0 else 0
        ema_buy  = min(round(eb / 7 * 100, 1), 100)
        ema_sell = min(round(es / 7 * 100, 1), 100)

        # SMA alignment score
        sl_s8  = (s8[i]  - s8[i-2])  / 2
        sl_s19 = (s19[i] - s19[i-2]) / 2
        sl_s50 = (s50[i] - s50[i-2]) / 2

        full_bull = (ls8 > ls19) and (ls19 > ls50)
        full_bear = (ls50 > ls19) and (ls19 > ls8)

        sb = 0.0; ss = 0.0
        if full_bull:  sb += 4
        elif full_bear: ss += 4
        else:
            if ls8 > ls19:  sb += 1
            elif ls8 < ls19: ss += 1
            if ls19 > ls50: sb += 1
            elif ls19 < ls50: ss += 1

        sb += 1 if sl_s8  > 0 else 0; ss += 1 if sl_s8  < 0 else 0
        sb += 1 if sl_s19 > 0 else 0; ss += 1 if sl_s19 < 0 else 0
        sb += 1 if sl_s50 > 0 else 0; ss += 1 if sl_s50 < 0 else 0

        # Cross bonus (last 10 bars)
        for lag in range(1, min(11, i)):
            i0 = i - lag; i1 = i - lag + 1
            if np.isnan(s8[i0]) or np.isnan(s19[i0]): continue
            c1 = (s8[i0] <= s19[i0]) and (s8[i1] > s19[i1])
            c2 = (s8[i0] <= s50[i0]) and (s8[i1] > s50[i1])
            if (c1 or c2) and full_bull:
                sb += 3 if lag <= 2 else (2 if lag <= 5 else 1); break
        for lag in range(1, min(11, i)):
            i0 = i - lag; i1 = i - lag + 1
            if np.isnan(s50[i0]) or np.isnan(s19[i0]): continue
            c3 = (s50[i0] <= s19[i0]) and (s50[i1] > s19[i1])
            c4 = (s50[i0] <= s8[i0])  and (s50[i1] > s8[i1])
            if (c3 or c4) and full_bear:
                ss += 3 if lag <= 2 else (2 if lag <= 5 else 1); break

        sma_unused = max(0.0, 10.0 - sb - ss)
        sma_buy  = min(round((sb + sma_unused / 2) / 10.0 * 100, 1), 100.0)
        sma_sell = min(round((ss + sma_unused / 2) / 10.0 * 100, 1), 100.0)

        blend_buy, blend_sell = _blend(ema_buy, ema_sell, sma_buy, sma_sell)
        label = _bias_label(blend_buy, blend_sell)

        bars.append({
            "time":       times[i],
            "close":      round(float(lc), 3),
            "high":       round(float(high[i]), 3),
            "low":        round(float(low[i]), 3),
            "ema20":      _safe(le20),
            "ema50":      _safe(le50),
            "ema200":     _safe(le200),
            "sma8":       _safe(ls8),
            "sma19":      _safe(ls19),
            "sma50_val":  _safe(ls50),
            "buy_pct":    blend_buy,
            "sell_pct":   blend_sell,
            "label":      label,
            "stack":      ("BULL" if full_bull else "BEAR" if full_bear else "MIXED"),
        })

    # ── Win-rate statistics ────────────────────────────────────────────────
    THRESHOLDS = {"strong": 70, "normal": 55}
    HORIZONS   = [3, 6, 24]   # bars ahead (H1 → 3h, 6h, 24h)

    stats = {}
    for label_key, threshold in THRESHOLDS.items():
        for h in HORIZONS:
            buy_wins  = buy_total  = 0
            sell_wins = sell_total = 0
            for j, bar in enumerate(bars):
                future_idx = j + h
                if future_idx >= len(bars):
                    break
                future_close = bars[future_idx]["close"]
                current_close = bar["close"]
                if bar["buy_pct"] >= threshold:
                    buy_total += 1
                    if future_close > current_close:
                        buy_wins += 1
                if bar["sell_pct"] >= threshold:
                    sell_total += 1
                    if future_close < current_close:
                        sell_wins += 1
            key = f"{label_key}_{h}h"
            stats[key] = {
                "buy_winrate":   round(buy_wins  / buy_total  * 100, 1) if buy_total  else None,
                "sell_winrate":  round(sell_wins / sell_total * 100, 1) if sell_total else None,
                "buy_signals":   buy_total,
                "sell_signals":  sell_total,
            }

    # Aggregate counts
    signal_counts = {"STRONG BUY": 0, "BUY": 0, "WEAK BUY": 0,
                     "NEUTRAL": 0, "WEAK SELL": 0, "SELL": 0, "STRONG SELL": 0}
    for bar in bars:
        lbl = bar["label"]
        if lbl in signal_counts:
            signal_counts[lbl] += 1

    return {
        "bars":           bars[-720:],   # last 30 days only (trim warmup)
        "stats":          stats,
        "signal_counts":  signal_counts,
        "demo_mode":      use_demo_bt,
        "meta": {
            "symbol":     SYMBOL,
            "timeframe":  "H1",
            "days":       DAYS,
            "total_bars": len(bars),
        }
    }


@app.route("/api/backtest")
def api_backtest():
    """Run 30-day historical backtest on H1 data and return results."""
    try:
        result = _run_backtest()
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  XAUUSD Buy/Sell Probability Server")
    print("=" * 60)
    print(f"  Symbol     : {SYMBOL}")
    print(f"  Timeframe  : M1")
    print(f"  EMA periods: {EMA_FAST} / {EMA_MID} / {EMA_SLOW}")
    print(f"  SMA periods: {SMA_FAST} / {SMA_MID} / {SMA_SLOW}")
    print(f"  Refresh    : every {REFRESH_SEC}s")
    print(f"  MT5 package: {'[OK] available' if MT5_AVAILABLE else '[--] not found (DEMO mode)'}")
    print("=" * 60)
    print("  Dashboard  : http://localhost:5000")
    print("  API signal : http://localhost:5000/api/signal")
    print("=" * 60)

    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

    # Give the background thread a moment to fetch the first batch
    time.sleep(2)

    app.run(host="0.0.0.0", port=5000, debug=False)
