"""
core/evaluator.py — Trading Metrics & PnL Simulation v3
Dipakai oleh pipeline/07_evaluate.py dan pipeline/08_backtest.py

Fungsi utama:
  simulate_trades()       — simulasi trade dari fixed ATR multiple (legacy v2)
  simulate_trades_swing() — simulasi trade dari H4 Swing Points (BARU v3)
  calc_drawdown()         — max drawdown dari equity curve
  calc_consecutive_loss() — streak loss terpanjang
  calc_trade_per_month()  — rata-rata trade per bulan
  full_trading_report()   — metrik PnL lengkap
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.utils import setup_logger

logger = setup_logger("evaluator")


# ─── Simulasi Trade (ATR Fixed Multiple - Legacy v2) ─────────────────────────

def simulate_trades(
    y_pred:       np.ndarray,
    close:        np.ndarray,
    atr:          np.ndarray,
    modal:        float = 1000.0,
    leverage:     float = 3.0,
    fee_per_side: float = 0.0004,
    tp_mult:      float = 2.0,
    sl_mult:      float = 1.0,
    max_hold:     int   = 48,
    min_hold:     int   = 4,
) -> dict:
    y_pred = np.asarray(y_pred, dtype=np.int32)
    close  = np.asarray(close,  dtype=np.float64)
    atr    = np.asarray(atr,    dtype=np.float64)
    n      = len(y_pred)

    equity_curve  = np.zeros(n, dtype=np.float64)
    pnl_per_trade = []
    trade_log     = []
    cumulative    = 0.0
    total_fee     = 0.0
    wins = losses = time_exits = 0
    win_long = win_short = loss_long = loss_short = 0

    last_exit_bar = -1

    i = 0
    while i < n:
        pred = y_pred[i]
        if pred == 1 or (i - last_exit_bar) < min_hold:
            equity_curve[i] = cumulative
            i += 1
            continue

        entry_price = close[i]
        atr_i       = atr[i]

        if np.isnan(entry_price) or np.isnan(atr_i) or atr_i == 0 or entry_price == 0:
            equity_curve[i] = cumulative
            i += 1
            continue

        if pred == 2:  # LONG
            tp_price = entry_price + tp_mult * atr_i
            sl_price = entry_price - sl_mult * atr_i
        else:          # SHORT
            tp_price = entry_price - tp_mult * atr_i
            sl_price = entry_price + sl_mult * atr_i

        fee = 2 * fee_per_side * modal
        outcome   = "time_exit"
        exit_bar  = min(i + max_hold, n - 1)
        exit_price = close[exit_bar]

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if np.isnan(close[j]):
                continue

            est_high = close[j] + 0.5 * (atr[j] if not np.isnan(atr[j]) else atr_i)
            est_low  = close[j] - 0.5 * (atr[j] if not np.isnan(atr[j]) else atr_i)

            if pred == 2:  # LONG
                if est_high >= tp_price and est_low <= sl_price:
                    outcome  = "win" if close[j] >= entry_price else "loss"
                elif est_high >= tp_price:
                    outcome = "win"
                elif est_low <= sl_price:
                    outcome = "loss"
            else:  # SHORT
                if est_low <= tp_price and est_high >= sl_price:
                    outcome = "win" if close[j] <= entry_price else "loss"
                elif est_low <= tp_price:
                    outcome = "win"
                elif est_high >= sl_price:
                    outcome = "loss"

            if outcome in ("win", "loss"):
                exit_bar = j
                exit_price = close[j]
                break

        tp_pct = (tp_mult * atr_i) / entry_price
        sl_pct = (sl_mult * atr_i) / entry_price

        if outcome == "win":
            trade_pnl = tp_pct * leverage * modal - fee
            wins += 1
            if pred == 2: win_long  += 1
            else:         win_short += 1
        elif outcome == "loss":
            trade_pnl = -(sl_pct * leverage * modal) - fee
            losses += 1
            if pred == 2: loss_long  += 1
            else:         loss_short += 1
        else:  # time_exit
            if pred == 2:
                actual_ret = (exit_price - entry_price) / entry_price
            else:
                actual_ret = (entry_price - exit_price) / entry_price
            trade_pnl = actual_ret * leverage * modal - fee
            time_exits += 1
            if trade_pnl >= 0:
                wins += 1
                if pred == 2: win_long  += 1
                else:         win_short += 1
            else:
                losses += 1
                if pred == 2: loss_long  += 1
                else:         loss_short += 1

        cumulative += trade_pnl
        total_fee  += fee
        pnl_per_trade.append(trade_pnl)

        trade_log.append({
            "entry_bar":   int(i),
            "exit_bar":    int(exit_bar),
            "pred":        int(pred),
            "outcome":     outcome,
            "entry_price": round(float(entry_price), 6),
            "exit_price":  round(float(exit_price), 6),
            "pnl":         round(float(trade_pnl), 4),
        })

        for k in range(i, min(exit_bar + 1, n)):
            equity_curve[k] = cumulative

        last_exit_bar = exit_bar
        i = exit_bar + 1

    if n > 0:
        last_val = equity_curve[last_exit_bar] if last_exit_bar >= 0 else 0.0
        for k in range(last_exit_bar + 1, n):
            equity_curve[k] = last_val

    total_trades = wins + losses
    winrate = round(wins / total_trades, 4) if total_trades > 0 else 0.0

    wl  = win_long  + loss_long
    ws  = win_short + loss_short
    win_by_class = {
        "LONG":  round(win_long  / wl, 4) if wl > 0 else 0.0,
        "SHORT": round(win_short / ws, 4) if ws > 0 else 0.0,
    }

    return {
        "equity_curve":   equity_curve.tolist(),
        "pnl_per_trade":  pnl_per_trade,
        "trade_log":      trade_log,
        "total_pnl":      round(float(cumulative), 4),
        "total_trades":   total_trades,
        "wins":           wins,
        "losses":         losses,
        "time_exits":     time_exits,
        "total_fee_paid": round(float(total_fee), 4),
        "winrate":        winrate,
        "win_by_class":   win_by_class,
    }


# ─── ★ BARU v3: Simulasi Trade (Dinamis dari H4 Swing Points) ────────────────

def simulate_trades_swing(
    y_pred:          np.ndarray,
    close:           np.ndarray,
    high:            np.ndarray,
    low:             np.ndarray,
    atr:             np.ndarray,
    h4_swing_highs:  np.ndarray,   # swing high H4, aligned ke base tf
    h4_swing_lows:   np.ndarray,   # swing low  H4, aligned ke base tf
    modal:           float = 1000.0,
    leverage:        float = 3.0,
    fee_per_side:    float = 0.0004,
    min_rr:          float = 1.5,
    min_tp_atr:      float = 1.5,
    max_sl_atr:      float = 3.0,
    max_hold:        int   = 48,
) -> dict:
    """
    Simulasi trade dengan TP/SL dinamis berbasis swing high/low H4.

    Berbeda dari simulate_trades() yang pakai fixed ATR multiple:
    - TP = swing high H4 terdekat di atas (LONG) / swing low H4 di bawah (SHORT)
    - SL = swing low  H4 terdekat di bawah (LONG) / swing high H4 di atas (SHORT)
    - Skip trade jika R:R < min_rr (capital preservation)
    """
    n          = len(close)
    trades     = []
    equity     = modal
    equity_curve = [equity]

    LONG, SHORT, FLAT = 2, 0, 1   # sesuai LABEL_MAP

    for i in range(n - 1):
        sig = y_pred[i]
        if sig == FLAT:
            equity_curve.append(equity)
            continue

        price  = close[i]
        atr_i  = atr[i]
        sh_i   = h4_swing_highs[i]
        sl_i   = h4_swing_lows[i]

        if np.isnan(price) or np.isnan(atr_i) or atr_i == 0:
            equity_curve.append(equity)
            continue

        # ── Tentukan TP/SL dinamis ────────────────────────────────────────────
        if sig == LONG:
            if np.isnan(sh_i) or np.isnan(sl_i):
                equity_curve.append(equity)
                continue
            tp_price = sh_i
            sl_price = sl_i
            tp_dist  = tp_price - price
            sl_dist  = price    - sl_price
        else:  # SHORT
            if np.isnan(sh_i) or np.isnan(sl_i):
                equity_curve.append(equity)
                continue
            tp_price = sl_i
            sl_price = sh_i
            tp_dist  = price    - tp_price
            sl_dist  = sl_price - price

        # Validasi R:R
        if tp_dist <= 0 or sl_dist <= 0:
            equity_curve.append(equity)
            continue
        if tp_dist < min_tp_atr * atr_i:
            equity_curve.append(equity)
            continue
        if sl_dist > max_sl_atr * atr_i:
            equity_curve.append(equity)
            continue
        rr = tp_dist / sl_dist
        if rr < min_rr:
            equity_curve.append(equity)
            continue

        # ── Scan ke depan ─────────────────────────────────────────────────────
        outcome = "TIMEOUT"
        exit_price = price

        end = min(i + max_hold, n)
        for j in range(i + 1, end):
            if np.isnan(high[j]) or np.isnan(low[j]):
                continue
            if sig == LONG:
                if high[j] >= tp_price:
                    outcome    = "WIN";  exit_price = tp_price; break
                if low[j]  <= sl_price:
                    outcome    = "LOSS"; exit_price = sl_price; break
            else:
                if low[j]  <= tp_price:
                    outcome    = "WIN";  exit_price = tp_price; break
                if high[j] >= sl_price:
                    outcome    = "LOSS"; exit_price = sl_price; break

        # ── Hitung PnL ────────────────────────────────────────────────────────
        pct_move = (exit_price - price) / price
        if sig == SHORT:
            pct_move = -pct_move

        gross_pnl  = modal * leverage * pct_move
        fee_total  = modal * leverage * fee_per_side * 2
        net_pnl    = gross_pnl - fee_total

        equity    += net_pnl
        equity_curve.append(equity)

        trades.append({
            "bar_in":    i,
            "bar_out":   j if outcome != "TIMEOUT" else end,
            "direction": "LONG" if sig == LONG else "SHORT",
            "entry":     price,
            "exit":      exit_price,
            "tp":        tp_price,
            "sl":        sl_price,
            "rr":        round(rr, 2),
            "outcome":   outcome,
            "net_pnl":   round(net_pnl, 4),
            "equity":    round(equity, 4),
        })

    # ── Summary & Compatibility Mapping ───────────────────────────────────────
    if not trades:
        return {
            "error": "no_trades", "total_trades": 0, "winrate": 0.0,
            "total_pnl": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "equity_curve": equity_curve, "pnl_per_trade": [],
            "wins": 0, "losses": 0, "time_exits": 0,
            "win_by_class": {"LONG": 0.0, "SHORT": 0.0}
        }

    wins   = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    time_e = [t for t in trades if t["outcome"] == "TIMEOUT"]

    winrate    = len(wins) / len(trades) if trades else 0.0
    avg_win    = np.mean([t["net_pnl"] for t in wins])   if wins   else 0.0
    avg_loss   = np.mean([t["net_pnl"] for t in losses]) if losses else 0.0
    
    profit_factor = 0.0
    sum_loss = abs(sum(t["net_pnl"] for t in losses))
    if sum_loss > 0:
        profit_factor = abs(sum(t["net_pnl"] for t in wins)) / sum_loss

    equity_arr   = np.array([e for e in equity_curve if not np.isnan(e)])
    peak         = np.maximum.accumulate(equity_arr)
    drawdown     = (equity_arr - peak) / (peak + 1e-10)
    max_drawdown = float(drawdown.min())

    # Map class winrate
    lw = len([t for t in wins if t["direction"] == "LONG"])
    lt = len([t for t in trades if t["direction"] == "LONG"])
    sw = len([t for t in wins if t["direction"] == "SHORT"])
    st = len([t for t in trades if t["direction"] == "SHORT"])

    total_net_pnl = sum(t["net_pnl"] for t in trades)

    return {
        "total_trades":   len(trades),
        "winrate":        round(winrate, 4),
        "avg_win":        round(avg_win, 4),
        "avg_loss":       round(avg_loss, 4),
        "profit_factor":  round(profit_factor, 4),
        "net_pnl_total":  round(total_net_pnl, 4),
        "max_drawdown":   round(max_drawdown, 4),
        "avg_rr":         round(np.mean([t["rr"] for t in trades]), 4),
        "trades":         trades,
        
        # Compatibility keys untuk full_trading_report dan pipeline
        "equity_curve":   [e - modal for e in equity_curve], # convert equity ke PnL cumulative
        "pnl_per_trade":  [t["net_pnl"] for t in trades],
        "trade_log":      trades,
        "total_pnl":      round(total_net_pnl, 4),
        "wins":           len(wins),
        "losses":         len(losses),
        "time_exits":     len(time_e),
        "win_by_class": {
            "LONG":  round(lw / lt, 4) if lt > 0 else 0.0,
            "SHORT": round(sw / st, 4) if st > 0 else 0.0,
        }
    }


# ─── Drawdown ────────────────────────────────────────────────────────────────

def calc_drawdown(equity_curve: list, modal_per_trade: float = 1000.0) -> dict:
    if not equity_curve:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0, "drawdown_curve": []}

    eq   = np.array(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd   = peak - eq

    dd_pct = dd / (modal_per_trade + 1e-9)

    return {
        "max_drawdown":     round(float(dd.max()), 4),
        "max_drawdown_pct": round(float(dd_pct.max()), 4),
        "drawdown_curve":   dd.tolist(),
    }


# ─── Consecutive Loss ────────────────────────────────────────────────────────

def calc_consecutive_loss(pnl_per_trade: list) -> int:
    if not pnl_per_trade:
        return 0
    max_streak = current = 0
    for pnl in pnl_per_trade:
        if pnl < 0:
            current   += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


# ─── Trade Per Month ─────────────────────────────────────────────────────────

def calc_trade_per_month(total_trades: int, index: pd.DatetimeIndex) -> float:
    if total_trades == 0 or len(index) == 0:
        return 0.0
    n_months = (index[-1] - index[0]).days / 30.44
    if n_months < 0.1:
        return float(total_trades)
    return round(total_trades / n_months, 2)


# ─── Full Report ─────────────────────────────────────────────────────────────

def full_trading_report(
    y_pred:       np.ndarray,
    y_actual:     np.ndarray,
    atr:          np.ndarray,
    close:        np.ndarray,
    index:        pd.DatetimeIndex,
    modal:        float = 1000.0,
    leverages:    list  = [3.0, 5.0],
    fee_per_side: float = 0.0004,
    tp_mult:      float = 2.0,
    sl_mult:      float = 1.0,
    max_hold:     int   = 48,
    min_hold:     int   = 4,
    symbol:       Optional[str] = None,
    # Parameters for Swing V3 Option:
    high:         Optional[np.ndarray] = None,
    low:          Optional[np.ndarray] = None,
    h4_swing_highs: Optional[np.ndarray] = None,
    h4_swing_lows:  Optional[np.ndarray] = None,
    min_rr:       float = 1.5,
    min_tp_atr:   float = 1.5,
    max_sl_atr:   float = 3.0,
) -> dict:
    """
    Jalankan full trading simulation dan return metrics lengkap.
    Mendukung legacy fixed ATR (v2) dan dynamic H4 Swing (v3).
    """
    label_prefix = f"[{symbol}] " if symbol else ""
    use_swing = h4_swing_highs is not None and h4_swing_lows is not None and high is not None

    def run_sim(lev):
        if use_swing:
            return simulate_trades_swing(
                y_pred=y_pred, close=close, high=high, low=low, atr=atr,
                h4_swing_highs=h4_swing_highs, h4_swing_lows=h4_swing_lows,
                modal=modal, leverage=lev, fee_per_side=fee_per_side,
                min_rr=min_rr, min_tp_atr=min_tp_atr, max_sl_atr=max_sl_atr,
                max_hold=max_hold
            )
        else:
            return simulate_trades(
                y_pred=y_pred, close=close, atr=atr,
                modal=modal, leverage=lev, fee_per_side=fee_per_side,
                tp_mult=tp_mult, sl_mult=sl_mult,
                max_hold=max_hold, min_hold=min_hold,
            )

    # Base simulation (leverage pertama) untuk winrate dan consecutive loss
    base = run_sim(leverages[0])

    tpm        = calc_trade_per_month(base.get("total_trades", 0), index)
    max_consec = calc_consecutive_loss(base.get("pnl_per_trade", []))

    logger.info(
        f"{label_prefix}Winrate: {base.get('winrate', 0):.2%} "
        f"({base.get('wins', 0)}W / {base.get('losses', 0)}L / {base.get('total_trades', 0)} trades "
        f"| time_exit={base.get('time_exits', 0)})"
    )

    report = {
        "symbol":               symbol,
        "winrate":              base.get("winrate", 0),
        "total_trades":         base.get("total_trades", 0),
        "wins":                 base.get("wins", 0),
        "losses":               base.get("losses", 0),
        "time_exits":           base.get("time_exits", 0),
        "win_by_class":         base.get("win_by_class", {}),
        "trade_per_month":      tpm,
        "max_consecutive_loss": max_consec,
    }

    # PnL & Drawdown per leverage
    for lev in leverages:
        sim = run_sim(lev)
        dd  = calc_drawdown(sim.get("equity_curve", []), modal_per_trade=modal)
        key = f"lev{int(lev)}x"

        report[f"pnl_{key}"]          = sim.get("total_pnl", 0)
        report[f"max_drawdown_{key}"] = dd.get("max_drawdown_pct", 0)
        report[f"total_fee_{key}"]    = sim.get("total_fee_paid", 0) # Fallback 0 for swing

        logger.info(
            f"{label_prefix}Lev {lev}x → "
            f"PnL: ${sim.get('total_pnl', 0):+.2f} | "
            f"DD: {dd.get('max_drawdown_pct', 0):.2%}"
        )

    return report