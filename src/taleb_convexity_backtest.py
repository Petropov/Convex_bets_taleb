#!/usr/bin/env python
"""
Taleb-style convexity layers backtest on SPY.

Layers:
- L0: naive 3M 20% OTM put, fixed premium rule
- L1: multi-strike 3M ladder
- L2: strike + tenor ladder, fixed budget per quarter
- L3: L2 + realized-vol skew + intralife take-profit
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


# ---------------------------
# Global parameters
# ---------------------------

YEARS = 20              # backtest horizon
MAX_TENOR_M = 6         # max tenor (months) used in ladders
OTM_LIST = [0.10, 0.15, 0.20, 0.25]
TENOR_M_LIST = [1, 3, 6]
BUDGET_PER_Q = 100.0    # total budget per quarter for ladder strategies

R = 0.01                # risk-free rate
DIV_YIELD = 0.015       # dividend yield for SPY


# ---------------------------
# BS pricer and helpers
# ---------------------------

def bs_put_price(S, K, T, r, q, sigma):
    """Black-Scholes European put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    return K * np.exp(-r * T) * Nd2 - S * np.exp(-q * T) * Nd1


def get_S0_and_window(df: pd.DataFrame, start_date, months: int):
    """
    Given a start date and tenor in months, return:
    - S0: spot at (or just before) start_date
    - S_exp: last price in the window [start_date, expiry]
    - S_low: minimum price in that window
    """
    col = "SPY"
    S0_series = df.loc[:start_date, col]
    if S0_series.empty:
        return None, None, None
    S0 = float(S0_series.iloc[-1])

    expiry = start_date + pd.DateOffset(months=months)
    window = df.loc[start_date:expiry, col]
    if window.empty:
        return None, None, None

    S_exp = float(window.iloc[-1])
    S_low = float(window.min())
    return S0, S_exp, S_low


def premium_pct_of_spot(otm: float) -> float:
    """
    Crude mapping from OTM% to premium% of spot.
    10% OTM ~ 2.5%
    25% OTM ~ 0.4%
    Linear interpolation in between.
    """
    x1, y1 = 0.10, 0.025
    x2, y2 = 0.25, 0.004
    if otm <= x1:
        return y1
    if otm >= x2:
        return y2
    return y1 + (y2 - y1) * (otm - x1) / (x2 - x1)


def build_qstarts(df: pd.DataFrame, years: int, max_tenor_m: int = 6) -> pd.DatetimeIndex:
    """
    Quarterly starts, truncated so that each has at least `max_tenor_m` months of data after it.
    """
    qs_all = df.resample("QS").first().index  # quarter starts
    # keep only those where we have data up to start + max_tenor_m
    qs_all = qs_all[qs_all + pd.DateOffset(months=max_tenor_m) <= df.index.max()]
    # last `years` of quarter starts
    qs = qs_all[qs_all >= qs_all.max() - pd.DateOffset(years=years)]
    return qs


# ---------------------------
# Realized vol & vol skew
# ------------------------

def realized_iv_on(df: pd.DataFrame, date, window_days=21):
    """
    Approximate annualized realized vol using last `window_days` returns up to `date`.

    Robust to edge cases where `date` is before the first index, or where
    the rolling window has not yet accumulated data.
    """
    col = "SPY"
    rets = df[col].pct_change().fillna(0.0)

    if rets.empty:
        # fallback: arbitrary default vol if there are no returns at all
        return 0.20

    iv_series = rets.rolling(window_days).std().fillna(rets.std()) * np.sqrt(252)

    # restrict to data up to `date`
    iv_up_to = iv_series.loc[:date]

    if iv_up_to.empty:
        # if nothing before `date`, fallback to first available value
        return float(iv_series.iloc[0])

    return float(iv_up_to.iloc[-1])


def skewed_vol(iv_base, otm: float):
    """
    Simple strike-vol skew: deeper OTM → higher implied vol.
    """
    if otm <= 0.10:
        mult = 1.3
    elif otm >= 0.25:
        mult = 1.8
    else:
        mult = 1.3 + (1.8 - 1.3) * (otm - 0.10) / (0.25 - 0.10)
    return iv_base * mult

def simulate_taleb_world(index: pd.DatetimeIndex, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a synthetic 'Taleb world' price series on the given date index.

    - Regime 0 (calm): low vol, t(5) noise
    - Regime 1 (crisis): high vol, t(3) noise + occasional big negative jumps
    - Simple Markov switching so crises cluster.

    Returns a DataFrame with one column 'SPY' to be compatible with existing code.
    """
    rng = np.random.default_rng(seed)
    n = len(index)

    # Regime parameters
    vol_calm = 0.01    # 1% daily vol
    vol_crisis = 0.04  # 4% daily vol
    df_calm = 5
    df_crisis = 3

    # Markov regime: 0 = calm, 1 = crisis
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        if regime[t-1] == 0:
            # from calm: mostly stay calm, small chance crisis
            regime[t] = 1 if rng.random() < 0.03 else 0
        else:
            # from crisis: mostly stay crisis, some chance revert
            regime[t] = 1 if rng.random() < 0.90 else 0

    rets = np.zeros(n)
    for t in range(n):
        if regime[t] == 0:
            # calm: small t noise
            z = rng.standard_t(df=df_calm)
            rets[t] = vol_calm * z / np.sqrt(df_calm / (df_calm - 2))
        else:
            # crisis: bigger t noise
            z = rng.standard_t(df=df_crisis)
            base = vol_crisis * z / np.sqrt(df_crisis / (df_crisis - 2))
            # occasional big crash jump
            if rng.random() < 0.10:
                jump = -0.15  # -15%
            else:
                jump = 0.0
            rets[t] = base + jump

    prices = np.empty(n)
    prices[0] = start_price
    for t in range(1, n):
        prices[t] = prices[t-1] * (1.0 + rets[t])

    df_syn = pd.DataFrame({"SPY": prices}, index=index)
    return df_syn


# ---------------------------
# Layers: L0 and L1
# ---------------------------

def layer0_naive_put(df: pd.DataFrame, qstarts, otm=0.20, tenor_m=3, prem_pct=0.08):
    """
    L0: naive 3M 20% OTM put with fixed 8% premium.
    """
    spent = gross = wins = 0.0
    rows = []

    for d in qstarts:
        S0, S_exp, S_low = get_S0_and_window(df, d, tenor_m)
        if S0 is None:
            continue
        K = S0 * (1 - otm)
        prem = prem_pct * S0
        intrinsic_exp = max(K - S_exp, 0.0)
        payout = intrinsic_exp
        pnl = payout - prem

        spent += prem
        gross += payout
        wins += int(payout > 0)

        rows.append({
            "Start": d.date(),
            "S0": S0,
            "Strike": K,
            "Prem": prem,
            "S_exp": S_exp,
            "Payout": payout,
            "PnL": pnl,
        })

    df_trades = pd.DataFrame(rows)
    summary = {
        "Layer": "L0_naive",
        "Total_Spent": round(spent, 2),
        "Total_Payout": round(gross, 2),
        "Net_PnL": round(gross - spent, 2),
        "Payoff_Ratio": round(gross / spent, 3) if spent > 0 else np.nan,
        "WinRate": round(wins / len(df_trades), 3) if len(df_trades) > 0 else np.nan,
    }
    return summary, df_trades


def layer1_strike_ladder(df: pd.DataFrame, qstarts, tenor_m=3):
    """
    L1: multi-strike ladder of 3M puts.
    """
    out = []

    for otm in OTM_LIST:
        spent = gross = wins = 0.0
        for d in qstarts:
            S0, S_exp, S_low = get_S0_and_window(df, d, tenor_m)
            if S0 is None:
                continue
            prem = premium_pct_of_spot(otm) * S0
            K = S0 * (1 - otm)
            intrinsic = max(K - S_exp, 0.0)

            spent += prem
            gross += intrinsic
            wins += int(intrinsic > 0)

        out.append({
            "Layer": "L1_strikes",
            "OTM_%": int(otm * 100),
            "Total_Spent": spent,
            "Total_Payout": gross,
            "Net_PnL": gross - spent,
            "Payoff_Ratio": gross / spent if spent > 0 else np.nan,
            "WinRate": wins / len(qstarts) if len(qstarts) > 0 else np.nan,
        })

    return pd.DataFrame(out)

# ---------------------------
# Layers: L2 and L3
# ---------------------------

def layer2_strike_tenor_ladder(df: pd.DataFrame, qstarts):
    """
    L2: Strike + tenor ladder. Fixed BUDGET_PER_Q per quarter.
    """
    n_legs = len(OTM_LIST) * len(TENOR_M_LIST)
    per_leg = BUDGET_PER_Q / n_legs
    spent = gross = wins = 0.0

    for d in qstarts:
        for otm in OTM_LIST:
            for tm in TENOR_M_LIST:
                S0, S_exp, S_low = get_S0_and_window(df, d, tm)
                if S0 is None:
                    continue
                prem = premium_pct_of_spot(otm) * S0
                if prem <= 0:
                    continue
                qty = per_leg / prem
                intrinsic = max(S0 * (1 - otm) - S_exp, 0.0) * qty

                spent += prem * qty
                gross += intrinsic
                wins += int(intrinsic > 0)

    summary = {
        "Layer": "L2_strike+tenor",
        "Total_Spent": round(spent, 2),
        "Total_Payout": round(gross, 2),
        "Net_PnL": round(gross - spent, 2),
        "Payoff_Ratio": round(gross / spent, 3) if spent > 0 else np.nan,
        "WinRate": round(wins / (len(qstarts) * n_legs), 3) if len(qstarts) * n_legs > 0 else np.nan,
    }
    return summary


def layer3_vol_and_tp(df: pd.DataFrame, qstarts, tp_mult=3.0):
    """
    L3: L2-style ladder + realized-vol-based BS pricing + intralife take-profit.
    """
    n_legs = len(OTM_LIST) * len(TENOR_M_LIST)
    per_leg = BUDGET_PER_Q / n_legs
    spent = gross = wins = tp_hits = 0.0

    for d in qstarts:
        iv_base = realized_iv_on(df, d)

        for otm in OTM_LIST:
            for tm in TENOR_M_LIST:
                S0, S_exp, S_low = get_S0_and_window(df, d, tm)
                if S0 is None:
                    continue

                T = tm / 12.0
                K = S0 * (1 - otm)
                sigma = skewed_vol(iv_base, otm)
                prem = bs_put_price(S0, K, T, R, DIV_YIELD, sigma)
                if prem <= 0:
                    continue

                qty = per_leg / prem
                paid = prem * qty

                intrinsic_low = max(K - S_low, 0.0) * qty
                intrinsic_exp = max(K - S_exp, 0.0) * qty
                tp_level = tp_mult * prem * qty

                if intrinsic_low >= tp_level:
                    payout = tp_level
                    tp_hits += 1
                else:
                    payout = intrinsic_exp

                spent += paid
                gross += payout
                wins += int(payout > 0)

    summary = {
        "Layer": "L3_vol+TP",
        "Total_Spent": round(spent, 2),
        "Total_Payout": round(gross, 2),
        "Net_PnL": round(gross - spent, 2),
        "Payoff_Ratio": round(gross / spent, 3) if spent > 0 else np.nan,
        "WinRate": round(wins / (len(qstarts) * n_legs), 3) if len(qstarts) * n_legs > 0 else np.nan,
        "TP_Hit_Rate": round(tp_hits / (len(qstarts) * n_legs), 3) if len(qstarts) * n_legs > 0 else np.nan,
    }
    return summary


# ---------------------------
# Main runner
# ---------------------------

def main():
    # 1) Download SPY
    print("Downloading SPY prices via yfinance...")
    start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")

    raw_spy = yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
    if raw_spy.empty:
        raise RuntimeError("Download failed for SPY.")

    # Robust extraction of Close as a Series
    if "Close" in raw_spy.columns:
        spy_close = raw_spy["Close"]
    elif ("SPY", "Close") in raw_spy.columns:
        spy_close = raw_spy["SPY"]["Close"]
    else:
        raise RuntimeError(f"SPY Close column not found. Columns returned: {raw_spy.columns}")

    spy_close = spy_close.squeeze()
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_close.name = "SPY"
    df_prices = spy_close.to_frame()

    print(f"Got {len(df_prices)} daily prices from {df_prices.index.min().date()} to {df_prices.index.max().date()}.")

    # 2) Build quarter starts for SPY
    qstarts_spy = build_qstarts(df_prices, YEARS, max_tenor_m=MAX_TENOR_M)
    print(f"SPY quarter starts: {len(qstarts_spy)} from {qstarts_spy.min().date()} to {qstarts_spy.max().date()}")

    # 3) Run layers on SPY

    # L0 - SPY
    summary_L0_spy, df_L0_spy = layer0_naive_put(df_prices, qstarts_spy)
    df_L0_spy.to_csv("L0_trades_spy.csv", index=False)

    # L1 - SPY
    summary_L1_spy_df = layer1_strike_ladder(df_prices, qstarts_spy)
    summary_L1_spy = {
        "Layer": "L1_strikes_all",
        "Total_Spent": summary_L1_spy_df["Total_Spent"].sum(),
        "Total_Payout": summary_L1_spy_df["Total_Payout"].sum(),
    }
    summary_L1_spy["Net_PnL"] = summary_L1_spy["Total_Payout"] - summary_L1_spy["Total_Spent"]
    summary_L1_spy["Payoff_Ratio"] = (
        summary_L1_spy["Total_Payout"] / summary_L1_spy["Total_Spent"]
        if summary_L1_spy["Total_Spent"] > 0 else np.nan
    )
    summary_L1_spy["WinRate"] = summary_L1_spy_df["WinRate"].mean()

    # L2 - SPY
    summary_L2_spy = layer2_strike_tenor_ladder(df_prices, qstarts_spy)

    # L3 - SPY
    summary_L3_spy = layer3_vol_and_tp(df_prices, qstarts_spy, tp_mult=3.0)

    summary_layers_spy = pd.DataFrame([
        summary_L0_spy,
        summary_L1_spy,
        summary_L2_spy,
        summary_L3_spy,
    ])[["Layer", "Total_Spent", "Total_Payout", "Net_PnL", "Payoff_Ratio", "WinRate"]]

    summary_layers_spy.to_csv("layer_summary_spy.csv", index=False)

    # 4) Simulate synthetic Taleb world using same date index
    print("\nSimulating synthetic Taleb world...")
    start_price_syn = float(df_prices["SPY"].iloc[0])
    df_syn = simulate_taleb_world(df_prices.index, start_price=start_price_syn, seed=123)

    # Quarter starts for synthetic
    qstarts_syn = build_qstarts(df_syn, YEARS, max_tenor_m=MAX_TENOR_M)
    print(f"SYNTH quarter starts: {len(qstarts_syn)} from {qstarts_syn.min().date()} to {qstarts_syn.max().date()}")

    # 5) Run layers on synthetic world

    # L0 - SYN
    summary_L0_syn, df_L0_syn = layer0_naive_put(df_syn, qstarts_syn)
    df_L0_syn.to_csv("L0_trades_synth.csv", index=False)

    # L1 - SYN
    summary_L1_syn_df = layer1_strike_ladder(df_syn, qstarts_syn)
    summary_L1_syn = {
        "Layer": "L1_strikes_all",
        "Total_Spent": summary_L1_syn_df["Total_Spent"].sum(),
        "Total_Payout": summary_L1_syn_df["Total_Payout"].sum(),
    }
    summary_L1_syn["Net_PnL"] = summary_L1_syn["Total_Payout"] - summary_L1_syn["Total_Spent"]
    summary_L1_syn["Payoff_Ratio"] = (
        summary_L1_syn["Total_Payout"] / summary_L1_syn["Total_Spent"]
        if summary_L1_syn["Total_Spent"] > 0 else np.nan
    )
    summary_L1_syn["WinRate"] = summary_L1_syn_df["WinRate"].mean()

    # L2 - SYN
    summary_L2_syn = layer2_strike_tenor_ladder(df_syn, qstarts_syn)

    # L3 - SYN
    summary_L3_syn = layer3_vol_and_tp(df_syn, qstarts_syn, tp_mult=3.0)

    summary_layers_syn = pd.DataFrame([
        summary_L0_syn,
        summary_L1_syn,
        summary_L2_syn,
        summary_L3_syn,
    ])[["Layer", "Total_Spent", "Total_Payout", "Net_PnL", "Payoff_Ratio", "WinRate"]]

    summary_layers_syn.to_csv("layer_summary_synth.csv", index=False)

    # 6) Print summaries

    print("\n=== Layer Summary: SPY (Historical) ===")
    print(summary_layers_spy.to_string(index=False))

    print("\n=== Layer Summary: Synthetic Taleb World ===")
    print(summary_layers_syn.to_string(index=False))

    # 7) Write HTML recap
    now = pd.Timestamp.today()
    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Taleb-Style Convexity Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
        th {{ background-color: #f2f2f2; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .universe-title {{ margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>Taleb-Style Convexity Backtest</h1>
    <p>Generated: {now}</p>

    <h2 class="universe-title">SPY (Historical Market)</h2>
    {summary_layers_spy.to_html(index=False, float_format=lambda x: f"{x:0.3f}")}

    <h2 class="universe-title">Synthetic Taleb World (Fat-Tailed, Regime-Switching)</h2>
    {summary_layers_syn.to_html(index=False, float_format=lambda x: f"{x:0.3f}")}

    <p>Note: Synthetic world intentionally has clustered crises and fatter tails than SPY, to illustrate convexity behaviour in a true Taleb-style environment.</p>
</body>
</html>
    """

    with open("convexity_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("\nSaved:")
    print("  - L0_trades_spy.csv")
    print("  - L0_trades_synth.csv")
    print("  - layer_summary_spy.csv")
    print("  - layer_summary_synth.csv")
    print("  - convexity_report.html")

if __name__ == "__main__":
    main()
