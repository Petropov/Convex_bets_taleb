#!/usr/bin/env python
"""
Extended Taleb-style convexity analysis.

Implements:
- Option 1: Multi-asset portfolio (TLT + GLD linear, SPY L3 hedge)
- Option 2: Branching-volatility synthetic world (Taleb counterfactual style)
- Option 4: Monte Carlo across synthetic worlds
- Option 5: Extended HTML report
- Option 6: Parameter sweeps on synthetic world

Run:
    python src/run_extended_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# allow importing from the same src directory
sys.path.append(os.path.dirname(__file__))

from taleb_convexity_backtest import (  # type: ignore
    simulate_taleb_world,
    layer3_vol_and_tp,
    build_qstarts,
    YEARS,
    MAX_TENOR_M,
    BUDGET_PER_Q,
)


# --------------------------------------------------
# Option 2: Branching-vol synthetic world
# --------------------------------------------------

def simulate_branching_world(index: pd.DatetimeIndex,
                             start_price: float = 100.0,
                             levels: int = 3,
                             base_vol: float = 0.010,
                             a: float = 0.35,
                             seed: int = 123) -> pd.DataFrame:
    """
    Taleb-style 'future thicker than past' generator:
    - Hierarchical volatility: sigma is a product of random multipliers
    - Returns ~ N(0, sigma^2)
    - Produces much fatter tails than plain Gaussian

    levels: number of volatility layers in the cascade
    a: strength of volatility modulation
    """
    rng = np.random.default_rng(seed)
    n = len(index)

    sigmas = np.empty(n)
    rets = np.empty(n)
    for t in range(n):
        # volatility cascade: multiply (1 + a * eps) across levels
        vol_mult = 1.0
        for _ in range(levels):
            eps = rng.normal(0.0, 1.0)
            vol_mult *= max(0.1, 1.0 + a * eps)  # avoid negative vol multipliers
        sigma_t = base_vol * vol_mult
        sigmas[t] = sigma_t
        rets[t] = rng.normal(0.0, sigma_t)

    prices = np.empty(n)
    prices[0] = start_price
    for t in range(1, n):
        prices[t] = prices[t - 1] * (1.0 + rets[t])

    return pd.DataFrame({"SPY": prices}, index=index)


# --------------------------------------------------
# Helper: load multi-asset prices
# --------------------------------------------------

def load_multi_asset_prices(start_date: str) -> pd.DataFrame:
    """
    Download SPY, TLT, GLD adjusted close prices from yfinance and align dates.
    """
    tickers = ["SPY", "TLT", "GLD"]
    df = yf.download(" ".join(tickers), start=start_date, auto_adjust=True, progress=False)

    # yfinance can return multi-index columns; normalize to a simple Close matrix
    if "Close" in df.columns:
        df_close = df["Close"].copy()
    else:
        # try multiindex: (ticker, 'Close')
        df_close = df.xs("Close", level=1, axis=1)

    if isinstance(df_close, pd.Series):
        df_close = df_close.to_frame()

    df_close = df_close.dropna(how="any")
    return df_close


# --------------------------------------------------
# Option 1: Multi-asset portfolio with SPY hedge
# --------------------------------------------------

def run_multi_asset_portfolio() -> dict:
    """
    Build a simple multi-asset portfolio:
    - 40% TLT, 40% GLD (buy-and-hold)
    - 20% of capital used to fund SPY L3 convex tail hedge

    Returns a dict summarizing portfolio metrics.
    """
    print("\n[Option 1] Running multi-asset portfolio with SPY hedge...")

    start_date = (pd.Timestamp.today().normalize()
                  - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")
    df_assets = load_multi_asset_prices(start_date)
    df_spy = df_assets[["SPY"]].copy()

    # Align horizon like backtest
    df_spy = df_spy.loc[df_spy.index.max() - pd.DateOffset(years=YEARS):]
    df_assets = df_assets.loc[df_spy.index.min(): df_spy.index.max()]

    qstarts = build_qstarts(df_spy, YEARS, max_tenor_m=MAX_TENOR_M)
    if len(qstarts) == 0:
        raise RuntimeError("No quarter starts for multi-asset portfolio.")

    # Run L3 on SPY using base budget BUDGET_PER_Q
    from taleb_convexity_backtest import layer3_vol_and_tp  # local import to avoid cycles

    summary_L3 = layer3_vol_and_tp(df_spy, qstarts, tp_mult=3.0)

    # Portfolio capital assumptions
    initial_capital = 1_000_000.0
    hedge_capital = 0.20 * initial_capital        # 20% for convex hedge
    linear_capital = 0.80 * initial_capital       # 80% in TLT+GLD

    total_spent_base = summary_L3["Total_Spent"]
    if total_spent_base <= 0:
        scale = 0.0
    else:
        scale = hedge_capital / total_spent_base

    L3_spent = summary_L3["Total_Spent"] * scale
    L3_payout = summary_L3["Total_Payout"] * scale
    L3_net = summary_L3["Net_PnL"] * scale

    # Linear legs: 40% TLT, 40% GLD
    alloc_tlt = alloc_gld = 0.5 * linear_capital
    tlt_start, tlt_end = df_assets["TLT"].iloc[0], df_assets["TLT"].iloc[-1]
    gld_start, gld_end = df_assets["GLD"].iloc[0], df_assets["GLD"].iloc[-1]

    tlt_final = alloc_tlt * (tlt_end / tlt_start)
    gld_final = alloc_gld * (gld_end / gld_start)

    # Hedge capital: assume premiums are actually spent from the 20% bucket
    hedge_cash_start = hedge_capital
    hedge_cash_final = hedge_cash_start - L3_spent + L3_payout

    final_capital = tlt_final + gld_final + hedge_cash_final
    total_return = (final_capital / initial_capital) - 1.0

    summary = {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return": total_return,
        "L3_spent": L3_spent,
        "L3_payout": L3_payout,
        "L3_net": L3_net,
        "L3_payoff_ratio": summary_L3["Payoff_Ratio"],
        "TLT_return": (tlt_end / tlt_start) - 1.0,
        "GLD_return": (gld_end / gld_start) - 1.0,
    }
    return summary


# --------------------------------------------------
# Option 4: Monte Carlo across synthetic worlds
# --------------------------------------------------

def run_mc_on_worlds(df_prices: pd.DataFrame,
                     n_paths: int = 50,
                     seed_base: int = 1000) -> pd.DataFrame:
    """
    Run L3 on many synthetic paths:
    - world_type='taleb': uses simulate_taleb_world
    - world_type='branching': uses simulate_branching_world

    Returns a DataFrame with Net_PnL distribution.
    """
    print(f"\n[Option 4] Running Monte Carlo ({n_paths} paths per world)...")

    index = df_prices.index
    start_price = float(df_prices["SPY"].iloc[0])

    records = []

    for world_type in ["taleb", "branching"]:
        for i in range(n_paths):
            seed = seed_base + i
            if world_type == "taleb":
                df_world = simulate_taleb_world(index, start_price=start_price, seed=seed)
            else:
                df_world = simulate_branching_world(index, start_price=start_price, seed=seed)

            qstarts = build_qstarts(df_world, YEARS, max_tenor_m=MAX_TENOR_M)
            if len(qstarts) == 0:
                continue

            summary = layer3_vol_and_tp(df_world, qstarts, tp_mult=3.0)
            records.append({
                "world": world_type,
                "path_id": i,
                "Total_Spent": summary["Total_Spent"],
                "Total_Payout": summary["Total_Payout"],
                "Net_PnL": summary["Net_PnL"],
                "Payoff_Ratio": summary["Payoff_Ratio"],
            })

    df_mc = pd.DataFrame(records)
    df_mc.to_csv("mc_paths_L3.csv", index=False)
    return df_mc


def summarize_mc(df_mc: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize Monte Carlo Net_PnL distribution by world.
    """
    def q(x, p):
        return x.quantile(p)

    out = df_mc.groupby("world")["Net_PnL"].agg(
        mean="mean",
        median="median",
        q05=lambda x: q(x, 0.05),
        q25=lambda x: q(x, 0.25),
        q75=lambda x: q(x, 0.75),
        q95=lambda x: q(x, 0.95),
    ).reset_index()
    return out


# --------------------------------------------------
# Option 6: Parameter sweeps on synthetic world
# --------------------------------------------------

def run_single_put_L3_like(df: pd.DataFrame,
                           qstarts: pd.DatetimeIndex,
                           otm: float,
                           tenor_m: int,
                           tp_mult: float) -> dict:
    """
    Simplified L3-like strategy:
    - single OTM and tenor
    - budget BUDGET_PER_Q per quarter
    """
    from taleb_convexity_backtest import realized_iv_on, skewed_vol, bs_put_price  # type: ignore

    spent = gross = wins = tp_hits = 0.0

    for d in qstarts:
        iv_base = realized_iv_on(df, d)
        S0, S_exp, S_low = build_qstarts.__globals__["get_S0_and_window"](df, d, tenor_m)  # hacky but reuses function
        if S0 is None:
            continue

        T = tenor_m / 12.0
        K = S0 * (1 - otm)
        sigma = skewed_vol(iv_base, otm)
        prem = bs_put_price(S0, K, T, 0.01, 0.015, sigma)
        if prem <= 0:
            continue

        qty = BUDGET_PER_Q / prem
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
        "OTM": otm,
        "TenorM": tenor_m,
        "TP_mult": tp_mult,
        "Total_Spent": round(spent, 2),
        "Total_Payout": round(gross, 2),
        "Net_PnL": round(gross - spent, 2),
        "Payoff_Ratio": round(gross / spent, 3) if spent > 0 else np.nan,
        "WinRate": round(wins / len(qstarts), 3) if len(qstarts) > 0 else np.nan,
        "TP_Hit_Rate": round(tp_hits / len(qstarts), 3) if len(qstarts) > 0 else np.nan,
    }
    return summary


def run_parameter_sweep(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Parameter sweep over OTM, tenor, and TP on a single branching synthetic world.
    """
    print("\n[Option 6] Parameter sweep on branching synthetic world...")

    # build one branching world on same index
    start_price = float(df_prices["SPY"].iloc[0])
    df_world = simulate_branching_world(df_prices.index, start_price=start_price, seed=777)
    qstarts = build_qstarts(df_world, YEARS, max_tenor_m=MAX_TENOR_M)

    otms = [0.10, 0.15, 0.20, 0.25]
    tenors = [1, 3, 6]
    tps = [2.0, 3.0, 5.0]

    results = []
    for otm in otms:
        for tm in tenors:
            for tp in tps:
                s = run_single_put_L3_like(df_world, qstarts, otm, tm, tp)
                results.append(s)

    df_sweep = pd.DataFrame(results)
    df_sweep.to_csv("sweep_results.csv", index=False)
    return df_sweep


# --------------------------------------------------
# Extended HTML report (Option 5)
# --------------------------------------------------

def write_extended_html(multi_summary: dict,
                        df_mc_summary: pd.DataFrame,
                        df_sweep: pd.DataFrame):
    """
    Write extended_report.html combining:
    - Multi-asset portfolio summary
    - Monte Carlo summary
    - Top parameter-sweep strategies
    """
    now = pd.Timestamp.today()

    # Top 10 strategies by Payoff_Ratio
    df_top = df_sweep.sort_values("Payoff_Ratio", ascending=False).head(10)

    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Extended Taleb Convexity Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
        th {{ background-color: #f2f2f2; }}
        td:first-child, th:first-child {{ text-align: left; }}
    </style>
</head>
<body>
    <h1>Extended Taleb Convexity Analysis</h1>
    <p>Generated: {now}</p>

    <h2>Option 1: Multi-Asset Portfolio (TLT + GLD Linear, SPY L3 Hedge)</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Initial Capital</td><td>{multi_summary['initial_capital']:0.2f}</td></tr>
        <tr><td>Final Capital</td><td>{multi_summary['final_capital']:0.2f}</td></tr>
        <tr><td>Total Return</td><td>{multi_summary['total_return']:0.3%}</td></tr>
        <tr><td>TLT Buy-and-Hold Return</td><td>{multi_summary['TLT_return']:0.3%}</td></tr>
        <tr><td>GLD Buy-and-Hold Return</td><td>{multi_summary['GLD_return']:0.3%}</td></tr>
        <tr><td>L3 Spent (scaled)</td><td>{multi_summary['L3_spent']:0.2f}</td></tr>
        <tr><td>L3 Payout (scaled)</td><td>{multi_summary['L3_payout']:0.2f}</td></tr>
        <tr><td>L3 Net PnL (scaled)</td><td>{multi_summary['L3_net']:0.2f}</td></tr>
        <tr><td>L3 Payoff Ratio (unscaled)</td><td>{multi_summary['L3_payoff_ratio']:0.3f}</td></tr>
    </table>

    <h2>Option 4: Monte Carlo L3 Distribution (Taleb vs Branching)</h2>
    {df_mc_summary.to_html(index=False, float_format=lambda x: f"{x:0.3f}")}

    <h2>Option 6: Top Parameter Sweep Results (Branching World)</h2>
    {df_top.to_html(index=False, float_format=lambda x: f"{x:0.3f}")}

    <p>Note: Parameter sweep uses a single branching synthetic world; Monte Carlo uses many independent paths.</p>
</body>
</html>
    """

    with open("extended_report.html", "w", encoding="utf-8") as f:
        f.write(html)


# --------------------------------------------------
# Main entry
# --------------------------------------------------

def main():
    # Load historical SPY for reference index & also used in sweeps / MC
    start_date = (pd.Timestamp.today().normalize()
                  - pd.DateOffset(years=YEARS + 2)).strftime("%Y-%m-%d")
    raw_spy = yf.download("SPY", start=start_date, auto_adjust=True, progress=False)
    if raw_spy.empty:
        raise RuntimeError("Download failed for SPY.")

    if "Close" in raw_spy.columns:
        spy_close = raw_spy["Close"]
    elif ("SPY", "Close") in raw_spy.columns:
        spy_close = raw_spy["SPY"]["Close"]
    else:
        raise RuntimeError(f"SPY Close column not found. Columns returned: {raw_spy.columns}")

    spy_close = spy_close.squeeze()
    spy_close.index = pd.to_datetime(spy_close.index)
    spy_close.name = "SPY"
    df_spy = spy_close.to_frame()

    # Align to YEARS like before
    df_spy = df_spy.loc[df_spy.index.max() - pd.DateOffset(years=YEARS):]

    # Option 1: multi-asset portfolio
    multi_summary = run_multi_asset_portfolio()

    # Option 4: Monte Carlo
    df_mc = run_mc_on_worlds(df_spy, n_paths=50)
    df_mc_summary = summarize_mc(df_mc)

    # Option 6: parameter sweep
    df_sweep = run_parameter_sweep(df_spy)

    # Option 5: extended HTML
    write_extended_html(multi_summary, df_mc_summary, df_sweep)

    print("\nExtended analysis complete. Files written:")
    print("  - mc_paths_L3.csv")
    print("  - sweep_results.csv")
    print("  - extended_report.html")


if __name__ == "__main__":
    main()

