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
import io
import base64
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# allow importing from the same src directory
sys.path.append(os.path.dirname(__file__))

from data_utils import load_multi_asset_history, load_spy_history
from taleb_convexity_backtest import (  # type: ignore
    simulate_taleb_world,
    layer0_naive_put,
    layer1_strike_ladder,
    layer2_strike_tenor_ladder,
    layer3_vol_and_tp,
    build_qstarts,
    YEARS,
    MAX_TENOR_M,
    BUDGET_PER_Q,
)


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def fmt_float(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.2f}"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:.1%}"


def fig_to_base64_png(fig) -> str:
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return img_b64


def make_payoff_ratio_chart(summary_layers_spy: pd.DataFrame,
                            summary_layers_syn: pd.DataFrame) -> str:
    labels = summary_layers_spy["Layer"].tolist()
    spy_vals = summary_layers_spy["Payoff_Ratio"].tolist()
    syn_vals = summary_layers_syn["Payoff_Ratio"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width / 2, spy_vals, width, label="SPY (Historical)")
    ax.bar(x + width / 2, syn_vals, width, label="Synthetic Taleb World")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Payoff Ratio")
    ax.set_title("Convexity Payoff Ratios: SPY vs Synthetic")
    ax.legend()

    return fig_to_base64_png(fig)


def make_mc_histogram(df_mc: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))

    for world, color in [("taleb", "tab:blue"), ("branching", "tab:orange")]:
        subset = df_mc[df_mc["world"] == world]["Net_PnL"]
        if subset.empty:
            continue
        ax.hist(subset, bins=20, alpha=0.6, label=world, color=color)

    ax.set_xlabel("L3 Net PnL")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo L3 Net PnL Distribution")
    ax.legend()

    return fig_to_base64_png(fig)


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
# Layer summaries (SPY vs Synthetic)
# --------------------------------------------------

def summarize_layers_for_universe(df: pd.DataFrame,
                                  qstarts: pd.DatetimeIndex) -> pd.DataFrame:
    summary_L0, _ = layer0_naive_put(df, qstarts)

    summary_L1_df = layer1_strike_ladder(df, qstarts)
    summary_L1 = {
        "Layer": "L1_strikes_all",
        "Total_Spent": summary_L1_df["Total_Spent"].sum(),
        "Total_Payout": summary_L1_df["Total_Payout"].sum(),
    }
    summary_L1["Net_PnL"] = summary_L1["Total_Payout"] - summary_L1["Total_Spent"]
    summary_L1["Payoff_Ratio"] = (
        summary_L1["Total_Payout"] / summary_L1["Total_Spent"]
        if summary_L1["Total_Spent"] > 0 else np.nan
    )
    summary_L1["WinRate"] = summary_L1_df["WinRate"].mean()

    summary_L2 = layer2_strike_tenor_ladder(df, qstarts)
    summary_L3 = layer3_vol_and_tp(df, qstarts, tp_mult=3.0)

    return pd.DataFrame([
        summary_L0,
        summary_L1,
        summary_L2,
        summary_L3,
    ])[["Layer", "Total_Spent", "Total_Payout", "Net_PnL", "Payoff_Ratio", "WinRate"]]


def build_layer_tables(df_spy: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qstarts_spy = build_qstarts(df_spy, YEARS, max_tenor_m=MAX_TENOR_M)
    if len(qstarts_spy) == 0:
        raise RuntimeError("No quarter starts for SPY data.")

    summary_layers_spy = summarize_layers_for_universe(df_spy, qstarts_spy)

    start_price_syn = float(df_spy["SPY"].iloc[0])
    df_syn = simulate_taleb_world(df_spy.index, start_price=start_price_syn, seed=123)
    qstarts_syn = build_qstarts(df_syn, YEARS, max_tenor_m=MAX_TENOR_M)
    if len(qstarts_syn) == 0:
        raise RuntimeError("No quarter starts for synthetic world.")
    summary_layers_syn = summarize_layers_for_universe(df_syn, qstarts_syn)

    return summary_layers_spy, summary_layers_syn


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
    df_assets = load_multi_asset_history(start_date)
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
# Unified report helpers
# --------------------------------------------------

def write_full_convexity_report(summary_layers_spy: pd.DataFrame,
                                summary_layers_syn: pd.DataFrame,
                                multi_summary: dict,
                                df_mc_summary: pd.DataFrame,
                                df_sweep: pd.DataFrame,
                                df_mc_paths: pd.DataFrame):
    now = pd.Timestamp.today()

    spy_fmt = summary_layers_spy.copy()
    for col in ["Total_Spent", "Total_Payout", "Net_PnL"]:
        spy_fmt[col] = spy_fmt[col].apply(fmt_money)
    spy_fmt["Payoff_Ratio"] = spy_fmt["Payoff_Ratio"].apply(fmt_float)
    spy_fmt["WinRate"] = spy_fmt["WinRate"].apply(fmt_float)

    syn_fmt = summary_layers_syn.copy()
    for col in ["Total_Spent", "Total_Payout", "Net_PnL"]:
        syn_fmt[col] = syn_fmt[col].apply(fmt_money)
    syn_fmt["Payoff_Ratio"] = syn_fmt["Payoff_Ratio"].apply(fmt_float)
    syn_fmt["WinRate"] = syn_fmt["WinRate"].apply(fmt_float)

    ms = multi_summary.copy()
    ms["initial_capital"] = fmt_money(ms["initial_capital"])
    ms["final_capital"] = fmt_money(ms["final_capital"])
    ms["total_return"] = fmt_pct(ms["total_return"])
    ms["TLT_return"] = fmt_pct(ms["TLT_return"])
    ms["GLD_return"] = fmt_pct(ms["GLD_return"])
    ms["L3_spent"] = fmt_money(ms["L3_spent"])
    ms["L3_payout"] = fmt_money(ms["L3_payout"])
    ms["L3_net"] = fmt_money(ms["L3_net"])
    ms["L3_payoff_ratio"] = fmt_float(ms["L3_payoff_ratio"])

    mc_fmt = df_mc_summary.copy()
    for col in ["mean", "median", "q05", "q25", "q75", "q95"]:
        mc_fmt[col] = mc_fmt[col].apply(fmt_money)

    df_top = df_sweep.sort_values("Payoff_Ratio", ascending=False).head(10).copy()
    for col in ["Total_Spent", "Total_Payout", "Net_PnL"]:
        df_top[col] = df_top[col].apply(fmt_money)
    df_top["Payoff_Ratio"] = df_top["Payoff_Ratio"].apply(fmt_float)
    df_top["WinRate"] = df_top["WinRate"].apply(fmt_float)
    df_top["TP_Hit_Rate"] = df_top["TP_Hit_Rate"].apply(fmt_float)

    payoff_chart_b64 = make_payoff_ratio_chart(summary_layers_spy, summary_layers_syn)
    mc_hist_b64 = make_mc_histogram(df_mc_paths)

    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Taleb-Style Convexity: Unified Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 30px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            margin-bottom: 30px;
            min-width: 650px;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 6px 10px;
        }}
        th {{
            background-color: #f4f4f4;
            text-align: center;
        }}
        td:first-child {{
            font-weight: bold;
            text-align: left;
        }}
        .section-note {{
            max-width: 900px;
            color: #555;
            margin-bottom: 20px;
        }}
        .chart-container {{
            margin: 15px 0 30px 0;
        }}
        .chart-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
    </style>
</head>

<body>

<h1>Taleb-Style Convexity Simulation Lab</h1>
<p><i>Generated: {now}</i></p>

<h2>Executive Summary</h2>
<p class="section-note">
This report evaluates Taleb-style convexity strategies (L0–L3) on historical SPY, a synthetic fat-tailed Taleb world,
a multi-asset barbell portfolio, Monte Carlo stress scenarios, and parameter sweeps. 
</p>
<ul class="section-note">
  <li>On historical SPY, all convex layers lose money: crises are too rare and implied volatility is expensive.</li>
  <li>In synthetic Taleb worlds with clustered crises and jumps, convex strategies (especially L2/L3) generate high payoff ratios and positive expected value.</li>
  <li>A TLT+GLD+L3 barbell performs very well historically due to GLD/TLT trends, with convex hedges acting as insurance against unobserved disasters.</li>
  <li>Monte Carlo confirms convexity is valuable in jump-intensive regimes and costly in mild, volatility-only regimes.</li>
  <li>Parameter sweeps show that deeper OTM, longer tenors, and higher take-profit multiples are more convex but have lower hit rates.</li>
</ul>

<h2>1. Convexity Layers on SPY vs Synthetic Taleb World</h2>
<p class="section-note">
We compare four strategies (L0–L3) on historical SPY and on a synthetic Taleb-style world. 
L0–L3 differ by moneyness, tenor laddering, and volatility-aware take-profit logic.
</p>

<div class="chart-container">
  <div class="chart-title">Figure 1: Convexity Payoff Ratios for L0–L3 (SPY vs Synthetic)</div>
  <img src="data:image/png;base64,{payoff_chart_b64}" alt="Payoff Ratios: SPY vs Synthetic"/>
</div>

<h3>1.1 SPY (Historical Market)</h3>
{spy_fmt.to_html(index=False, escape=False)}

<h3>1.2 Synthetic Taleb World (Fat-Tailed, Regime-Switching)</h3>
{syn_fmt.to_html(index=False, escape=False)}

<p class="section-note">
On historical SPY, all convex layers appear to lose money: realized crises are too infrequent to justify the premium paid.
In the synthetic Taleb world, deeper tails and clustered crises allow convex payoffs to recover far more than their premium,
especially for L2 and L3.
</p>

<h2>2. Multi-Asset Barbell: TLT + GLD + SPY L3 Hedge</h2>
<p class="section-note">
We construct a simple barbell: 40% TLT, 40% GLD, and 20% of capital allocated to the SPY L3 convex hedge.
This table summarizes the evolution of a $1,000,000 portfolio over the historical sample.
</p>

<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Initial Capital</td><td>{ms['initial_capital']}</td></tr>
  <tr><td>Final Capital</td><td>{ms['final_capital']}</td></tr>
  <tr><td>Total Return</td><td>{ms['total_return']}</td></tr>
  <tr><td>TLT Buy-and-Hold Return</td><td>{ms['TLT_return']}</td></tr>
  <tr><td>GLD Buy-and-Hold Return</td><td>{ms['GLD_return']}</td></tr>
  <tr><td>L3 Spent (scaled)</td><td>{ms['L3_spent']}</td></tr>
  <tr><td>L3 Payout (scaled)</td><td>{ms['L3_payout']}</td></tr>
  <tr><td>L3 Net PnL (scaled)</td><td>{ms['L3_net']}</td></tr>
  <tr><td>L3 Payoff Ratio (unscaled)</td><td>{ms['L3_payoff_ratio']}</td></tr>
</table>

<p class="section-note">
In this historical sample, the barbell performs well primarily due to GLD and TLT trends. The SPY convex hedge is a drag in
realized history, but it exists to protect against unobserved, unmodeled disasters rather than the specific crises we saw since 2003.
</p>

<h2>3. Monte Carlo: Many Universes, L3 Convexity Distribution</h2>
<p class="section-note">
We simulate many independent synthetic worlds of two types:
<b>taleb</b> (regime-switching with fat tails and jumps), and
<b>branching</b> (nested volatility with uncertainty about uncertainty).
Each path runs the L3 strategy; we summarize the Net PnL distribution below.
</p>

<div class="chart-container">
  <div class="chart-title">Figure 2: Monte Carlo L3 Net PnL Distribution</div>
  <img src="data:image/png;base64,{mc_hist_b64}" alt="MC L3 Net PnL Histogram"/>
</div>

{mc_fmt.to_html(index=False, escape=False)}

<p class="section-note">
The Taleb synthetic world typically shows a positive expectation for convex strategies, while the branching-volatility world
remains slightly costly on average unless tail severity or jump intensity is increased. This illustrates how convexity is not valued in
mild worlds but dominates in truly harsh, fat-tailed environments.
</p>

<h2>4. Parameter Sweep: OTM × Tenor × Take-Profit</h2>
<p class="section-note">
We explore a grid of OTM levels, option tenors, and take-profit multiples in a branching synthetic world and list the top
configurations ranked by payoff ratio. These are not guaranteed to be profitable in a single realization, but they highlight which
convex structures harvest tail payoffs most efficiently relative to premium.
</p>

{df_top.to_html(index=False, escape=False)}

<p class="section-note">
Parameter sweeps show that deeper OTM, longer tenors, and higher TP levels are more convex but come with lower hit rates.
</p>

<p style="margin-top:40px;color:#777;">
<small>
This unified report summarizes the Taleb-style convexity lab: how convex strategies behave in historical markets, synthetic
stress scenarios, multi-asset portfolios, Monte Carlo universes, and across a landscape of parameter choices.
</small>
</p>

<hr style="margin:50px 0;"/>

<h2>Limitations, Assumptions, and What This Analysis Is (and Is Not)</h2>

<p class="section-note">
This simulation is designed to test <b>structural behavior</b>, not to forecast returns or recommend
a tradable strategy. The results should be read as conditional statements:
<i>“If the world behaves like this, then convexity behaves like that.”</i>
</p>

<h3>Key Assumptions Made</h3>

<ul class="section-note">
  <li>
    <b>World structure matters more than parameter tuning.</b><br/>
    The analysis assumes that tail behavior (fat tails, jumps, crisis clustering)
    is the dominant driver of convexity outcomes, not fine-grained option calibration.
  </li>
  <li>
    <b>Synthetic worlds are stylized, not forecasts.</b><br/>
    The “Taleb World” is not a prediction of the future; it is a stress environment
    designed to include features missing from historical samples.
  </li>
  <li>
    <b>Option pricing is approximate.</b><br/>
    Implied volatility, skew, and take-profit logic are modeled heuristically.
    Real-world option markets include liquidity effects, bid–ask spreads,
    margin constraints, and execution frictions not modeled here.
  </li>
  <li>
    <b>Historical SPY is a censored sample.</b><br/>
    The historical period studied (post-2003) excludes many extreme systemic events
    and includes heavy policy intervention, which suppresses observed tail risk.
  </li>
</ul>

<h3>What This Analysis Does Not Claim</h3>

<ul class="section-note">
  <li>It does <b>not</b> claim convexity is profitable in all environments.</li>
  <li>It does <b>not</b> claim historical backtests are sufficient to evaluate tail strategies.</li>
  <li>It does <b>not</b> claim that any specific parameter set is “optimal.”</li>
  <li>It does <b>not</b> attempt to predict the timing of crashes.</li>
</ul>

<p class="section-note">
Convex strategies are evaluated here as <b>insurance-like structures</b>,
not as alpha-generating trades. Their role is survival under uncertainty,
not maximizing average returns.
</p>

<h2>Next Steps — Focused on the Biggest Unknowns</h2>

<p class="section-note">
Further exploration should prioritize uncertainty that materially affects real-world outcomes,
rather than refinements that improve elegance but not insight.
</p>

<h3>Priority 1 — Calibrate Tail Severity and Clustering</h3>
<p class="section-note">
The single most important unknown is <b>how severe and clustered future tail events will be</b>.
Before refining strategy mechanics, stress-testing should explore:
</p>
<ul class="section-note">
  <li>Higher-frequency crash regimes</li>
  <li>More extreme jump sizes</li>
  <li>Longer crisis persistence</li>
</ul>

<h3>Priority 2 — Portfolio-Level Survival Metrics</h3>
<p class="section-note">
Convexity should be evaluated by its impact on <b>portfolio ruin, drawdown, and recovery</b>,
not by standalone PnL. Key questions:
</p>
<ul class="section-note">
  <li>How does convexity affect maximum drawdown?</li>
  <li>How does it change recovery time after large losses?</li>
  <li>Does it prevent permanent capital impairment?</li>
</ul>

<h3>Priority 3 — Realistic Implementation Frictions</h3>
<p class="section-note">
Once tail structure is understood, realism should be increased where it affects outcomes:
</p>
<ul class="section-note">
  <li>Bid–ask spreads and liquidity under stress</li>
  <li>Margin and capital constraints during crises</li>
  <li>Execution delays and forced deleveraging</li>
</ul>

<h3>Lower Priority — Parameter Optimization</h3>
<p class="section-note">
Fine-tuning strikes, tenors, or take-profit levels should come <b>after</b>
tail assumptions are stress-tested. Optimization in the wrong world
creates false confidence.
</p>

<p style="margin-top:40px;color:#777;">
<small>
Bottom line: the most important uncertainty is not <i>how</i> convexity is implemented,
but <i>how hostile the future distribution truly is</i>.
</small>
</p>


</body>
</html>
    """

    with open("full_convexity_report.html", "w", encoding="utf-8") as f:
        f.write(html)


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
    df_spy = load_spy_history(start_date)

    # Align to YEARS like before
    df_spy = df_spy.loc[df_spy.index.max() - pd.DateOffset(years=YEARS):]

    summary_layers_spy, summary_layers_syn = build_layer_tables(df_spy)

    # Option 1: multi-asset portfolio
    multi_summary = run_multi_asset_portfolio()

    # Option 4: Monte Carlo
    df_mc = run_mc_on_worlds(df_spy, n_paths=50)
    df_mc_summary = summarize_mc(df_mc)

    # Option 6: parameter sweep
    df_sweep = run_parameter_sweep(df_spy)

    # Unified HTML report
    write_full_convexity_report(summary_layers_spy,
                                summary_layers_syn,
                                multi_summary,
                                df_mc_summary,
                                df_sweep,
                                df_mc)

    # Legacy extended HTML
    write_extended_html(multi_summary, df_mc_summary, df_sweep)

    print("\nExtended analysis complete. Files written:")
    print("  - mc_paths_L3.csv")
    print("  - sweep_results.csv")
    print("  - full_convexity_report.html")
    print("  - extended_report.html")


if __name__ == "__main__":
    main()

