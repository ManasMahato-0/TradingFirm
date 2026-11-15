"""Turn a strategy's decision into plain-English 'thinking' for the Trading Floor."""
from __future__ import annotations

LABELS = {
    "ret_5": "last week's return", "ret_21": "last month's return", "ret_63": "3-month return",
    "ret_126": "6-month return", "mom_12_1": "12-month momentum",
    "dist_sma50": "price vs 50-day average", "dist_sma200": "price vs 200-day average",
    "vol_21": "1-month volatility", "vol_63": "3-month volatility", "volume_trend": "volume trend",
    "dist_52w_high": "closeness to 52-week high", "sector_rel_63": "strength vs its sector",
    "days_to_results": "days until results", "days_since_results": "days since results",
    "results_reaction": "reaction to recent results",
    "mkt_ret_21": "Nifty's last month", "mkt_ret_63": "Nifty's last 3 months",
    "mkt_dist_sma200": "Nifty vs its 200-day average", "mkt_vol_21": "Nifty volatility",
    "mkt_dist_52w_high": "Nifty vs its 52-week high", "breadth_200": "share of stocks above 200-day average",
    "vix_z": "India VIX level", "ew_dist_sma200": "broad market vs its 200-day average",
    "ew_drawdown": "broad market drawdown",
}


def label(feature: str) -> str:
    return LABELS.get(feature, feature)


def _names(xs: list[str], limit: int = 12) -> str:
    return ", ".join(xs[:limit]) + (f" +{len(xs) - limit} more" if len(xs) > limit else "") if xs else "none"


def explain_decision(decision: dict | None, weights: dict[str, float], held_before: list[str]) -> list[str]:
    """Thoughts behind one rebalance. `decision` is MomentumPlus.last_decision (or None
    for simple strategies, which are described from their target weights)."""
    chosen = sorted(weights, key=lambda s: -weights[s])
    added = [s for s in chosen if s not in held_before]
    dropped = [s for s in held_before if s not in weights]
    if not decision:
        out = [f"Month review: holding the top {len(chosen)} stocks by 12-month momentum: {_names(chosen)}."]
        if held_before:
            out.append(f"Adding {_names(added)}. Dropping {_names(dropped)}.")
        return out

    d = decision
    mkt = "above" if d["broad_market_up"] else "BELOW"
    out = [
        f"Month-end review. {len(d['shortlist'])} stocks on the momentum shortlist. Broad market is "
        f"{mkt} its 200-day average; our own drawdown is {d['own_drawdown']:.1%}. "
        f"Investing {d['exposure']:.0%} of capital across {len(d['chosen'])} stocks.",
    ]
    rank = {s: i + 1 for i, s in enumerate(d["ranked"])}
    if d["kept"]:
        out.append(f"Keeping {len(d['kept'])}: {_names(d['kept'])} (still ranked in the top 15, "
                   "so no need to pay costs to swap them).")
    if d["dropped"]:
        why = [f"{s} (rank {rank[s]})" if s in rank else f"{s} (fell off the shortlist)" for s in d["dropped"]]
        out.append(f"Dropping {len(why)}: {', '.join(why)}.")
    for s in [s for s in d["chosen"] if s not in d["kept"]]:
        line = f"Adding {s}: ranked #{rank[s]}"
        if d.get("scores"):
            line += f", expected {d['scores'][s]:+.2f}% vs the average stock next month"
        if d.get("reasons"):
            parts = [f"{label(f)} {v:+.2f}" for f, v in d["reasons"][s][:3]]
            line += ". Biggest reasons: " + "; ".join(parts)
        else:
            line += f", 12-month momentum {d['momentum'][s]:+.0%}"
        out.append(line + ".")
    return out
