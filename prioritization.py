"""
Prioritization Matrix for Coinbase Diversification Opportunities

Scores 10 potential diversification opportunities across 5 dimensions:
1. BTC Decorrelation — how uncorrelated is this revenue model from Bitcoin price?
2. Market Size — how large is the addressable market? (log-scaled)
3. Competitive Gap — do competitors offer this while Coinbase doesn't?
4. Customer Demand — Reddit community engagement (upvotes from r/CoinBase)
5. Feasibility — regulatory barriers, existing infrastructure, technical complexity

Outputs: data/prioritization_matrix.csv, prioritization_matrix.png, weighted_rankings.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ===== LOAD DATA =====
earnings = pd.read_csv("data/coinbase_earnings.csv", parse_dates=["date"])
market = pd.read_csv("data/market_sizing.csv")
reddit = pd.read_csv("data/reddit_demand_scores.csv")
features = pd.read_csv("data/competitor_features.csv")

# ===== OPPORTUNITIES TO SCORE =====
opportunities = [
    "Staking / Earn",
    "Banking / Cash Management",
    "Crypto Debit Card",
    "Lending / Borrowing",
    "Stablecoin Payments",
    "Base (L2 Blockchain)",
    "DeFi / Wallet",
    "Subscription (Coinbase One)",
    "Institutional Custody",
    "Advanced Trading / Derivatives"
]


# ============================================================
# DIMENSION 1: BTC DECORRELATION
# Each opportunity is mapped to its closest existing revenue stream.
# Lower correlation with BTC = higher score (inverted).
# ============================================================

btc_hist = yf.download("BTC-USD", start="2020-10-01", end="2025-10-01", interval="1d")
btc_hist = btc_hist.reset_index()
btc_hist["quarter_end"] = btc_hist["Date"].dt.to_period("Q").dt.end_time.dt.normalize()
btc_q = btc_hist.groupby("quarter_end")["Close"].mean().reset_index()
btc_q.columns = ["date", "btc_avg_price"]
btc_q["date"] = pd.to_datetime(btc_q["date"])

rev_merged = earnings.sort_values("date").merge(btc_q, on="date", how="left")

def calc_corr(col):
    """Calculate correlation between a revenue column and BTC quarterly price."""
    mask = (rev_merged[col] > 0) & (rev_merged["btc_avg_price"].notna())
    if mask.sum() >= 4:
        return rev_merged.loc[mask, col].corr(rev_merged.loc[mask, "btc_avg_price"])
    return None

revenue_correlations = {
    "retail_transactions": calc_corr("retail_transactions"),
    "institutional_transactions": calc_corr("institutional_transactions"),
    "stablecoin_revenue": calc_corr("stablecoin_revenue"),
    "blockchain_rewards": calc_corr("blockchain_rewards"),
    "interest_income": calc_corr("interest_income"),
    "custodial_fees": calc_corr("custodial_fees"),
    "other_subscriptions": calc_corr("other_subscriptions"),
}

# Map each opportunity to the most relevant existing revenue stream
decorrelation_map = {
    "Staking / Earn": revenue_correlations["blockchain_rewards"],
    "Banking / Cash Management": revenue_correlations["interest_income"],
    "Crypto Debit Card": revenue_correlations["interest_income"],
    "Lending / Borrowing": revenue_correlations["interest_income"],
    "Stablecoin Payments": revenue_correlations["stablecoin_revenue"],
    "Base (L2 Blockchain)": revenue_correlations["other_subscriptions"],
    "DeFi / Wallet": revenue_correlations["blockchain_rewards"],
    "Subscription (Coinbase One)": revenue_correlations["other_subscriptions"],
    "Institutional Custody": revenue_correlations["custodial_fees"],
    "Advanced Trading / Derivatives": revenue_correlations["retail_transactions"],
}

def corr_to_score(corr):
    """Convert BTC correlation to 1-5 score. Lower correlation = higher score."""
    if corr is None:
        return 3
    return max(1, min(5, round(5 - (corr * 4))))

btc_scores = {k: corr_to_score(v) for k, v in decorrelation_map.items()}


# ============================================================
# DIMENSION 2: MARKET SIZE
# Log-scaled scoring to handle the wide range ($0.5B to $245B).
# ============================================================

market_map = {
    "Staking / Earn": 245.0,
    "Banking / Cash Management": 10.0,
    "Crypto Debit Card": 1.7,
    "Lending / Borrowing": 10.7,
    "Stablecoin Payments": 170.0,
    "Base (L2 Blockchain)": 31.0,
    "DeFi / Wallet": 54.0,
    "Subscription (Coinbase One)": 0.5,
    "Institutional Custody": 3.3,
    "Advanced Trading / Derivatives": 6.0,
}

log_values = [np.log10(max(v, 0.1)) for v in market_map.values()]
min_log, max_log = min(log_values), max(log_values)

def market_to_score(val):
    """Convert market size ($B) to 1-5 score using log scale."""
    log_val = np.log10(max(val, 0.1))
    normalized = (log_val - min_log) / (max_log - min_log)
    return max(1, min(5, round(1 + normalized * 4)))

market_scores = {k: market_to_score(v) for k, v in market_map.items()}


# ============================================================
# DIMENSION 3: COMPETITIVE GAP
# Count competitors offering what Coinbase doesn't.
# More competitors with it = bigger gap = higher priority.
# ============================================================

gap_map = {
    "Staking / Earn": 0,
    "Banking / Cash Management": 2,  # Robinhood + CashApp
    "Crypto Debit Card": 0,
    "Lending / Borrowing": 2,        # Binance + Kraken
    "Stablecoin Payments": 0,
    "Base (L2 Blockchain)": 0,
    "DeFi / Wallet": 1,              # Binance
    "Subscription (Coinbase One)": 0,
    "Institutional Custody": 0,
    "Advanced Trading / Derivatives": 0,
}

def gap_to_score(gap):
    """Convert competitive gap count to 1-5 score."""
    mapping = {0: 1, 1: 2.5, 2: 4, 3: 5, 4: 5}
    return mapping.get(gap, 5)

gap_scores = {k: gap_to_score(v) for k, v in gap_map.items()}


# ============================================================
# DIMENSION 4: CUSTOMER DEMAND (Reddit)
# Upvotes from r/CoinBase normalized to 1-5 scale.
# ============================================================

reddit_map = {
    "Staking / Earn": "Staking / Earn",
    "Banking / Cash Management": "Banking / Payments",
    "Crypto Debit Card": "Debit Card",
    "Lending / Borrowing": None,
    "Stablecoin Payments": "Stablecoins / USDC",
    "Base (L2 Blockchain)": "DeFi / Wallet",
    "DeFi / Wallet": "DeFi / Wallet",
    "Subscription (Coinbase One)": None,
    "Institutional Custody": None,
    "Advanced Trading / Derivatives": "Advanced Trading",
}

reddit_dict = dict(zip(reddit["category"], reddit["total_upvotes"]))
max_upvotes = max(reddit_dict.values())

def demand_to_score(opp):
    """Convert Reddit upvotes to 1-5 score. No data = 2 (low-moderate)."""
    reddit_cat = reddit_map.get(opp)
    if reddit_cat is None:
        return 2
    upvotes = reddit_dict.get(reddit_cat, 0)
    normalized = upvotes / max_upvotes
    return max(1, min(5, round(1 + normalized * 4)))

demand_scores = {opp: demand_to_score(opp) for opp in opportunities}


# ============================================================
# DIMENSION 5: FEASIBILITY (judgment-based)
# Considers: existing products, regulatory barriers, technical complexity.
# ============================================================

feasibility_scores = {
    "Staking / Earn": 5,               # Already operational, expand to more assets
    "Banking / Cash Management": 2,    # Requires banking license/partnership, heavy regulation
    "Crypto Debit Card": 4,            # Coinbase Card exists, expand features
    "Lending / Borrowing": 3,          # USDC lending launched late 2025, regulatory caution
    "Stablecoin Payments": 5,          # Aggressively building, Shopify integration live
    "Base (L2 Blockchain)": 4,         # Base already live with growing ecosystem
    "DeFi / Wallet": 4,               # Coinbase Wallet exists, integrating Jupiter DEX
    "Subscription (Coinbase One)": 5,  # Already exists, add features
    "Institutional Custody": 5,        # Market leader, custodies for BlackRock ETF
    "Advanced Trading / Derivatives": 5, # Acquired Deribit ($2.9B, May 2025)
}


# ============================================================
# BUILD THE MATRIX
# ============================================================

results = []
for opp in opportunities:
    results.append({
        "opportunity": opp,
        "btc_decorrelation": btc_scores[opp],
        "market_size": market_scores[opp],
        "competitive_gap": gap_scores[opp],
        "customer_demand": demand_scores[opp],
        "feasibility": feasibility_scores[opp],
        "total": (btc_scores[opp] + market_scores[opp] + gap_scores[opp] 
                  + demand_scores[opp] + feasibility_scores[opp])
    })

results_df = pd.DataFrame(results).sort_values("total", ascending=False)
results_df.to_csv("data/prioritization_matrix.csv", index=False)


# ============================================================
# VISUALIZATION: Stacked bar chart
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

results_sorted = results_df.sort_values("total", ascending=True)
y_pos = range(len(results_sorted))

dimensions = ["btc_decorrelation", "market_size", "competitive_gap", "customer_demand", "feasibility"]
dim_labels = ["BTC Decorrelation", "Market Size", "Competitive Gap", "Customer Demand", "Feasibility"]
dim_colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#2ecc71"]

left = np.zeros(len(results_sorted))
for dim, label, color in zip(dimensions, dim_labels, dim_colors):
    values = results_sorted[dim].values
    ax.barh(y_pos, values, left=left, color=color, label=label, edgecolor="white", linewidth=0.5)
    left += values

ax.set_yticks(y_pos)
ax.set_yticklabels(results_sorted["opportunity"])
ax.set_xlabel("Total Score (max 25)")
ax.set_title("Coinbase Diversification Prioritization Matrix")
ax.legend(loc="lower right", fontsize=9)

for i, total in enumerate(results_sorted["total"]):
    ax.text(total + 0.3, i, f"{total:.0f}", va="center", fontweight="bold")

fig.tight_layout()
plt.savefig("prioritization_matrix.png", dpi=150)
plt.close()


# ============================================================
# WEIGHTED SENSITIVITY ANALYSIS
# Tests how rankings shift when different strategic goals are prioritized.
# ============================================================

weight_schemes = {
    "Equal": {
        "btc_decorrelation": 1, "market_size": 1, "competitive_gap": 1,
        "customer_demand": 1, "feasibility": 1
    },
    "Diversification (2x BTC)": {
        "btc_decorrelation": 2, "market_size": 1, "competitive_gap": 1,
        "customer_demand": 1, "feasibility": 1
    },
    "Growth (2x Market)": {
        "btc_decorrelation": 1, "market_size": 2, "competitive_gap": 1,
        "customer_demand": 1, "feasibility": 1
    },
    "User-Driven (2x Demand)": {
        "btc_decorrelation": 1, "market_size": 1, "competitive_gap": 1,
        "customer_demand": 2, "feasibility": 1
    },
    "Quick Wins (2x Feasibility)": {
        "btc_decorrelation": 1, "market_size": 1, "competitive_gap": 1,
        "customer_demand": 1, "feasibility": 2
    },
}

short_names = ["Equal", "Diversification", "Growth", "User-Driven", "Quick Wins"]

# Track rank changes across weight schemes
fig, ax = plt.subplots(figsize=(12, 7))

for opp in opportunities:
    ranks_across = []
    for scheme_name, weights in weight_schemes.items():
        scored = []
        for r in results:
            w_total = sum(r[dim] * weights[dim] for dim in dimensions)
            scored.append({"opportunity": r["opportunity"], "score": w_total})
        scored_df = pd.DataFrame(scored).sort_values("score", ascending=False)
        scored_df["rank"] = scored_df["score"].rank(ascending=False, method="min")
        rank = scored_df[scored_df["opportunity"] == opp]["rank"].values[0]
        ranks_across.append(rank)

    # Only plot opportunities that reach top 5 in at least one scheme
    if min(ranks_across) <= 5:
        color = ("#e74c3c" if opp == "Banking / Cash Management" else
                 "#3498db" if opp == "Advanced Trading / Derivatives" else
                 "#2ecc71" if opp == "Lending / Borrowing" else "gray")
        linewidth = 2.5 if min(ranks_across) <= 3 else 1
        ax.plot(short_names, ranks_across, marker="o", label=opp,
                linewidth=linewidth, color=color, markersize=6)

ax.set_ylabel("Rank (1 = best)")
ax.set_xlabel("Strategic Priority")
ax.set_title("How Rankings Shift Depending on Strategic Goals")
ax.invert_yaxis()
ax.set_ylim(10.5, 0.5)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("weighted_rankings.png", dpi=150)
plt.close()

print("Outputs saved:")
print("  data/prioritization_matrix.csv")
print("  prioritization_matrix.png")
print("  weighted_rankings.png")
