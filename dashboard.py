import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="Coinbase Diversification Analysis",
    layout="wide"
)

# ===== SIDEBAR NAVIGATION =====
st.sidebar.title("Navigation")
st.sidebar.markdown("Jump to any section:")

sections = {
    "1. Bitcoin Dependency": "dependency",
    "2. Why It's Bad": "why-bad",
    "3. The Structural Shift": "shift",
    "4. Market Sizing": "market",
    "5. Competitive Landscape": "competitive",
    "6. Customer Demand": "reddit",
    "7. Revenue BTC Correlations": "revenue",
    "8. Feasibility": "feasibility",
    "9. Prioritization Matrix": "matrix",
    "10. Strategic Recommendations": "recommendations",
    "11. The Bottom Line": "bottom-line",
}

for label, anchor in sections.items():
    st.sidebar.markdown(f"[{label}](#{anchor})")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by Hiram Lannes**")
st.sidebar.markdown("Data Sources: Coinbase 10-Q filings, Yahoo Finance, Reddit, industry research reports")

# ===== TITLE =====
st.title("Decoupling from Bitcoin")
st.markdown("### A Data-Driven Diversification Strategy for Coinbase")
st.markdown("""
*Coinbase is the largest US crypto exchange with 110+ million verified users. But what happens to their business 
when Bitcoin crashes? This analysis uses 20 quarters of earnings data, Reddit NLP, competitive feature mapping, 
and market research to build a prioritized diversification roadmap.*
""")
st.markdown("---")

# ----- LOAD DATA -----
@st.cache_data
def load_data():
    btc = pd.read_csv("data/btc_data.csv", parse_dates=["date"])
    coin = pd.read_csv("data/coin_data.csv", parse_dates=["date"])
    return btc, coin

btc_df, coin_df = load_data()

# Load earnings data
earnings = pd.read_csv("data/coinbase_earnings.csv", parse_dates=["date"])
earnings = earnings.sort_values("date").reset_index(drop=True)

# Get BTC quarterly prices
import yfinance as yf

@st.cache_data
def load_btc_quarterly():
    btc_hist = yf.download("BTC-USD", start="2020-10-01", end="2025-10-01", interval="1d")
    btc_hist = btc_hist.reset_index()
    btc_hist["quarter_end"] = btc_hist["Date"].dt.to_period("Q").dt.end_time.dt.normalize()
    btc_q = btc_hist.groupby("quarter_end")["Close"].mean().reset_index()
    btc_q.columns = ["date", "btc_avg_price"]
    btc_q["date"] = pd.to_datetime(btc_q["date"])
    return btc_q

btc_q = load_btc_quarterly()
rev_merged = earnings.merge(btc_q, on="date", how="left")
rev_merged["total_trading"] = rev_merged["retail_transactions"] + rev_merged["institutional_transactions"]


# Calculate revenue correlations (used in multiple sections)
revenue_columns = {
    "retail_transactions": "Retail Trading Fees",
    "institutional_transactions": "Institutional Trading",
    "stablecoin_revenue": "Stablecoin Revenue",
    "blockchain_rewards": "Blockchain Rewards",
    "interest_income": "Interest Income",
    "custodial_fees": "Custodial Fees",
    "other_subscriptions": "Other Subscriptions",
    "other": "Other"
}

correlations = {}
for col, name in revenue_columns.items():
    mask = (rev_merged[col] > 0) & (rev_merged["btc_avg_price"].notna())
    if mask.sum() >= 4:
        corr = rev_merged.loc[mask, col].corr(rev_merged.loc[mask, "btc_avg_price"])
        correlations[name] = corr


# ============================================================
# SECTION 1: BITCOIN DEPENDENCY
# ============================================================
st.markdown('<a id="dependency"></a>', unsafe_allow_html=True)
st.header("1. Bitcoin Dependency")
st.markdown("""
> Does Coinbase's revenue rise and fall with Bitcoin? I compared 20 quarters of 
> earnings data against BTC price to find out.
""")

# Dual-axis chart: BTC price vs retail trading revenue
fig, ax1 = plt.subplots(figsize=(12, 6))

plot_data = rev_merged.dropna(subset=["btc_avg_price"]).sort_values("date")

ax1.set_xlabel("Quarter")
ax1.set_ylabel("BTC Quarterly Avg Price ($)", color="orange")
ax1.plot(plot_data["date"], plot_data["btc_avg_price"], color="orange", linewidth=2.5, marker="o", markersize=6, label="BTC Price")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.set_ylabel("Retail Trading Revenue ($M)", color="#e74c3c")
ax2.bar(plot_data["date"], plot_data["total_trading"], width=60, color="#e74c3c", alpha=0.6, label="Retail Trading Revenue")
ax2.tick_params(axis="y", labelcolor="#e74c3c")

plt.title("BTC Price Keeps Rising — Trading Revenue Doesn't Follow")
fig.tight_layout()
st.pyplot(fig)
plt.close()

# Revenue per $1K of BTC
plot_data = plot_data.copy()
plot_data["rev_per_1k_btc"] = plot_data["total_trading"] / (plot_data["btc_avg_price"] / 1000)

col1, col2, col3 = st.columns(3)

early_yield = plot_data[plot_data["date"] <= "2021-12-31"]["rev_per_1k_btc"].mean()
mid_yield = plot_data[(plot_data["date"] >= "2022-01-01") & (plot_data["date"] <= "2023-12-31")]["rev_per_1k_btc"].mean()
recent_yield = plot_data[plot_data["date"] >= "2024-01-01"]["rev_per_1k_btc"].mean()

col1.metric("2020-2021 Avg", f"${early_yield:.1f}M", "per $1K of BTC")
col2.metric("2022-2023 Avg", f"${mid_yield:.1f}M", f"{((mid_yield - early_yield) / early_yield * 100):+.0f}% decline", delta_color="inverse")
col3.metric("2024-2025 Avg", f"${recent_yield:.1f}M", f"{((recent_yield - early_yield) / early_yield * 100):+.0f}% decline", delta_color="inverse")

st.warning("""
Each crypto cycle generates less trading revenue per dollar of BTC price.
BTC nearly doubled from 45k (Q1 2021) to 89K in (Q1 2025), but trading revenue 
fell from 34.1M to 12.2M. Coinbase still has BTC downside exposure, 
but gets diminishing returns from BTC upside.
""")

st.markdown("---")

# ============================================================
# SECTION 2: WHY Bitcoin Dependency is a problem
# ============================================================
st.markdown('<a id="why-bad"></a>', unsafe_allow_html=True)
st.header("2. Why Bitcoin Dependency Is a Problem")
st.markdown("""
> The dependency runs deep, from Wall Street's pricing of COIN stock 
> to the composition of Coinbase's revenue.
""")

# --- Stock correlation ---
st.subheader("Stock Correlation")

coin_merged = btc_df.merge(coin_df, on="date", how="inner")
coin_merged["btc_change"] = coin_merged["price"].pct_change() * 100
coin_merged["coin_change"] = coin_merged["coin_price"].pct_change() * 100

# --- Part B: Stock correlation (condensed) ---
coin_merged = btc_df.merge(coin_df, on="date", how="inner")
coin_merged["btc_change"] = coin_merged["price"].pct_change() * 100
coin_merged["coin_change"] = coin_merged["coin_price"].pct_change() * 100

btc_coin_corr = coin_merged["price"].corr(coin_merged["coin_price"])
stock_downturns = coin_merged[coin_merged["btc_change"] < -3]
avg_coin_drop = stock_downturns["coin_change"].mean()

col1, col2 = st.columns(2)
col1.metric("BTC vs COIN Stock Correlation", f"{btc_coin_corr:.3f}", "Very strong dependency", delta_color="inverse")
col2.metric("Avg COIN Drop on BTC Down Days", f"{avg_coin_drop:.2f}%", f"across {len(stock_downturns)} days", delta_color="inverse")

st.markdown("COIN stock moves almost in lockstep with Bitcoin.")

# --- Part C: Revenue concentration ---
st.subheader("Revenue Is Concentrated in BTC-Dependent Streams")

latest = earnings.sort_values("date").iloc[-1]
latest_date = latest["date"].strftime("%B %Y")

rev_streams = {
    "Retail Trading": latest["retail_transactions"],
    "Stablecoin Revenue": latest["stablecoin_revenue"],
    "Blockchain Rewards": latest["blockchain_rewards"],
    "Other Subscriptions": latest["other_subscriptions"],
    "Institutional Trading": latest["institutional_transactions"],
    "Other": latest["other"],
    "Interest Income": latest["interest_income"],
    "Custodial Fees": latest["custodial_fees"]
}

total_rev = sum(rev_streams.values())
rev_sorted = dict(sorted(rev_streams.items(), key=lambda x: x[1], reverse=True))

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 6))
    
    colors_pie = []
    for name in rev_sorted.keys():
        if "Trading" in name:
            colors_pie.append("#e74c3c")
        else:
            colors_pie.append("#2ecc71")
    
    wedges, texts, autotexts = ax.pie(
        rev_sorted.values(),
        labels=rev_sorted.keys(),
        autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
        colors=colors_pie,
        startangle=90
    )
    
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
    
    ax.set_title(f"Revenue Breakdown ({latest_date})")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    trading_rev = rev_streams["Retail Trading"] + rev_streams["Institutional Trading"]
    trading_pct = (trading_rev / total_rev) * 100
    non_trading_pct = 100 - trading_pct
    
    st.metric("Total Revenue", f"${total_rev:,.0f}M")
    st.metric("Trading Revenue", f"{trading_pct:.1f}%", "BTC-dependent", delta_color="inverse")
    st.metric("Non-Trading Revenue", f"{non_trading_pct:.1f}%", "More resilient")
    
    st.markdown("---")
    st.markdown(f"**{len(rev_streams)} revenue streams** but trading still dominates.")
    st.markdown("Red = transaction-based (BTC-dependent)")
    st.markdown("Green = subscription/services (more resilient)")

# Revenue concentration over time
st.subheader("Is Revenue Diversifying Over Time?")

earnings_sorted = earnings.sort_values("date")
earnings_sorted["total"] = earnings_sorted[["retail_transactions", "institutional_transactions", 
    "stablecoin_revenue", "blockchain_rewards", "interest_income", 
    "custodial_fees", "other_subscriptions", "other"]].sum(axis=1)
earnings_sorted["trading_pct"] = (
    (earnings_sorted["retail_transactions"] + earnings_sorted["institutional_transactions"]) 
    / earnings_sorted["total"] * 100
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(earnings_sorted["date"], earnings_sorted["trading_pct"], 
        color="#e74c3c", linewidth=2, marker="o", markersize=5)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
ax.fill_between(earnings_sorted["date"], earnings_sorted["trading_pct"], 50,
                where=earnings_sorted["trading_pct"] > 50,
                alpha=0.2, color="#e74c3c", label="Over-concentrated")
ax.set_ylabel("Trading Revenue as % of Total")
ax.set_title("Coinbase Revenue Concentration Over Time")
ax.legend()
ax.set_ylim(0, 100)
fig.tight_layout()
st.pyplot(fig)
plt.close()

st.info("Despite launching 20+ products, trading fees still account for the majority of revenue. However, Coinbase does seem to be diversifying. But the challenge isn't just building more products, but rather shifting user adoption toward existing non-trading products.")

st.markdown("---")


# ============================================================
# SECTION 3: THE STRUCTURAL SHIFT
# ============================================================
st.markdown('<a id="shift"></a>', unsafe_allow_html=True)
st.header("3. The Structural Shift")
st.markdown("""
> The revenue data revealed something unexpected. There's a deeper problem than simple BTC correlation.
> Even when BTC recovers, trading revenue doesn't fully come back. So Coinbase needs to diversify.
""")

early_mask = rev_merged["date"] < "2023-01-01"
recent_mask = rev_merged["date"] >= "2023-01-01"

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(rev_merged.loc[early_mask, "btc_avg_price"],
           rev_merged.loc[early_mask, "total_trading"],
           color="red", s=80, alpha=0.7, label="2020-2022", zorder=5)
ax.scatter(rev_merged.loc[recent_mask, "btc_avg_price"],
           rev_merged.loc[recent_mask, "total_trading"],
           color="blue", s=80, alpha=0.7, label="2023-2025", zorder=5)

for mask, color in [(early_mask, "red"), (recent_mask, "blue")]:
    subset = rev_merged.loc[mask].dropna(subset=["btc_avg_price", "total_trading"])
    if len(subset) > 1:
        z = np.polyfit(subset["btc_avg_price"], subset["total_trading"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset["btc_avg_price"].min(), subset["btc_avg_price"].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linestyle="--", alpha=0.5)

ax.set_xlabel("BTC Quarterly Average Price ($)")
ax.set_ylabel("Trading Revenue ($M)")
ax.set_title("Same BTC Price, Less Trading Revenue")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close()

early_corr = (rev_merged.loc[early_mask, "retail_transactions"] + rev_merged.loc[early_mask, "institutional_transactions"]).corr(rev_merged.loc[early_mask, "btc_avg_price"])
recent_corr = (rev_merged.loc[recent_mask, "retail_transactions"] + rev_merged.loc[recent_mask, "institutional_transactions"]).corr(rev_merged.loc[recent_mask, "btc_avg_price"])

col1, col2 = st.columns(2)
col1.metric("BTC-Trading Correlation (2020-2022)", f"{early_corr:.3f}", "Bull run era")
col2.metric("BTC-Trading Correlation (2023-2025)", f"{recent_corr:.3f}", "Post-crash era")

st.warning("""
Coinbase doesn't just have a Bitcoin problem.
Trading revenue remains correlated with BTC within each period (0.71-0.93), but the overall level has permanently declined. 
Each crypto cycle brings fewer retail traders back, making diversification increasingly urgent. This seems to be because of the crypto market maturing and becoming less volatile over time.

The rest of this analysis explores where Coinbase should diversify, using five dimensions: 
Market Size, Competitive Gaps, Customer Demand, Revenue BTC Correlation, and Feasibility.
""")

st.markdown("---")


# ============================================================
# SECTION 4: MARKET SIZING
# ============================================================
st.markdown('<a id="market"></a>', unsafe_allow_html=True)
st.header("4. Market Opportunity Sizing")
st.markdown("""
> First dimension: how big is each potential market? Not every opportunity is worth pursuing 
> if the addressable market is tiny.
""")

market_data = pd.read_csv("data/market_sizing.csv")
market_data = market_data.sort_values("market_size_2025_billions", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

colors_market = []
for _, row in market_data.iterrows():
    if row["cagr_percent"] >= 25:
        colors_market.append("#2ecc71")
    elif row["cagr_percent"] >= 15:
        colors_market.append("#f39c12")
    else:
        colors_market.append("#3498db")

bars = ax.barh(market_data["product"], market_data["market_size_2025_billions"], color=colors_market)
ax.set_xlabel("Estimated 2025 Market Size ($ Billions)")
ax.set_title("How Big Is Each Opportunity?")
ax.set_xscale("log")

for bar, val in zip(bars, market_data["market_size_2025_billions"]):
    ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
            f"${val:.1f}B", va="center", fontsize=10)

fig.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("🟢 High growth (25%+ CAGR) · 🟠 Moderate growth (15-25% CAGR) · 🔵 Steady growth (<15% CAGR)")

st.subheader("Growth Rates")
market_sorted_growth = market_data.sort_values("cagr_percent", ascending=False)

cols = st.columns(3)
for i, (_, row) in enumerate(market_sorted_growth.head(3).iterrows()):
    cols[i].metric(
        row["product"],
        f"{row['cagr_percent']:.0f}% CAGR",
        f"${row['market_size_2025_billions']:.1f}B market"
    )

st.warning("Market sizes are from different research firms and measure different things. But the big picture here is what we're after.")

st.markdown("---")


# ============================================================
# SECTION 5: COMPETITIVE LANDSCAPE
# ============================================================
st.markdown('<a id="competitive"></a>', unsafe_allow_html=True)
st.header("5. Competitive Feature Mapping")
st.markdown("""
> Second dimension: what do competitors already offer that Coinbase doesn't? 
> And where does Coinbase have a unique advantage?
""")

features = pd.read_csv("data/competitor_features.csv")

categories = features["category"].unique().tolist()
selected_cat = st.selectbox("Filter by category:", ["All"] + categories)

if selected_cat != "All":
    display_df = features[features["category"] == selected_cat]
else:
    display_df = features.copy()

table_df = display_df[["product_service", "coinbase", "robinhood", "binance", "kraken", "cashapp"]].copy()
table_df.columns = ["Product / Service", "Coinbase", "Robinhood", "Binance", "Kraken", "CashApp"]

st.dataframe(table_df, use_container_width=True, hide_index=True)

# Gap analysis
st.subheader("Coinbase's Competitive Gaps")
st.markdown("Products where competitors offer something Coinbase does **not**:")

competitors = ["robinhood", "binance", "kraken", "cashapp"]
gaps = []
for _, row in features.iterrows():
    if row["coinbase"] == "N":
        has_it = [c.title() for c in competitors if row[c] == "Y"]
        if has_it:
            gaps.append({
                "product": row["product_service"],
                "offered_by": ", ".join(has_it),
                "num_competitors": len(has_it)
            })

gaps_df = pd.DataFrame(gaps).sort_values("num_competitors", ascending=False)

if len(gaps_df) > 0:
    for _, gap in gaps_df.iterrows():
        st.markdown(f"- **{gap['product']}** — offered by {gap['offered_by']}")

# Unique advantages
st.subheader("Coinbase's Unique Advantages")
st.markdown("Products where Coinbase offers something **no competitor** does:")

for _, row in features.iterrows():
    if row["coinbase"] == "Y":
        others_have = any(row[c] == "Y" for c in competitors)
        if not others_have:
            st.markdown(f"- **{row['product_service']}** — {row['notes']}")

st.markdown("---")


# ============================================================
# SECTION 6: CUSTOMER DEMAND (REDDIT)
# ============================================================
st.markdown('<a id="reddit"></a>', unsafe_allow_html=True)
st.header("6. Customer Demand")
st.markdown("""
> Third dimension: what do actual Coinbase users want? I analyzed ~500 posts 
> from r/CoinBase to measure demand by volume and engagement.
""")

reddit_demand = pd.read_csv("data/reddit_demand_scores.csv")

complaints = reddit_demand[reddit_demand["category"].isin(["Customer Support", "Security"])]
opps = reddit_demand[~reddit_demand["category"].isin(["Customer Support", "Security"])]
opps = opps.sort_values("total_upvotes", ascending=True)

view = st.radio("View:", ["Product Opportunities", "All Categories (including complaints)"], horizontal=True)

if view == "All Categories (including complaints)":
    plot_data = reddit_demand.sort_values("total_upvotes", ascending=True)
else:
    plot_data = opps

fig, ax = plt.subplots(figsize=(10, 6))

colors_reddit = []
for _, row in plot_data.iterrows():
    if row["category"] in ["Customer Support", "Security"]:
        colors_reddit.append("gray")
    elif row["total_upvotes"] > 8000:
        colors_reddit.append("coral")
    elif row["total_upvotes"] > 4000:
        colors_reddit.append("orange")
    else:
        colors_reddit.append("steelblue")

ax.barh(plot_data["category"], plot_data["total_upvotes"], color=colors_reddit)
ax.set_xlabel("Total Upvotes (Community Engagement)")
ax.set_title("r/CoinBase: Product Demand by Community Engagement")
fig.tight_layout()
st.pyplot(fig)
plt.close()

top3_reddit = opps.nlargest(3, "total_upvotes")
cols = st.columns(3)
for col, (_, row) in zip(cols, top3_reddit.iterrows()):
    col.metric(
        f"#{top3_reddit.index.get_loc(_)+1}: {row['category']}",
        f"{row['total_upvotes']:,} upvotes",
        f"{row['post_count']} posts"
    )

st.markdown("Users want Coinbase to become a broader financial platform (DeFi, Banking, Staking), not just a place to buy and sell Bitcoin.")

st.markdown("---")


# ============================================================
# SECTION 7: REVENUE STREAM BTC CORRELATIONS
# ============================================================
st.markdown('<a id="revenue"></a>', unsafe_allow_html=True)
st.header("7. Revenue Stream BTC Correlations")
st.markdown("""
> Fourth dimension: which of Coinbase's existing revenue streams are most and least 
> tied to Bitcoin? This tells us which revenue models to replicate.
""")

corr_df = pd.DataFrame({
    "stream": list(correlations.keys()),
    "correlation": list(correlations.values())
}).sort_values("correlation", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
colors_corr = ["green" if c < 0.4 else "orange" if c < 0.7 else "red" for c in corr_df["correlation"]]
ax.barh(corr_df["stream"], corr_df["correlation"], color=colors_corr, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Correlation with BTC Quarterly Average Price")
ax.set_title("Which Revenue Streams Are Most Tied to Bitcoin?")
ax.axvline(x=0.4, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=0.7, color="gray", linestyle=":", alpha=0.5)
ax.set_xlim(-0.1, 1.05)
fig.tight_layout()
st.pyplot(fig)
plt.close()

col1, col2, col3 = st.columns(3)
sorted_corr = sorted(correlations.items(), key=lambda x: x[1])
col1.metric("Least BTC-Dependent", sorted_corr[0][0], f"Correlation: {sorted_corr[0][1]:.3f}")
col2.metric("2nd Least Dependent", sorted_corr[1][0], f"Correlation: {sorted_corr[1][1]:.3f}")
col3.metric("Most BTC-Dependent", sorted_corr[-1][0], f"Correlation: {sorted_corr[-1][1]:.3f}", delta_color="inverse")

st.markdown("Low dependency (<0.4) · Moderate (0.4-0.7) · High dependency (>0.7)")
st.info("Interest Income (0.117) and Custodial Fees are the most resilient revenue streams. New products should replicate these revenue models. These are recurring, service-based income rather than transaction fees.")

st.markdown("---")


# ============================================================
# SECTION 8: FEASIBILITY
# ============================================================
st.markdown('<a id="feasibility"></a>', unsafe_allow_html=True)
st.header("8. Feasibility Assessment")
st.markdown("""
> The final dimension is qualitative: how realistic is each opportunity given 
> Coinbase's current capabilities, regulatory environment, and recent moves?
""")

feasibility_data = {
    "Staking / Earn": {"score": 5, "rationale": "Already operational. Coinbase has 5.1% of ETH staking. Just expand to more assets."},
    "Banking / Cash Management": {"score": 2, "rationale": "Requires banking license or bank partnership. Heavy regulatory burden. Furthest from current capabilities."},
    "Crypto Debit Card": {"score": 4, "rationale": "Coinbase Card already exists. Expand features and distribution."},
    "Lending / Borrowing": {"score": 3, "rationale": "USDC lending launched late 2025. Regulatory caution needed after Celsius/BlockFi collapses."},
    "Stablecoin Payments": {"score": 5, "rationale": "Aggressively building. Shopify integration live. Custom stablecoins launched Dec 2025."},
    "Base (L2 Blockchain)": {"score": 4, "rationale": "Base already live with growing ecosystem. Requires continued developer investment."},
    "DeFi / Wallet": {"score": 4, "rationale": "Coinbase Wallet exists. Integrating Jupiter DEX for Solana. Infrastructure in place."},
    "Subscription (Coinbase One)": {"score": 5, "rationale": "Already exists. Add features to increase subscriber value."},
    "Institutional Custody": {"score": 5, "rationale": "Market leader. Qualified custodian. Custodies assets for BlackRock ETF."},
    "Advanced Trading / Derivatives": {"score": 5, "rationale": "Acquired Deribit for $2.9B (May 2025). Already have futures. Integration underway."},
}

feas_df = pd.DataFrame([
    {"Opportunity": k, "Score": v["score"], "Rationale": v["rationale"]} 
    for k, v in feasibility_data.items()
]).sort_values("Score", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

colors_feas = []
for s in feas_df["Score"]:
    if s >= 4:
        colors_feas.append("#2ecc71")
    elif s >= 3:
        colors_feas.append("#f39c12")
    else:
        colors_feas.append("#e74c3c")

ax.barh(feas_df["Opportunity"], feas_df["Score"], color=colors_feas)
ax.set_xlabel("Feasibility Score (1 = Hard, 5 = Easy)")
ax.set_title("How Realistic Is Each Opportunity?")
ax.set_xlim(0, 5.5)

for i, (_, row) in enumerate(feas_df.iterrows()):
    ax.text(row["Score"] + 0.1, i, f'{row["Score"]}/5', va="center", fontweight="bold")

fig.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("Ready to scale (4-5) · Possible with investment (3) · Significant barriers (1-2)")

st.subheader("Scoring Rationale")
st.markdown("*Unlike the other dimensions, feasibility is judgment-based. Here's the reasoning:*")

for _, row in feas_df.sort_values("Score", ascending=False).iterrows():
    st.markdown(f"- **{row['Opportunity']}** ({row['Score']}/5): {row['Rationale']}")

st.warning("Feasibility is inversely related to strategic value for diversification. "
           "The opportunities that score lowest on feasibility (Banking at 2/5) are often the ones "
           "with the highest decorrelation potential because they're hard and nobody has done them yet.")

st.markdown("---")


# ============================================================
# SECTION 9: PRIORITIZATION MATRIX
# ============================================================
st.markdown('<a id="matrix"></a>', unsafe_allow_html=True)
st.header("9. Prioritization Matrix")

matrix = pd.read_csv("data/prioritization_matrix.csv")
matrix_sorted = matrix.sort_values("total", ascending=False)

# Stacked bar chart
fig, ax = plt.subplots(figsize=(12, 7))

matrix_plot = matrix_sorted.sort_values("total", ascending=True)
y_pos = range(len(matrix_plot))

dims = ["btc_decorrelation", "market_size", "competitive_gap", "customer_demand", "feasibility"]
dim_labels = ["BTC Decorrelation", "Market Size", "Competitive Gap", "Customer Demand", "Feasibility"]
dim_colors = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#2ecc71"]

left = np.zeros(len(matrix_plot))
for dim, label, color in zip(dims, dim_labels, dim_colors):
    values = matrix_plot[dim].values
    ax.barh(y_pos, values, left=left, color=color, label=label, edgecolor="white", linewidth=0.5)
    left += values

ax.set_yticks(y_pos)
ax.set_yticklabels(matrix_plot["opportunity"])
ax.set_xlabel("Total Score (max 25)")
ax.set_title("Coinbase Diversification Prioritization Matrix")
ax.legend(loc="lower right", fontsize=9)

for i, total in enumerate(matrix_plot["total"]):
    ax.text(total + 0.3, i, f"{total:.0f}", va="center", fontweight="bold")

fig.tight_layout()
st.pyplot(fig)
plt.close()

# Top recommendations
st.subheader("Top Recommendations")

top3_matrix = matrix_sorted.head(3)
cols = st.columns(3)
medals = ["1)", "2)", "3)"]

for i, (_, row) in enumerate(top3_matrix.iterrows()):
    with cols[i]:
        st.markdown(f"### {medals[i]} {row['opportunity']}")
        st.metric("Total Score", f"{row['total']:.0f}/25")
        st.markdown(f"""
        - BTC Decorrelation: **{row['btc_decorrelation']:.0f}**/5
        - Market Size: **{row['market_size']:.0f}**/5
        - Competitive Gap: **{row['competitive_gap']:.0f}**/5
        - Customer Demand: **{row['customer_demand']:.0f}**/5
        - Feasibility: **{row['feasibility']:.0f}**/5
        """)

# Weighted sensitivity analysis
st.subheader("How Strategy Changes the Answer")
st.markdown("What happens when we prioritize different goals? Each scheme doubles one dimension's weight.")

weight_schemes = {
    "Equal": {"btc_decorrelation": 1, "market_size": 1, "competitive_gap": 1, "customer_demand": 1, "feasibility": 1},
    "Diversification (2x BTC)": {"btc_decorrelation": 2, "market_size": 1, "competitive_gap": 1, "customer_demand": 1, "feasibility": 1},
    "Growth (2x Market)": {"btc_decorrelation": 1, "market_size": 2, "competitive_gap": 1, "customer_demand": 1, "feasibility": 1},
    "User-Driven (2x Demand)": {"btc_decorrelation": 1, "market_size": 1, "competitive_gap": 1, "customer_demand": 2, "feasibility": 1},
    "Quick Wins (2x Feasibility)": {"btc_decorrelation": 1, "market_size": 1, "competitive_gap": 1, "customer_demand": 1, "feasibility": 2},
}

selected_scheme = st.radio(
    "Choose a strategic lens:",
    list(weight_schemes.keys()),
    horizontal=True
)

weights = weight_schemes[selected_scheme]
matrix_weighted = matrix.copy()
matrix_weighted["weighted_total"] = sum(
    matrix_weighted[dim] * weights[dim] for dim in dims
)
max_possible = sum(5 * w for w in weights.values())

matrix_weighted = matrix_weighted.sort_values("weighted_total", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

colors_bar = []
for _, row in matrix_weighted.iterrows():
    if row["weighted_total"] >= matrix_weighted["weighted_total"].quantile(0.7):
        colors_bar.append("#2ecc71")
    elif row["weighted_total"] >= matrix_weighted["weighted_total"].quantile(0.4):
        colors_bar.append("#f39c12")
    else:
        colors_bar.append("#e74c3c")

ax.barh(range(len(matrix_weighted)), matrix_weighted["weighted_total"], color=colors_bar)
ax.set_yticks(range(len(matrix_weighted)))
ax.set_yticklabels(matrix_weighted["opportunity"])
ax.set_xlabel(f"Weighted Score (max {max_possible})")
ax.set_title(f"Rankings Under: {selected_scheme}")

for i, total in enumerate(matrix_weighted["weighted_total"]):
    ax.text(total + 0.2, i, f"{total:.0f}", va="center", fontweight="bold")

fig.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")


# ============================================================
# SECTION 10: STRATEGIC RECOMMENDATIONS
# ============================================================
st.markdown('<a id="recommendations"></a>', unsafe_allow_html=True)
st.header("10. Strategic Recommendations")
st.markdown("""
> There appears to be a two track strategy.
""")

# Track 1
st.subheader("Track 1: Reduce Bitcoin Dependency (Underinvested)")
st.markdown("These opportunities generate revenue that doesn't rise and fall with crypto markets.")

track1 = {
    "Banking / Cash Management": {
        "why": "Add direct deposit, bill pay, and cash management so users engage with Coinbase daily, even when crypto is quiet. Currently, users close the app during downturns. Banking keeps them.",
        "how": "Partner with a chartered bank (as Robinhood does with Sutton Bank). Coinbase builds the user experience; the partner handles regulated banking infrastructure. USDC rewards at 4.1% already functions like a savings account. Direct deposit and bill pay can complete the picture.",
        "evidence": "Interest income has a 0.117 BTC correlation (lowest of any revenue stream). Reddit demand: 11,813 upvotes. Competitive gap: Robinhood and CashApp both offer banking; Coinbase doesn't.",
        "risk": "Heavy regulatory burden. Users may not trust a crypto company with their paycheck. Competing against established banks and fintechs.",
        "score": 18
    },
    "Lending / Borrowing": {
        "why": "Let users earn interest by lending USDC, and borrow cash against their crypto without selling. Generates steady interest income regardless of BTC price.",
        "how": "Expand the USDC lending product launched in late 2025. Offer overcollateralized loans only. Auto-liquidate collateral if it drops. This is the opposite of what Celsius did (risky, uncollateralized loans).",
        "evidence": "Interest income has 0.117 BTC correlation. Lending market hit $73.6B in outstanding loans (Q3 2025). Major competitors (Celsius, BlockFi, Genesis) collapsed in 2022, clearing the field.",
        "risk": "Crypto lending has a terrible reputation after 2022. Regulators watching closely. Must move conservatively to maintain trust.",
        "score": 17
    },
}

for name, details in track1.items():
    with st.expander(f"{name} — Score: {details['score']}/25"):
        st.markdown(f"Why: {details['why']}")
        st.markdown(f"How: {details['how']}")
        st.markdown(f"Data Support: {details['evidence']}")
        st.markdown(f"Key Risk: {details['risk']}")

# Track 2
st.subheader("Track 2: Strengthen the Core (Already Underway)")
st.markdown("This investment makes Coinbase's existing business more competitive — but does **not** reduce Bitcoin dependency.")

track2 = {
    "Advanced Trading / Derivatives": {
        "why": "Become the one-stop shop for every type of crypto trading (spot, futures, options, perps). Serious traders currently use 3-4 platforms. Consolidating them onto Coinbase increases fee revenue.",
        "how": "Integrate the $2.9B Deribit acquisition into the main Coinbase platform. Build simplified derivative products for retail users — e.g., 'protect my portfolio' (a put option in plain language).",
        "evidence": "Highest Reddit demand at 16,860 upvotes. Derivatives market $6B+ growing at 20% CAGR. Coinbase already committed via Deribit acquisition.",
        "risk": "Derivatives volume is highly correlated with BTC. This deepens the core business rather than diversifying it. Important to pursue alongside and not instead of Track 1.",
        "score": 18
    }
}

for name, details in track2.items():
    with st.expander(f"{name} — Score: {details['score']}/25"):
        st.markdown(f"Why: {details['why']}")
        st.markdown(f"How: {details['how']}")
        st.markdown(f"Data Support: {details['evidence']}")
        st.markdown(f"Key Risk: {details['risk']}")

st.markdown("---")


# ============================================================
# SECTION 11: THE BOTTOM LINE
# ============================================================
st.markdown('<a id="bottom-line"></a>', unsafe_allow_html=True)
st.header("11. The Bottom Line")

st.info("""
Coinbase is already executing on Track 2: the Deribit acquisition, stock trading, and prediction markets 
all strengthen the core crypto business. 

My data suggests Track 1 is where the bigger strategic gap exists. Banking and lending 
would generate revenue streams with low BTC correlation, address clear competitive gaps, and meet demonstrated 
user demand. These investments would make Coinbase resilient in the next crypto downturn, and not just dominant 
in the next bull run.
""")

st.markdown("""
### How We Got Here

| Dimension | Method | Key Finding |
|-----------|--------|-------------|
| BTC Dependency | Stock correlation, 20 quarters of earnings | Coinbase has 0.931 BTC correlation (2020-2022) and 0.728 (2023-2025) |
| Structural Shift | Split-period regression on earnings data | Users don't come back after crypto crashes |
| Market Size | Industry research reports (10 markets sized) | Staking (245B) and stablecoins (170B) are the largest opportunities |
| Competitive Gaps | Feature mapping across 5 platforms, 30 products | Banking is the biggest gap. Robinhood and CashApp have it, Coinbase doesn't |
| Customer Demand | NLP analysis of ~500 Reddit posts | Users want DeFi, banking, and lower fees |
| BTC Correlations | Quarterly earnings vs BTC price regression | Interest income (0.117) is least BTC-dependent |
| Feasibility | Qualitative assessment of capabilities and regulation | Banking is hardest (2/5) but most valuable for diversification |
""")

st.markdown("---")
st.caption("Built by Hiram Lannes· Data Sources: Coinbase 10-Q filings, Yahoo Finance, Reddit, industry research reports")
