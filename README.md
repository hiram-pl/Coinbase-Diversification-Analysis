# Decoupling from Bitcoin: A Coinbase Diversification Strategy

Coinbase is the largest US crypto exchange with 110+ million verified users — but its revenue rises and falls with Bitcoin. This project quantifies that dependency and builds a data-driven framework for where Coinbase should diversify.


## Key Findings

- **Coinbase's Google Trends correlation with BTC is 0.78** — roughly 2x higher than Robinhood (0.38) or Binance (0.47). When Bitcoin drops, people stop thinking about Coinbase.
- **The problem is structural, not cyclical.** Even when BTC recovers to prior highs, retail trading revenue doesn't come back. Each crypto cycle brings fewer traders.
- **Trading fees still dominate revenue** despite 20+ product launches. The challenge isn't building products — it's shifting users to them.
- **Interest income (0.117 BTC correlation) is the most resilient revenue stream.** Products that generate recurring, service-based income are the path to diversification.
- **Banking/Cash Management is the biggest competitive gap** — Robinhood and CashApp both offer it; Coinbase doesn't.

## Two-Track Recommendation

| Track | Goal | Top Picks | Status |
|-------|------|-----------|--------|
| **Track 1** | Reduce BTC dependency | Banking, Lending | Underinvested |
| **Track 2** | Strengthen core | Derivatives | Already underway (Deribit acquisition) |

**The punchline:** Coinbase is already executing Track 2. Our data suggests Track 1 is where the bigger strategic gap exists.

## Methodology

Ten diversification opportunities scored across five dimensions:

| Dimension | Source | Method |
|-----------|--------|--------|
| BTC Decorrelation | Coinbase 10-Q filings (20 quarters) | Revenue stream correlation with quarterly BTC price |
| Market Size | Industry reports (Grand View, Galaxy, Statista) | Log-scaled normalization ($0.5B–$245B range) |
| Competitive Gap | Feature mapping across 5 platforms | Count of competitors offering what Coinbase doesn't |
| Customer Demand | ~500 Reddit posts from r/CoinBase | NLP categorization, upvote-weighted scoring |
| Feasibility | Qualitative assessment | Regulatory barriers, existing infrastructure, recent moves |

Weighted sensitivity analysis tests how rankings change under different strategic priorities (diversification-first, growth-first, user-driven, quick wins).

## Interactive Dashboard

The Streamlit dashboard provides an interactive exploration of all findings with 11 narrative sections.

**To run locally:**

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Project Structure

```
├── dashboard.py              # Streamlit dashboard (main deliverable)
├── prioritization.py         # Scoring engine — generates the prioritization matrix
├── requirements.txt
├── data/
│   ├── coinbase_earnings.csv     # 20 quarters of revenue by stream (Coinbase 10-Q)
│   ├── btc_data.csv              # Daily BTC prices (Yahoo Finance)
│   ├── coin_data.csv             # Daily COIN stock prices (Yahoo Finance)
│   ├── trends_broad_data.csv     # Google Trends: Coinbase vs Robinhood vs Binance
│   ├── reddit_demand_scores.csv  # Categorized Reddit posts with upvote totals
│   ├── market_sizing.csv         # Market size estimates for 10 opportunities
│   ├── competitor_features.csv   # 30+ features across 5 platforms (Y/N/P)
│   └── prioritization_matrix.csv # Final scores (output of prioritization.py)
```

## Data Sources

- **Coinbase earnings**: 10-Q/10-K filings (SEC EDGAR), Q4 2020 – Q3 2025
- **BTC / COIN prices**: Yahoo Finance via `yfinance`
- **Google Trends**: Trends API, weekly search interest 2020–2025
- **Reddit**: ~500 posts from r/CoinBase, categorized via NLP
- **Market sizing**: Grand View Research, Galaxy Research, Statista, Everstake, 360iResearch
- **Competitor features**: Product pages and documentation for Coinbase, Robinhood, Binance, Kraken, CashApp
# Coinbase-Diversification-Analysis
