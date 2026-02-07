"""
Portfolio Analysis Dashboard
Main Streamlit application for interactive portfolio analytics.

Author: Senior Quantitative Developer
Purpose: Retail investor portfolio analysis, risk assessment, and stress testing
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# Import custom modules
from data_handler import (
    fetch_price_data, fetch_current_prices, calculate_returns,
    fetch_spy_data, get_risk_free_rate, get_gics_classification
)
from portfolio_analytics import (
    Portfolio, recommend_diversification_sector, propose_rebalanced_weights
)
from risk_metrics import (
    calculate_portfolio_returns, RiskMetrics
)
from options_pricing import PortfolioHedge
from monte_carlo import MonteCarloSimulator, StressScenario
from visualization import (
    plot_sector_allocation, plot_crash_scenario,
    plot_monte_carlo_paths, plot_return_distribution,
    plot_hedged_vs_unhedged_monte_carlo
)

# Page configuration
st.set_page_config(
    page_title="Fullport",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# ===========================
# Sidebar: Portfolio Input
# ===========================

st.sidebar.title("📊 Portfolio Input")
st.sidebar.markdown("---")

input_method = st.sidebar.radio(
    "Input Method",
    ["Manual Entry", "CSV Upload"]
)

holdings = {}

if input_method == "Manual Entry":
    st.sidebar.subheader("Enter Holdings")
    
    n_positions = st.sidebar.number_input("Number of Positions", min_value=1, max_value=50, value=5)
    
    for i in range(int(n_positions)):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ticker = st.text_input(f"Ticker {i+1}", value="", key=f"ticker_{i}")
        with col2:
            shares = st.number_input(f"Shares", min_value=0.0, value=0.0, key=f"shares_{i}")
        
        if ticker and shares > 0:
            holdings[ticker.upper()] = shares

else:  # CSV Upload
    st.sidebar.subheader("Upload Portfolio CSV")
    st.sidebar.markdown("CSV format: `ticker,shares`")
    
    uploaded_file = st.sidebar.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            for _, row in df.iterrows():
                ticker = str(row['ticker']).upper().strip()
                shares = float(row['shares'])
                if shares > 0:
                    holdings[ticker] = shares
            st.sidebar.success(f"Loaded {len(holdings)} positions")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")

# Sample portfolio button
if st.sidebar.button("Load Sample Portfolio"):
    holdings = {
        'AAPL': 50,
        'MSFT': 40,
        'GOOGL': 30,
        'NVDA': 25,
        'META': 20,
        'AMZN': 15,
        'TSLA': 10,
        'JPM': 30,
        'V': 25
    }
    st.sidebar.success("Sample portfolio loaded!")

st.sidebar.markdown("---")

# ===========================
# Main Application
# ===========================

st.title("🎯 Fullport")
st.markdown("### Investment Portfolio Analysis Dashboard")
st.markdown("---")

if len(holdings) == 0:
    st.info("👈 Please enter your portfolio holdings in the sidebar to begin analysis.")
    st.stop()

# ===========================
# Data Loading
# ===========================

with st.spinner("Fetching market data..."):
    try:
        tickers = list(holdings.keys())
        
        # Fetch historical data
        price_data = fetch_price_data(tickers, period='2y')
        returns_data = calculate_returns(price_data)
        
        # Fetch current prices
        current_prices = fetch_current_prices(tickers)
        
        # Fetch S&P 500 benchmark
        spy_prices, spy_returns = fetch_spy_data(period='2y')
        
        # Risk-free rate
        rf_rate = get_risk_free_rate()
        
        # Initialize portfolio
        portfolio = Portfolio(holdings, current_prices)
        total_value = portfolio.get_total_value()
        weights = portfolio.get_weights()
        
        # Calculate portfolio returns
        portfolio_returns = calculate_portfolio_returns(returns_data, weights)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# ===========================
# Section 1: Portfolio Composition & Sector Analysis
# ===========================

st.header("1️⃣ Portfolio Composition & Sector Analysis")

# Display total value prominently
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Portfolio Value", f"${total_value:,.2f}")
with col2:
    st.metric("Number of Positions", len(holdings))
with col3:
    avg_position = total_value / len(holdings)
    st.metric("Average Position Size", f"${avg_position:,.2f}")

st.markdown("---")

# Sector allocation
sector_df = portfolio.get_sector_allocation()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Sector Allocation")
    fig_sector = plot_sector_allocation(sector_df)
    st.plotly_chart(fig_sector, use_container_width=True)

with col2:
    st.subheader("Sector Breakdown")
    display_df = sector_df.copy()
    display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.2f}")
    display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True, height=400)

# Industry group breakdown
st.subheader("Industry Group Breakdown")
industry_df = portfolio.get_industry_allocation()
display_industry = industry_df.copy()
display_industry['Value'] = display_industry['Value'].apply(lambda x: f"${x:,.2f}")
display_industry['Weight'] = display_industry['Weight'].apply(lambda x: f"{x:.2%}")
st.dataframe(display_industry, use_container_width=True)

# ===========================
# Section 2: Concentration, Diversification & Risk Metrics
# ===========================

st.markdown("---")
st.header("2️⃣ Sector Concentration, Diversification & Risk Metrics")

# Check for overweight sectors
overweight_sectors = portfolio.identify_overweight_sectors(threshold=0.20)

if overweight_sectors:
    st.warning(f"⚠️ **Overweight Sectors Detected:** {', '.join(overweight_sectors)}")
    st.markdown("Sectors exceeding 20% allocation pose concentration risk.")
    
    # Recommend diversification
    overweight_sector = overweight_sectors[0]
    current_sectors = sector_df['Sector'].tolist()
    
    recommended_sector, correlation, example_tickers = recommend_diversification_sector(
        current_sectors, returns_data, weights, overweight_sector
    )
    
    st.subheader("📈 Diversification Recommendation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Recommended Sector:** {recommended_sector}
        
        **Reason:** This sector has the lowest correlation ({correlation:.3f}) to your current portfolio, 
        providing maximum diversification benefit.
        
        **Example Tickers:**
        """)
        for ticker in example_tickers:
            sector, industry = get_gics_classification(ticker)
            st.markdown(f"- `{ticker}`: {industry}")
    
    with col2:
        st.markdown("**Proposed Rebalancing:**")
        current_weight = sector_df[sector_df['Sector'] == overweight_sector]['Weight'].iloc[0]
        
        sector_weights = dict(zip(sector_df['Sector'], sector_df['Weight']))
        rebalanced = propose_rebalanced_weights(sector_weights, overweight_sector, recommended_sector)
        
        rebal_df = pd.DataFrame([
            {'Sector': sector, 
             'Current': f"{sector_weights.get(sector, 0):.2%}",
             'Target': f"{weight:.2%}"}
            for sector, weight in rebalanced.items()
        ])
        st.dataframe(rebal_df, use_container_width=True)
else:
    st.success("✅ No sector concentration issues detected. Portfolio is well-diversified.")

# Risk Metrics
st.subheader("📊 Portfolio Risk Metrics")

risk_metrics = RiskMetrics(portfolio_returns, spy_returns, weights)
metrics_df = risk_metrics.to_dataframe()

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(metrics_df, use_container_width=True)

with col2:
    st.markdown("""
    **Metric Definitions:**
    - **Annualized Volatility:** Standard deviation of returns (252 trading days)
    - **Maximum Drawdown:** Largest peak-to-trough decline
    - **Portfolio Beta:** Sensitivity to S&P 500 movements (1.0 = market)
    - **VaR (Value-at-Risk):** Potential loss at given confidence level
    - **CVaR (Conditional VaR):** Expected loss beyond VaR threshold
    
    **Key Assumptions:**
    - Returns are log-normally distributed
    - Historical volatility is stationary
    - Risk-free rate: {:.2%}
    """.format(rf_rate))

# ===========================
# Section 3: Black Swan Stress Test & Hedging
# ===========================

st.markdown("---")
st.header("3️⃣ Black Swan Stress Test, Hedging & Interactive Simulation")

st.markdown("""
This section simulates a severe market crash and demonstrates portfolio-level hedging strategies.
Adjust parameters to explore different scenarios.
""")

col1, col2, col3 = st.columns(3)

with col1:
    crash_magnitude = st.slider(
        "Market Crash Magnitude",
        min_value=-50,
        max_value=-10,
        value=-30,
        step=5,
        format="%d%%"
    ) / 100

with col2:
    hedge_ratio = st.slider(
        "Portfolio Hedge Ratio",
        min_value=0,
        max_value=100,
        value=50,
        step=10,
        format="%d%%"
    ) / 100

with col3:
    hedge_type = st.selectbox(
        "Hedge Strategy",
        ["SPY Put Options", "Inverse ETF (SH)"]
    )

# Get current SPY price
spy_current = spy_prices.iloc[-1]

# Initialize hedge calculator
hedge_calc = PortfolioHedge(
    portfolio_value=total_value,
    portfolio_beta=risk_metrics.beta,
    spy_price=spy_current,
    risk_free_rate=rf_rate
)

# Calculate hedge parameters
if hedge_type == "SPY Put Options":
    hedge_params = hedge_calc.calculate_put_hedge(
        hedge_ratio=hedge_ratio,
        moneyness=0.95,  # 5% OTM
        days_to_expiry=30,
        implied_vol=0.20  # 20% IV assumption
    )
else:
    hedge_params = hedge_calc.calculate_inverse_etf_hedge(hedge_ratio)

# Display hedge details
st.subheader("🛡️ Hedge Specification")

col1, col2 = st.columns(2)

with col1:
    if hedge_type == "SPY Put Options":
        st.markdown(f"""
        **Strategy:** {hedge_params['strategy']}
        
        **Underlying:** {hedge_params['underlying']}
        
        **Strike Price:** ${hedge_params['strike']:.2f}
        
        **Current Price:** ${hedge_params['current_price']:.2f}
        
        **Moneyness:** {hedge_params['moneyness']:.1%}
        
        **Expiry:** {hedge_params['expiry_days']} days
        
        **Implied Volatility:** {hedge_params['implied_vol']:.1%}
        """)
    else:
        st.markdown(f"""
        **Strategy:** {hedge_params['strategy']}
        
        **Ticker:** {hedge_params['ticker']}
        
        **Description:** {hedge_params['description']}
        
        **Multiplier:** {hedge_params['multiplier']}x
        
        **Shares Needed:** {hedge_params['shares']:.0f}
        
        **Estimated Price:** ${hedge_params['estimated_price']:.2f}
        """)

with col2:
    if hedge_type == "SPY Put Options":
        st.markdown(f"""
        **Put Price:** ${hedge_params['put_price']:.2f}
        
        **Contracts Needed:** {hedge_params['contracts']:.2f}
        
        **Total Premium:** ${hedge_params['total_premium']:,.2f}
        
        **Premium % of Portfolio:** {hedge_params['premium_pct']:.2%}
        
        **Put Delta:** {hedge_params['delta']:.3f}
        
        **Effective Hedge Ratio:** {hedge_params['effective_hedge_ratio']:.2%}
        """)
    else:
        st.markdown(f"""
        **Position Value:** ${hedge_params['position_value']:,.2f}
        
        **Cost % of Portfolio:** {hedge_params['cost_pct']:.2%}
        
        **Effective Hedge Ratio:** {hedge_params['effective_hedge_ratio']:.2%}
        
        ---
        
        *Inverse ETFs provide inverse exposure to S&P 500. 
        When market falls, inverse ETF gains.*
        """)

# Simulate crash scenario
st.subheader("📉 Crash Scenario Simulation")

# Use historical volatility to create realistic crash path
crash_days = 60
returns_sample = portfolio_returns.iloc[-crash_days:].values

unhedged_values, hedged_values = hedge_calc.simulate_crash_with_hedge(
    portfolio_returns=returns_sample,
    crash_magnitude=crash_magnitude,
    hedge_type='put' if hedge_type == "SPY Put Options" else 'inverse_etf',
    hedge_params=hedge_params
)

# Plot crash scenario
fig_crash = plot_crash_scenario(
    unhedged_values,
    hedged_values,
    crash_magnitude,
    hedge_type
)
st.plotly_chart(fig_crash, use_container_width=True)

# Crash statistics
col1, col2, col3 = st.columns(3)

final_unhedged = unhedged_values[-1]
final_hedged = hedged_values[-1]
unhedged_loss = (final_unhedged - total_value) / total_value
hedged_loss = (final_hedged - total_value) / total_value
protection = unhedged_loss - hedged_loss

with col1:
    st.metric(
        "Unhedged Final Value",
        f"${final_unhedged:,.2f}",
        f"{unhedged_loss:.2%}"
    )

with col2:
    st.metric(
        "Hedged Final Value",
        f"${final_hedged:,.2f}",
        f"{hedged_loss:.2%}"
    )

with col3:
    st.metric(
        "Hedge Protection",
        f"{protection:.2%}",
        "Reduced Loss"
    )

st.markdown("""
**Black-Scholes Model Assumptions:**
- European-style options (exercise at expiry only)
- Constant risk-free rate and volatility
- No dividends, transaction costs, or liquidity constraints
- Lognormal asset price distribution

**Limitations:**
- Actual hedge performance depends on realized volatility vs implied volatility
- Options may be illiquid during extreme market stress
- Inverse ETFs have tracking error and daily rebalancing effects
""")

# ===========================
# Section 4: Monte Carlo Stress Testing
# ===========================

st.markdown("---")
st.header("4️⃣ Monte Carlo Stress Testing")

st.markdown("""
Parametric stress testing using Geometric Brownian Motion (GBM) framework.
Simulations calibrated from historical data with elevated volatility to model crisis conditions.
""")

# Monte Carlo parameters
col1, col2, col3, col4 = st.columns(4)

with col1:
    mc_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

with col2:
    mc_days = st.number_input("Simulation Horizon (days)", min_value=30, max_value=252, value=60, step=30)

with col3:
    stress_multiplier = st.slider("Volatility Stress Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

with col4:
    mc_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)

# Run Monte Carlo simulation
if st.button("🎲 Run Monte Carlo Simulation", type="primary"):
    with st.spinner("Running simulations..."):
        
        # Initialize simulator
        mc_sim = MonteCarloSimulator(portfolio_returns)
        
        # Run hedged vs unhedged simulations
        unhedged_paths, hedged_paths = mc_sim.simulate_with_hedge(
            initial_value=total_value,
            n_days=mc_days,
            n_simulations=int(mc_simulations),
            hedge_params=hedge_params,
            stress_multiplier=stress_multiplier,
            seed=mc_seed
        )
        
        # Store in session state
        st.session_state['mc_unhedged'] = unhedged_paths
        st.session_state['mc_hedged'] = hedged_paths
        st.session_state['mc_initial'] = total_value

if 'mc_unhedged' in st.session_state:
    
    mc_sim = MonteCarloSimulator(portfolio_returns)
    unhedged_paths = st.session_state['mc_unhedged']
    hedged_paths = st.session_state['mc_hedged']
    initial_value = st.session_state['mc_initial']
    
    # Plot hedged vs unhedged
    st.subheader("Monte Carlo Results: Hedged vs Unhedged")
    fig_mc_compare = plot_hedged_vs_unhedged_monte_carlo(unhedged_paths, hedged_paths, initial_value)
    st.plotly_chart(fig_mc_compare, use_container_width=True)
    
    # Statistics comparison
    st.subheader("Simulation Statistics")
    
    unhedged_stats = mc_sim.calculate_statistics(unhedged_paths, initial_value)
    hedged_stats = mc_sim.calculate_statistics(hedged_paths, initial_value)
    
    comparison_df = pd.DataFrame({
        'Metric': [
            'Mean Final Value',
            'Median Final Value',
            'Std Dev',
            '5th Percentile',
            '95th Percentile',
            'Mean Return',
            'Probability of Loss',
            'Avg Max Drawdown'
        ],
        'Unhedged': [
            f"${unhedged_stats['mean_final_value']:,.2f}",
            f"${unhedged_stats['median_final_value']:,.2f}",
            f"${unhedged_stats['std_final_value']:,.2f}",
            f"${unhedged_stats['percentile_5']:,.2f}",
            f"${unhedged_stats['percentile_95']:,.2f}",
            f"{unhedged_stats['mean_return']:.2%}",
            f"{unhedged_stats['prob_loss']:.2%}",
            f"{abs(unhedged_stats['max_drawdown']):.2%}"
        ],
        'Hedged': [
            f"${hedged_stats['mean_final_value']:,.2f}",
            f"${hedged_stats['median_final_value']:,.2f}",
            f"${hedged_stats['std_final_value']:,.2f}",
            f"${hedged_stats['percentile_5']:,.2f}",
            f"${hedged_stats['percentile_95']:,.2f}",
            f"{hedged_stats['mean_return']:.2%}",
            f"{hedged_stats['prob_loss']:.2%}",
            f"{abs(hedged_stats['max_drawdown']):.2%}"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Return distributions
    st.subheader("Return Distribution Comparison")
    
    unhedged_returns = (unhedged_paths[:, -1] - initial_value) / initial_value
    hedged_returns = (hedged_paths[:, -1] - initial_value) / initial_value
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_unhedged_dist = plot_return_distribution(unhedged_returns, "Unhedged Return Distribution")
        st.plotly_chart(fig_unhedged_dist, use_container_width=True)
    
    with col2:
        fig_hedged_dist = plot_return_distribution(hedged_returns, "Hedged Return Distribution")
        st.plotly_chart(fig_hedged_dist, use_container_width=True)
    
    st.markdown("""
    **GBM Model Assumptions:**
    - Returns follow lognormal distribution
    - Drift (μ) and volatility (σ) are constant (stationarity assumption)
    - Independent increments (no autocorrelation)
    - Continuous rebalancing
    
    **Calibration:**
    - Drift calibrated from historical mean return
    - Base volatility from historical standard deviation
    - Stressed volatility = base × stress multiplier
    
    **Limitations:**
    - Real markets exhibit volatility clustering and fat tails
    - Correlation structure may break down in crises
    - Liquidity constraints not modeled
    - Options delta changes with market moves (not static hedge)
    """)

# ===========================
# Footer
# ===========================

st.markdown("---")
st.markdown("""
**Disclaimer:** This tool is for educational and analytical purposes only. 
It does not constitute financial advice. All models involve assumptions and simplifications.
Actual market behavior may differ significantly from simulations. Consult a qualified financial advisor before making investment decisions.

**Data Source:** Yahoo Finance (yfinance)  
**Risk-Free Rate:** Current 13-week Treasury rate  
**Benchmark:** S&P 500 (SPY)  
""")
