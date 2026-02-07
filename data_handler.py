"""
Data Handler Module
Handles data fetching from yfinance and GICS sector classification.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# GICS Sector Classification (11 sectors)
# Mapping major US equities to their GICS sectors and industry groups
GICS_MAPPING = {
    # Technology
    'AAPL': ('Information Technology', 'Technology Hardware, Storage & Peripherals'),
    'MSFT': ('Information Technology', 'Software'),
    'GOOGL': ('Communication Services', 'Interactive Media & Services'),
    'GOOG': ('Communication Services', 'Interactive Media & Services'),
    'META': ('Communication Services', 'Interactive Media & Services'),
    'NVDA': ('Information Technology', 'Semiconductors & Semiconductor Equipment'),
    'AMD': ('Information Technology', 'Semiconductors & Semiconductor Equipment'),
    'INTC': ('Information Technology', 'Semiconductors & Semiconductor Equipment'),
    'CRM': ('Information Technology', 'Software'),
    'ORCL': ('Information Technology', 'Software'),
    'ADBE': ('Information Technology', 'Software'),
    'CSCO': ('Information Technology', 'Communications Equipment'),
    'AVGO': ('Information Technology', 'Semiconductors & Semiconductor Equipment'),
    'QCOM': ('Information Technology', 'Semiconductors & Semiconductor Equipment'),
    'TXN': ('Information Technology', 'Semiconductors & Semiconductor Equipment'),
    'IBM': ('Information Technology', 'IT Services'),
    'NOW': ('Information Technology', 'Software'),
    
    # Communication Services
    'NFLX': ('Communication Services', 'Entertainment'),
    'DIS': ('Communication Services', 'Entertainment'),
    'CMCSA': ('Communication Services', 'Media'),
    'T': ('Communication Services', 'Diversified Telecommunication Services'),
    'VZ': ('Communication Services', 'Diversified Telecommunication Services'),
    'TMUS': ('Communication Services', 'Wireless Telecommunication Services'),
    
    # Consumer Discretionary
    'AMZN': ('Consumer Discretionary', 'Broadline Retail'),
    'TSLA': ('Consumer Discretionary', 'Automobiles'),
    'HD': ('Consumer Discretionary', 'Specialty Retail'),
    'NKE': ('Consumer Discretionary', 'Textiles, Apparel & Luxury Goods'),
    'MCD': ('Consumer Discretionary', 'Hotels, Restaurants & Leisure'),
    'SBUX': ('Consumer Discretionary', 'Hotels, Restaurants & Leisure'),
    'TGT': ('Consumer Discretionary', 'Broadline Retail'),
    'LOW': ('Consumer Discretionary', 'Specialty Retail'),
    'BKNG': ('Consumer Discretionary', 'Hotels, Restaurants & Leisure'),
    'GM': ('Consumer Discretionary', 'Automobiles'),
    'F': ('Consumer Discretionary', 'Automobiles'),
    
    # Consumer Staples
    'WMT': ('Consumer Staples', 'Food & Staples Retailing'),
    'PG': ('Consumer Staples', 'Household Products'),
    'KO': ('Consumer Staples', 'Beverages'),
    'PEP': ('Consumer Staples', 'Beverages'),
    'COST': ('Consumer Staples', 'Food & Staples Retailing'),
    'PM': ('Consumer Staples', 'Tobacco'),
    'MO': ('Consumer Staples', 'Tobacco'),
    'CL': ('Consumer Staples', 'Household Products'),
    
    # Financials
    'JPM': ('Financials', 'Banks'),
    'BAC': ('Financials', 'Banks'),
    'WFC': ('Financials', 'Banks'),
    'C': ('Financials', 'Banks'),
    'GS': ('Financials', 'Capital Markets'),
    'MS': ('Financials', 'Capital Markets'),
    'BLK': ('Financials', 'Capital Markets'),
    'SCHW': ('Financials', 'Capital Markets'),
    'V': ('Financials', 'Financial Services'),
    'MA': ('Financials', 'Financial Services'),
    'AXP': ('Financials', 'Financial Services'),
    'BRK-B': ('Financials', 'Diversified Financial Services'),
    'BRK.B': ('Financials', 'Diversified Financial Services'),
    
    # Health Care
    'UNH': ('Health Care', 'Health Care Providers & Services'),
    'JNJ': ('Health Care', 'Pharmaceuticals'),
    'PFE': ('Health Care', 'Pharmaceuticals'),
    'ABBV': ('Health Care', 'Biotechnology'),
    'MRK': ('Health Care', 'Pharmaceuticals'),
    'TMO': ('Health Care', 'Life Sciences Tools & Services'),
    'ABT': ('Health Care', 'Health Care Equipment & Supplies'),
    'DHR': ('Health Care', 'Health Care Equipment & Supplies'),
    'LLY': ('Health Care', 'Pharmaceuticals'),
    'AMGN': ('Health Care', 'Biotechnology'),
    'GILD': ('Health Care', 'Biotechnology'),
    
    # Industrials
    'BA': ('Industrials', 'Aerospace & Defense'),
    'CAT': ('Industrials', 'Machinery'),
    'UPS': ('Industrials', 'Air Freight & Logistics'),
    'HON': ('Industrials', 'Industrial Conglomerates'),
    'GE': ('Industrials', 'Industrial Conglomerates'),
    'MMM': ('Industrials', 'Industrial Conglomerates'),
    'LMT': ('Industrials', 'Aerospace & Defense'),
    'RTX': ('Industrials', 'Aerospace & Defense'),
    'DE': ('Industrials', 'Machinery'),
    
    # Energy
    'XOM': ('Energy', 'Oil, Gas & Consumable Fuels'),
    'CVX': ('Energy', 'Oil, Gas & Consumable Fuels'),
    'COP': ('Energy', 'Oil, Gas & Consumable Fuels'),
    'SLB': ('Energy', 'Energy Equipment & Services'),
    'EOG': ('Energy', 'Oil, Gas & Consumable Fuels'),
    'MPC': ('Energy', 'Oil, Gas & Consumable Fuels'),
    'PSX': ('Energy', 'Oil, Gas & Consumable Fuels'),
    
    # Materials
    'LIN': ('Materials', 'Chemicals'),
    'APD': ('Materials', 'Chemicals'),
    'ECL': ('Materials', 'Chemicals'),
    'DD': ('Materials', 'Chemicals'),
    'NEM': ('Materials', 'Metals & Mining'),
    'FCX': ('Materials', 'Metals & Mining'),
    
    # Real Estate
    'AMT': ('Real Estate', 'Equity Real Estate Investment Trusts (REITs)'),
    'PLD': ('Real Estate', 'Equity Real Estate Investment Trusts (REITs)'),
    'CCI': ('Real Estate', 'Equity Real Estate Investment Trusts (REITs)'),
    'EQIX': ('Real Estate', 'Equity Real Estate Investment Trusts (REITs)'),
    'SPG': ('Real Estate', 'Equity Real Estate Investment Trusts (REITs)'),
    
    # Utilities
    'NEE': ('Utilities', 'Electric Utilities'),
    'DUK': ('Utilities', 'Electric Utilities'),
    'SO': ('Utilities', 'Electric Utilities'),
    'D': ('Utilities', 'Electric Utilities'),
    'AEP': ('Utilities', 'Electric Utilities'),
}


def get_gics_classification(ticker: str) -> Tuple[str, str]:
    """
    Get GICS sector and industry group for a ticker.
    Returns ('Unknown', 'Unknown') if not in mapping.
    """
    ticker_clean = ticker.upper().replace('-', '.')
    return GICS_MAPPING.get(ticker_clean, ('Unknown', 'Unknown'))


def fetch_price_data(tickers: List[str], period: str = '2y') -> pd.DataFrame:
    """
    Fetch historical price data for given tickers.
    
    Args:
        tickers: List of ticker symbols
        period: Historical period (default 2 years)
    
    Returns:
        DataFrame with close prices
    """
    if len(tickers) == 1:
        # Single ticker - yfinance returns simple DataFrame
        data = yf.download(tickers[0], period=period, progress=False)
        if data.empty:
            raise ValueError(f"No data retrieved for {tickers[0]}")
        return data[['Close']].rename(columns={'Close': tickers[0]}).ffill()
    else:
        # Multiple tickers - yfinance returns MultiIndex DataFrame
        data = yf.download(tickers, period=period, progress=False, group_by='column')
        if data.empty:
            raise ValueError(f"No data retrieved for tickers: {tickers}")
        
        # Extract Close column(s)
        if 'Close' in data.columns:
            # MultiIndex case: data has (column_name, ticker) structure
            close = data['Close']
        elif isinstance(data.columns, pd.MultiIndex):
            # Alternative MultiIndex structure
            close = data.xs('Close', level=0, axis=1)
        else:
            # Fallback: assume all columns are tickers
            close = data
        
        return close.ffill()


def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Fetch current/latest prices for tickers.
    
    Returns:
        Dictionary mapping ticker to current price
    """
    prices = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            if not hist.empty:
                prices[ticker] = hist['Close'].iloc[-1]
            else:
                prices[ticker] = np.nan
        except:
            prices[ticker] = np.nan
    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily log returns from price data.
    """
    return np.log(prices / prices.shift(1)).dropna()


def fetch_spy_data(period: str = '2y') -> Tuple[pd.Series, pd.Series]:
    """
    Fetch S&P 500 (SPY) price and return data for benchmark.
    """
    # Download data
    data = yf.download('SPY', period=period, progress=False)
    
    # Handle yfinance MultiIndex or single-column DataFrame return
    if 'Close' in data.columns:
        spy_data = data['Close']
    else:
        # Fallback if structure is unexpected
        spy_data = data.iloc[:, 0]
        
    # FIX: Ensure data is 1D Series (removes extra dimension)
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()
        
    spy_returns = np.log(spy_data / spy_data.shift(1)).dropna()
    return spy_data, spy_returns


def get_risk_free_rate() -> float:
    """
    Fetch current risk-free rate (13-week Treasury).
    Fallback to 4.5% if fetch fails.
    """
    try:
        tnx = yf.Ticker('^IRX')
        hist = tnx.history(period='5d')
        if not hist.empty:
            return hist['Close'].iloc[-1] / 100
    except:
        pass
    return 0.045  # Fallback rate
