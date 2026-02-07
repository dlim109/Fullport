"""
Portfolio Analytics Module
Handles portfolio composition, sector analysis, and diversification logic.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from data_handler import get_gics_classification


class Portfolio:
    """Portfolio analytics and composition management."""
    
    def __init__(self, holdings: Dict[str, float], current_prices: Dict[str, float]):
        """
        Initialize portfolio.
        
        Args:
            holdings: Dict mapping ticker to shares held
            current_prices: Dict mapping ticker to current price
        """
        self.holdings = holdings
        self.prices = current_prices
        self.tickers = list(holdings.keys())
        
    def get_total_value(self) -> float:
        """Calculate total portfolio market value."""
        total = 0
        for ticker, shares in self.holdings.items():
            if ticker in self.prices and not np.isnan(self.prices[ticker]):
                total += shares * self.prices[ticker]
        return total
    
    def get_position_values(self) -> Dict[str, float]:
        """Get market value for each position."""
        values = {}
        for ticker, shares in self.holdings.items():
            if ticker in self.prices and not np.isnan(self.prices[ticker]):
                values[ticker] = shares * self.prices[ticker]
            else:
                values[ticker] = 0
        return values
    
    def get_weights(self) -> Dict[str, float]:
        """Get portfolio weights for each position."""
        total_value = self.get_total_value()
        position_values = self.get_position_values()
        return {ticker: value / total_value for ticker, value in position_values.items()}
    
    def get_sector_allocation(self) -> pd.DataFrame:
        """
        Calculate sector-level allocation with GICS classification.
        
        Returns:
            DataFrame with columns: Sector, Value, Weight, Tickers
        """
        position_values = self.get_position_values()
        total_value = self.get_total_value()
        
        sector_data = {}
        for ticker, value in position_values.items():
            sector, _ = get_gics_classification(ticker)
            if sector not in sector_data:
                sector_data[sector] = {'value': 0, 'tickers': []}
            sector_data[sector]['value'] += value
            sector_data[sector]['tickers'].append(ticker)
        
        rows = []
        for sector, data in sector_data.items():
            rows.append({
                'Sector': sector,
                'Value': data['value'],
                'Weight': data['value'] / total_value,
                'Tickers': ', '.join(sorted(data['tickers']))
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('Weight', ascending=False).reset_index(drop=True)
    
    def get_industry_allocation(self) -> pd.DataFrame:
        """
        Calculate industry group-level allocation with GICS classification.
        
        Returns:
            DataFrame with columns: Sector, Industry, Value, Weight, Tickers
        """
        position_values = self.get_position_values()
        total_value = self.get_total_value()
        
        industry_data = {}
        for ticker, value in position_values.items():
            sector, industry = get_gics_classification(ticker)
            key = (sector, industry)
            if key not in industry_data:
                industry_data[key] = {'value': 0, 'tickers': []}
            industry_data[key]['value'] += value
            industry_data[key]['tickers'].append(ticker)
        
        rows = []
        for (sector, industry), data in industry_data.items():
            rows.append({
                'Sector': sector,
                'Industry': industry,
                'Value': data['value'],
                'Weight': data['value'] / total_value,
                'Tickers': ', '.join(sorted(data['tickers']))
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values(['Sector', 'Weight'], ascending=[True, False]).reset_index(drop=True)
    
    def identify_overweight_sectors(self, threshold: float = 0.20) -> List[str]:
        """
        Identify sectors exceeding weight threshold.
        
        Args:
            threshold: Weight threshold (default 20%)
        
        Returns:
            List of overweight sector names
        """
        sector_alloc = self.get_sector_allocation()
        overweight = sector_alloc[sector_alloc['Weight'] > threshold]['Sector'].tolist()
        return overweight
    

def calculate_portfolio_correlation(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Calculate correlation of each asset to the weighted portfolio.
    
    Args:
        returns: DataFrame of asset returns
        weights: Dict of portfolio weights
    
    Returns:
        Series of correlations for each asset
    """
    # Calculate portfolio returns
    weight_array = np.array([weights.get(col, 0) for col in returns.columns])
    portfolio_returns = (returns * weight_array).sum(axis=1)
    
    # Correlation of each asset with portfolio
    correlations = returns.corrwith(portfolio_returns)
    return correlations


def recommend_diversification_sector(
    current_sectors: List[str],
    returns: pd.DataFrame,
    weights: Dict[str, float],
    overweight_sector: str
) -> Tuple[str, float, List[str]]:
    """
    Recommend a sector for diversification based on lowest correlation to portfolio.
    
    Args:
        current_sectors: List of sectors currently in portfolio
        returns: Historical returns data
        weights: Portfolio weights
        overweight_sector: The sector that is overweight
    
    Returns:
        Tuple of (recommended_sector, correlation, example_tickers)
    """
    # All GICS sectors
    all_sectors = [
        'Information Technology',
        'Health Care',
        'Financials',
        'Consumer Discretionary',
        'Communication Services',
        'Industrials',
        'Consumer Staples',
        'Energy',
        'Utilities',
        'Real Estate',
        'Materials'
    ]
    
    # Calculate average correlation by sector
    portfolio_corr = calculate_portfolio_correlation(returns, weights)
    
    sector_correlations = {}
    for sector in all_sectors:
        if sector == overweight_sector or sector == 'Unknown':
            continue
            
        # Get tickers in this sector
        sector_tickers = [ticker for ticker in returns.columns 
                         if get_gics_classification(ticker)[0] == sector]
        
        if sector_tickers:
            avg_corr = portfolio_corr[sector_tickers].mean()
            sector_correlations[sector] = avg_corr
    
    # Find sector with lowest correlation
    if sector_correlations:
        recommended_sector = min(sector_correlations, key=sector_correlations.get)
        correlation = sector_correlations[recommended_sector]
    else:
        # Fallback: recommend sector not in portfolio
        underrepresented = [s for s in all_sectors if s not in current_sectors and s != 'Unknown']
        recommended_sector = underrepresented[0] if underrepresented else 'Utilities'
        correlation = 0.3
    
    # Get example tickers from recommended sector
    example_tickers = get_example_tickers_for_sector(recommended_sector)
    
    return recommended_sector, correlation, example_tickers


def get_example_tickers_for_sector(sector: str) -> List[str]:
    """
    Get 3-5 example liquid tickers for a given GICS sector.
    """
    from data_handler import GICS_MAPPING
    
    sector_tickers = [ticker for ticker, (sec, _) in GICS_MAPPING.items() if sec == sector]
    
    # Prioritize liquid, well-known names
    priority_map = {
        'Information Technology': ['MSFT', 'AAPL', 'NVDA', 'AVGO'],
        'Health Care': ['UNH', 'JNJ', 'LLY', 'ABBV'],
        'Financials': ['JPM', 'V', 'MA', 'BAC'],
        'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD'],
        'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS'],
        'Industrials': ['CAT', 'BA', 'HON', 'UPS'],
        'Consumer Staples': ['WMT', 'PG', 'COST', 'KO'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG'],
        'Utilities': ['NEE', 'DUK', 'SO', 'D'],
        'Real Estate': ['AMT', 'PLD', 'EQIX', 'CCI'],
        'Materials': ['LIN', 'APD', 'NEM', 'FCX']
    }
    
    if sector in priority_map:
        return priority_map[sector][:4]
    elif sector_tickers:
        return sector_tickers[:4]
    else:
        return []


def propose_rebalanced_weights(
    current_sector_weights: Dict[str, float],
    overweight_sector: str,
    recommended_sector: str,
    target_max_weight: float = 0.20
) -> Dict[str, float]:
    """
    Propose target sector weights after rebalancing.
    
    Args:
        current_sector_weights: Current sector allocations
        overweight_sector: Sector to reduce
        recommended_sector: Sector to increase
        target_max_weight: Target maximum weight for any sector
    
    Returns:
        Dictionary of proposed sector weights
    """
    proposed = current_sector_weights.copy()
    
    # Calculate excess weight to redistribute
    if overweight_sector in proposed:
        excess = proposed[overweight_sector] - target_max_weight
        proposed[overweight_sector] = target_max_weight
        
        # Allocate to recommended sector
        if recommended_sector in proposed:
            proposed[recommended_sector] += excess
        else:
            proposed[recommended_sector] = excess
    
    return proposed
