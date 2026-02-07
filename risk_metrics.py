"""
Risk Metrics Module
Computes portfolio risk measures: volatility, beta, drawdown, VaR.
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy import stats


def calculate_portfolio_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Calculate portfolio returns from constituent returns and weights.
    
    Args:
        returns: DataFrame of asset returns
        weights: Dict mapping ticker to weight
    
    Returns:
        Series of portfolio returns
    """
    weight_array = np.array([weights.get(col, 0) for col in returns.columns])
    return (returns * weight_array).sum(axis=1)


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """
    Calculate annualized volatility from daily returns.
    
    Args:
        returns: Series of daily returns
        trading_days: Number of trading days per year
    
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(trading_days)


def maximum_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from return series.
    
    Args:
        returns: Series of returns
    
    Returns:
        Maximum drawdown (positive value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())


def portfolio_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate portfolio beta relative to market.
    
    Args:
        portfolio_returns: Series of portfolio returns
        market_returns: Series of market (benchmark) returns
    
    Returns:
        Portfolio beta
    """
    # Align the series
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 1.0
    
    covariance = aligned['portfolio'].cov(aligned['market'])
    market_variance = aligned['market'].var()
    
    return covariance / market_variance if market_variance != 0 else 1.0


def value_at_risk(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate historical Value-at-Risk (VaR).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (default 95%)
    
    Returns:
        VaR (positive value representing potential loss)
    """
    return abs(np.percentile(returns, (1 - confidence_level) * 100))


def parametric_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate parametric VaR assuming normal distribution.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level
    
    Returns:
        Parametric VaR
    """
    mu = returns.mean()
    sigma = returns.std()
    z_score = stats.norm.ppf(1 - confidence_level)
    return abs(mu + z_score * sigma)


def conditional_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value-at-Risk (Expected Shortfall).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level
    
    Returns:
        CVaR (average loss beyond VaR)
    """
    var_threshold = -value_at_risk(returns, confidence_level)
    tail_losses = returns[returns <= var_threshold]
    return abs(tail_losses.mean()) if len(tail_losses) > 0 else 0


class RiskMetrics:
    """Container for portfolio risk metrics."""
    
    def __init__(self, 
                 portfolio_returns: pd.Series,
                 market_returns: pd.Series,
                 weights: Dict[str, float]):
        """
        Calculate and store all risk metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market benchmark return series
            weights: Portfolio weights
        """
        self.returns = portfolio_returns
        self.market_returns = market_returns
        self.weights = weights
        
        # Calculate metrics
        self.volatility = annualized_volatility(portfolio_returns)
        self.max_drawdown = maximum_drawdown(portfolio_returns)
        self.beta = portfolio_beta(portfolio_returns, market_returns)
        self.var_95 = value_at_risk(portfolio_returns, 0.95)
        self.var_99 = value_at_risk(portfolio_returns, 0.99)
        self.cvar_95 = conditional_var(portfolio_returns, 0.95)
        
    def to_dict(self) -> Dict[str, float]:
        """Export metrics as dictionary."""
        return {
            'Annualized Volatility': self.volatility,
            'Maximum Drawdown': self.max_drawdown,
            'Portfolio Beta': self.beta,
            'VaR (95%)': self.var_95,
            'VaR (99%)': self.var_99,
            'CVaR (95%)': self.cvar_95
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export metrics as formatted DataFrame."""
        data = {
            'Metric': [
                'Annualized Volatility',
                'Maximum Drawdown',
                'Portfolio Beta (vs S&P 500)',
                'Value-at-Risk (95% confidence)',
                'Value-at-Risk (99% confidence)',
                'Conditional VaR (95%)'
            ],
            'Value': [
                f'{self.volatility:.2%}',
                f'{self.max_drawdown:.2%}',
                f'{self.beta:.3f}',
                f'{self.var_95:.2%}',
                f'{self.var_99:.2%}',
                f'{self.cvar_95:.2%}'
            ]
        }
        return pd.DataFrame(data)
