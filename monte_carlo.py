"""
Monte Carlo Simulation Module
Implements parametric GBM simulations for portfolio stress testing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class MonteCarloSimulator:
    """
    Monte Carlo simulator using Geometric Brownian Motion.
    
    Assumptions:
    - Log-normal returns
    - Constant drift and volatility (stationarity)
    - Independent increments
    """
    
    def __init__(self, returns: pd.Series):
        """
        Initialize simulator from historical returns.
        
        Args:
            returns: Historical daily returns
        """
        self.returns = returns
        self.mu = returns.mean()  # Daily drift
        self.sigma = returns.std()  # Daily volatility
        
    def simulate_paths(self,
                      initial_value: float,
                      n_days: int,
                      n_simulations: int,
                      stress_multiplier: float = 1.0,
                      seed: int = 42) -> np.ndarray:
        """
        Simulate portfolio value paths using GBM.
        
        Args:
            initial_value: Starting portfolio value
            n_days: Number of days to simulate
            n_simulations: Number of simulation paths
            stress_multiplier: Volatility multiplier for stress scenarios
            seed: Random seed for reproducibility
        
        Returns:
            Array of shape (n_simulations, n_days) with portfolio values
        """
        np.random.seed(seed)
        
        # Stressed volatility
        sigma_stressed = self.sigma * stress_multiplier
        
        # Generate random shocks
        dt = 1  # Daily time step
        random_shocks = np.random.normal(0, 1, (n_simulations, n_days))
        
        # GBM formula: dS/S = mu*dt + sigma*dW
        returns = self.mu * dt + sigma_stressed * np.sqrt(dt) * random_shocks
        
        # Calculate cumulative returns
        price_paths = initial_value * np.exp(np.cumsum(returns, axis=1))
        
        return price_paths
    
    def simulate_with_hedge(self,
                           initial_value: float,
                           n_days: int,
                           n_simulations: int,
                           hedge_params: Dict,
                           stress_multiplier: float = 1.5,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths with and without hedging under stress.
        
        Args:
            initial_value: Starting portfolio value
            n_days: Number of days to simulate
            n_simulations: Number of simulation paths
            hedge_params: Hedge parameters from PortfolioHedge
            stress_multiplier: Volatility multiplier for stress
            seed: Random seed
        
        Returns:
            Tuple of (unhedged_paths, hedged_paths)
        """
        # Unhedged simulation
        unhedged = self.simulate_paths(
            initial_value, n_days, n_simulations, stress_multiplier, seed
        )
        
        # Hedged simulation - subtract hedge cost, add protection
        hedge_cost = hedge_params.get('total_premium', 0) or hedge_params.get('position_value', 0)
        initial_hedged = initial_value - hedge_cost
        
        # Simulate hedged portfolio (same random seed for fair comparison)
        hedged_base = self.simulate_paths(
            initial_hedged, n_days, n_simulations, stress_multiplier, seed
        )
        
        # Approximate hedge benefit in tail scenarios
        # For put options: benefit increases with losses
        final_values_unhedged = unhedged[:, -1]
        losses = np.maximum(initial_value - final_values_unhedged, 0)
        
        # Hedge effectiveness scales with loss magnitude
        if 'strike' in hedge_params:
            # Put option hedge
            hedge_ratio = hedge_params.get('effective_hedge_ratio', 0.5)
            protection = losses * hedge_ratio
        else:
            # Inverse ETF hedge
            hedge_ratio = hedge_params.get('effective_hedge_ratio', 0.5)
            protection = losses * hedge_ratio
        
        # Apply protection to final values
        hedged = hedged_base.copy()
        hedged[:, -1] += protection
        
        return unhedged, hedged
    
    def calculate_statistics(self, paths: np.ndarray, initial_value: float) -> Dict:
        """
        Calculate statistics from simulated paths.
        
        Args:
            paths: Array of simulated portfolio values
            initial_value: Starting value
        
        Returns:
            Dictionary of statistics
        """
        final_values = paths[:, -1]
        returns = (final_values - initial_value) / initial_value
        
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_95': np.percentile(final_values, 95),
            'max_drawdown': self._calculate_max_drawdown(paths, initial_value),
            'prob_loss': np.mean(final_values < initial_value)
        }
    
    def _calculate_max_drawdown(self, paths: np.ndarray, initial_value: float) -> float:
        """Calculate average maximum drawdown across paths."""
        drawdowns = []
        for path in paths:
            cummax = np.maximum.accumulate(path)
            dd = (path - cummax) / cummax
            drawdowns.append(dd.min())
        return np.mean(drawdowns)
    
    def get_percentile_paths(self, paths: np.ndarray, percentiles: list = [5, 50, 95]) -> Dict:
        """
        Extract specific percentile paths for visualization.
        
        Args:
            paths: Array of simulated paths
            percentiles: List of percentiles to extract
        
        Returns:
            Dictionary mapping percentile to path
        """
        final_values = paths[:, -1]
        result = {}
        
        for p in percentiles:
            idx = np.argmin(np.abs(final_values - np.percentile(final_values, p)))
            result[f'p{p}'] = paths[idx]
        
        return result


class StressScenario:
    """Predefined stress scenarios for testing."""
    
    @staticmethod
    def black_swan_2008() -> Dict:
        """2008 Financial Crisis parameters."""
        return {
            'name': '2008 Financial Crisis',
            'volatility_multiplier': 2.5,
            'duration_days': 252,  # 1 year
            'market_drop': -0.38
        }
    
    @staticmethod
    def covid_2020() -> Dict:
        """COVID-19 2020 parameters."""
        return {
            'name': 'COVID-19 2020',
            'volatility_multiplier': 3.0,
            'duration_days': 60,
            'market_drop': -0.34
        }
    
    @staticmethod
    def custom(volatility_multiplier: float, duration_days: int, market_drop: float) -> Dict:
        """Custom stress scenario."""
        return {
            'name': 'Custom Stress',
            'volatility_multiplier': volatility_multiplier,
            'duration_days': duration_days,
            'market_drop': market_drop
        }
