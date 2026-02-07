"""
Options Pricing Module
Implements Black-Scholes option pricing and hedging strategies.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict
from datetime import datetime, timedelta


class BlackScholes:
    """Black-Scholes option pricing model."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
        
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option delta."""
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1


class PortfolioHedge:
    """Portfolio hedging strategies using options and inverse ETFs."""
    
    def __init__(self, 
                 portfolio_value: float,
                 portfolio_beta: float,
                 spy_price: float,
                 risk_free_rate: float = 0.045):
        """
        Initialize hedging calculator.
        
        Args:
            portfolio_value: Total portfolio value
            portfolio_beta: Portfolio beta to S&P 500
            spy_price: Current SPY price
            risk_free_rate: Risk-free rate (annual)
        """
        self.portfolio_value = portfolio_value
        self.beta = portfolio_beta
        self.spy_price = spy_price
        self.risk_free_rate = risk_free_rate
        
    def calculate_put_hedge(self, 
                           hedge_ratio: float,
                           moneyness: float = 0.95,
                           days_to_expiry: int = 30,
                           implied_vol: float = 0.20) -> Dict:
        """
        Calculate protective put hedge on SPY.
        
        Args:
            hedge_ratio: Fraction of portfolio to hedge (0 to 1)
            moneyness: Strike as fraction of current price
            days_to_expiry: Days until option expiration
            implied_vol: Implied volatility assumption
        
        Returns:
            Dictionary with hedge details
        """
        # Effective exposure considering beta
        effective_exposure = self.portfolio_value * self.beta * hedge_ratio
        
        # Strike price
        strike = self.spy_price * moneyness
        
        # Time to expiration
        T = days_to_expiry / 365.0
        
        # Put option price
        put_price = BlackScholes.put_price(
            self.spy_price, strike, T, self.risk_free_rate, implied_vol
        )
        
        # Number of puts needed (each put covers 100 shares)
        shares_to_hedge = effective_exposure / self.spy_price
        contracts_needed = shares_to_hedge / 100
        
        # Total premium cost
        total_premium = contracts_needed * put_price * 100
        
        # Put delta for hedge effectiveness
        put_delta = BlackScholes.put_delta(
            self.spy_price, strike, T, self.risk_free_rate, implied_vol
        )
        
        return {
            'strategy': 'SPY Put Options',
            'underlying': 'SPY',
            'strike': strike,
            'current_price': self.spy_price,
            'moneyness': moneyness,
            'expiry_days': days_to_expiry,
            'implied_vol': implied_vol,
            'put_price': put_price,
            'contracts': contracts_needed,
            'total_premium': total_premium,
            'premium_pct': total_premium / self.portfolio_value,
            'delta': put_delta,
            'effective_hedge_ratio': abs(put_delta) * hedge_ratio
        }
    
    def calculate_inverse_etf_hedge(self, hedge_ratio: float) -> Dict:
        """
        Calculate inverse ETF hedge (e.g., SH, SPXU).
        
        Args:
            hedge_ratio: Fraction of portfolio to hedge
        
        Returns:
            Dictionary with hedge details
        """
        # Effective exposure considering beta
        effective_exposure = self.portfolio_value * self.beta * hedge_ratio
        
        # Use -1x inverse ETF (SH)
        inverse_multiplier = -1.0
        position_size = effective_exposure * abs(inverse_multiplier)
        
        # Estimated cost (SH typically trades around $13-15)
        estimated_price = 13.50
        shares_needed = position_size / estimated_price
        
        return {
            'strategy': 'Inverse ETF',
            'ticker': 'SH',
            'description': 'ProShares Short S&P 500',
            'multiplier': inverse_multiplier,
            'shares': shares_needed,
            'estimated_price': estimated_price,
            'position_value': position_size,
            'cost_pct': position_size / self.portfolio_value,
            'effective_hedge_ratio': hedge_ratio
        }
    
    def simulate_crash_with_hedge(self,
                                   portfolio_returns: np.ndarray,
                                   crash_magnitude: float,
                                   hedge_type: str,
                                   hedge_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate portfolio performance under crash with and without hedge.
        
        Args:
            portfolio_returns: Array of portfolio returns
            crash_magnitude: Market crash size (e.g., -0.30 for -30%)
            hedge_type: 'put' or 'inverse_etf'
            hedge_params: Hedge parameters from calculate_*_hedge
        
        Returns:
            Tuple of (unhedged_values, hedged_values)
        """
        initial_value = self.portfolio_value
        
        # Simulate crash: scale returns by crash magnitude
        crash_returns = portfolio_returns * (crash_magnitude / portfolio_returns.mean())
        
        # Unhedged portfolio
        unhedged_values = initial_value * (1 + crash_returns).cumprod()
        
        # Hedged portfolio
        if hedge_type == 'put':
            # Put option payoff
            final_spy_price = self.spy_price * (1 + crash_magnitude)
            put_payoff = max(hedge_params['strike'] - final_spy_price, 0)
            total_put_payoff = put_payoff * hedge_params['contracts'] * 100
            
            # Net hedge benefit (payoff minus premium)
            hedge_benefit = total_put_payoff - hedge_params['total_premium']
            
            # Approximate hedge protection throughout crash
            # Assume linear interpolation of hedge benefit
            hedge_protection = np.linspace(0, hedge_benefit, len(crash_returns))
            
        else:  # inverse_etf
            # Inverse ETF gains when market falls
            market_crash = crash_magnitude * self.beta
            inverse_gain = -market_crash * hedge_params['position_value']
            
            # Approximate gain accumulation
            hedge_protection = np.linspace(0, inverse_gain, len(crash_returns))
        
        hedged_values = unhedged_values + hedge_protection
        
        return unhedged_values, hedged_values
