"""
Visualization Module
Plotting utilities for portfolio analytics using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List


def plot_sector_allocation(sector_df: pd.DataFrame, title: str = "Portfolio Allocation by Sector") -> go.Figure:
    """
    Create interactive doughnut chart for sector allocation.
    
    Args:
        sector_df: DataFrame with Sector, Weight columns
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Pie(
        labels=sector_df['Sector'],
        values=sector_df['Weight'],
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        )
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True,
        height=500,
        font=dict(size=12)
    )
    
    return fig


def plot_crash_scenario(unhedged: np.ndarray, 
                        hedged: np.ndarray,
                        crash_magnitude: float,
                        hedge_type: str) -> go.Figure:
    """
    Plot portfolio performance under crash scenario with/without hedge.
    
    Args:
        unhedged: Array of unhedged portfolio values
        hedged: Array of hedged portfolio values
        crash_magnitude: Crash size (e.g., -0.30)
        hedge_type: Type of hedge used
    
    Returns:
        Plotly figure
    """
    days = np.arange(len(unhedged))
    
    fig = go.Figure()
    
    # Unhedged portfolio
    fig.add_trace(go.Scatter(
        x=days,
        y=unhedged,
        mode='lines',
        name='Unhedged Portfolio',
        line=dict(color='red', width=2),
        hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Hedged portfolio
    fig.add_trace(go.Scatter(
        x=days,
        y=hedged,
        mode='lines',
        name=f'Hedged Portfolio ({hedge_type})',
        line=dict(color='green', width=2),
        hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Black Swan Stress Test: {crash_magnitude:.0%} Market Crash',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def plot_monte_carlo_paths(paths: np.ndarray,
                           percentile_paths: Dict,
                           initial_value: float,
                           title: str = "Monte Carlo Simulation Paths") -> go.Figure:
    """
    Plot Monte Carlo simulation paths with percentiles highlighted.
    
    Args:
        paths: Array of all simulated paths
        percentile_paths: Dictionary of specific percentile paths
        initial_value: Starting portfolio value
        title: Chart title
    
    Returns:
        Plotly figure
    """
    n_days = paths.shape[1]
    days = np.arange(n_days)
    
    fig = go.Figure()
    
    # Plot sample of paths (transparent)
    sample_size = min(100, paths.shape[0])
    sample_indices = np.random.choice(paths.shape[0], sample_size, replace=False)
    
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=days,
            y=paths[idx],
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Highlight percentile paths
    colors = {'p5': 'red', 'p50': 'blue', 'p95': 'green'}
    names = {'p5': '5th Percentile', 'p50': 'Median', 'p95': '95th Percentile'}
    
    for key, path in percentile_paths.items():
        fig.add_trace(go.Scatter(
            x=days,
            y=path,
            mode='lines',
            name=names.get(key, key),
            line=dict(color=colors.get(key, 'black'), width=3),
            hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ))
    
    # Initial value reference line
    fig.add_hline(y=initial_value, line_dash="dash", line_color="gray",
                  annotation_text="Initial Value")
    
    fig.update_layout(
        title=title,
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_return_distribution(returns: np.ndarray, 
                             title: str = "Simulated Return Distribution") -> go.Figure:
    """
    Plot histogram of simulated returns.
    
    Args:
        returns: Array of returns
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Returns',
        marker=dict(color='steelblue', line=dict(color='white', width=1))
    ))
    
    # Add mean line
    mean_return = np.mean(returns)
    fig.add_vline(x=mean_return, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_return:.2%}")
    
    # Merge the axis formatting into the main layout update
    fig.update_layout(
        title=title,
        xaxis_title='Return',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='.1%')  # <--- Moved here
    )
    
    # DELETE or COMMENT OUT this line:
    # fig.update_xaxis(tickformat='.1%') 
    
    return fig


def plot_hedged_vs_unhedged_monte_carlo(unhedged_paths: np.ndarray,
                                        hedged_paths: np.ndarray,
                                        initial_value: float) -> go.Figure:
    """
    Compare hedged vs unhedged Monte Carlo outcomes.
    
    Args:
        unhedged_paths: Unhedged simulation paths
        hedged_paths: Hedged simulation paths
        initial_value: Starting value
    
    Returns:
        Plotly figure
    """
    n_days = unhedged_paths.shape[1]
    days = np.arange(n_days)
    
    # Calculate percentile bands
    unhedged_p5 = np.percentile(unhedged_paths, 5, axis=0)
    unhedged_p50 = np.percentile(unhedged_paths, 50, axis=0)
    unhedged_p95 = np.percentile(unhedged_paths, 95, axis=0)
    
    hedged_p5 = np.percentile(hedged_paths, 5, axis=0)
    hedged_p50 = np.percentile(hedged_paths, 50, axis=0)
    hedged_p95 = np.percentile(hedged_paths, 95, axis=0)
    
    fig = go.Figure()
    
    # Unhedged confidence band
    fig.add_trace(go.Scatter(
        x=days, y=unhedged_p95,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=unhedged_p5,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.2)',
        fill='tonexty',
        name='Unhedged (5th-95th percentile)',
        hoverinfo='skip'
    ))
    
    # Unhedged median
    fig.add_trace(go.Scatter(
        x=days, y=unhedged_p50,
        mode='lines',
        name='Unhedged (Median)',
        line=dict(color='red', width=2)
    ))
    
    # Hedged confidence band
    fig.add_trace(go.Scatter(
        x=days, y=hedged_p95,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=hedged_p5,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(0, 255, 0, 0.2)',
        fill='tonexty',
        name='Hedged (5th-95th percentile)',
        hoverinfo='skip'
    ))
    
    # Hedged median
    fig.add_trace(go.Scatter(
        x=days, y=hedged_p50,
        mode='lines',
        name='Hedged (Median)',
        line=dict(color='green', width=2)
    ))
    
    fig.add_hline(y=initial_value, line_dash="dash", line_color="gray",
                  annotation_text="Initial Value")
    
    fig.update_layout(
        title='Monte Carlo Stress Test: Hedged vs Unhedged',
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig
