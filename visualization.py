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
    # Minimalist grayscale palette with dark blue accents
    colors = [
        '#ffffff', '#cccccc', '#1e3a5f', '#999999', '#666666',
        '#2c5282', '#aaaaaa', '#bbbbbb', '#0f1c2e', '#888888',
        '#dddddd'
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=sector_df['Sector'],
        values=sector_df['Weight'],
        hole=0.5,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=colors[:len(sector_df)],
            line=dict(color='#000000', width=1)
        ),
        textfont=dict(color='#ffffff', size=11, family='Cousine')
    )])
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#ffffff', size=16, family='Alliance No.2'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        height=500,
        font=dict(size=11, color='#cccccc', family='Cousine'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#1a1a1a',
            borderwidth=0,
            font=dict(color='#cccccc', size=10)
        )
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
    
    # Unhedged portfolio - Light gray
    fig.add_trace(go.Scatter(
        x=days,
        y=unhedged,
        mode='lines',
        name='Unhedged',
        line=dict(color='#666666', width=2),
        hovertemplate='<b>Day %{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Hedged portfolio - White/Dark Blue
    fig.add_trace(go.Scatter(
        x=days,
        y=hedged,
        mode='lines',
        name=f'Hedged ({hedge_type})',
        line=dict(color='#ffffff', width=2),
        hovertemplate='<b>Day %{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'BLACK SWAN STRESS TEST: {crash_magnitude:.0%} CRASH',
            font=dict(color='#ffffff', size=16, family='Alliance No.2'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cccccc', size=11, family='Cousine'),
        xaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc',
            showgrid=True
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#1a1a1a',
            borderwidth=0,
            font=dict(color='#cccccc', size=10),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
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
    
    # Plot sample of paths (semi-transparent gray)
    sample_size = min(100, paths.shape[0])
    sample_indices = np.random.choice(paths.shape[0], sample_size, replace=False)
    
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=days,
            y=paths[idx],
            mode='lines',
            line=dict(color='rgba(102, 102, 102, 0.15)', width=1),
            opacity=0.5,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Highlight percentile paths
    colors = {'p5': '#666666', 'p50': '#ffffff', 'p95': '#1e3a5f'}
    names = {'p5': '5th Percentile', 'p50': 'Median', 'p95': '95th Percentile'}
    
    for key, path in percentile_paths.items():
        fig.add_trace(go.Scatter(
            x=days,
            y=path,
            mode='lines',
            name=names.get(key, key),
            line=dict(color=colors.get(key, '#ffffff'), width=2),
            hovertemplate='<b>Day %{x}</b><br>Value: $%{y:,.0f}<extra></extra>'
        ))
    
    # Initial value reference line
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color="#1e3a5f",
        line_width=1,
        annotation_text="Initial",
        annotation_font_color="#888888",
        annotation_font_size=10
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#ffffff', size=16, family='Alliance No.2'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cccccc', size=11, family='Cousine'),
        xaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc'
        ),
        yaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#1a1a1a',
            borderwidth=0,
            font=dict(color='#cccccc', size=10)
        )
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
        marker=dict(
            color='#ffffff',
            line=dict(color='#1a1a1a', width=1),
            opacity=0.7
        )
    ))
    
    # Add mean line
    mean_return = np.mean(returns)
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="#1e3a5f",
        line_width=2,
        annotation_text=f"Mean: {mean_return:.2%}",
        annotation_font_color="#1e3a5f",
        annotation_font_size=11
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#ffffff', size=16, family='Alliance No.2'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Return',
        yaxis_title='Frequency',
        height=450,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cccccc', size=11, family='Cousine'),
        xaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc',
            tickformat='.1%'
        ),
        yaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc'
        )
    )
    
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
    
    # Unhedged confidence band - Gray
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
        fillcolor='rgba(102, 102, 102, 0.2)',
        fill='tonexty',
        name='Unhedged Range (5th-95th)',
        hoverinfo='skip'
    ))
    
    # Unhedged median - Gray
    fig.add_trace(go.Scatter(
        x=days, y=unhedged_p50,
        mode='lines',
        name='Unhedged (Median)',
        line=dict(color='#666666', width=2)
    ))
    
    # Hedged confidence band - White/Blue
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
        fillcolor='rgba(30, 58, 95, 0.15)',
        fill='tonexty',
        name='Hedged Range (5th-95th)',
        hoverinfo='skip'
    ))
    
    # Hedged median - White
    fig.add_trace(go.Scatter(
        x=days, y=hedged_p50,
        mode='lines',
        name='Hedged (Median)',
        line=dict(color='#ffffff', width=2)
    ))
    
    # Initial value reference line
    fig.add_hline(
        y=initial_value,
        line_dash="dash",
        line_color="#1e3a5f",
        line_width=1,
        annotation_text="Initial",
        annotation_font_color="#888888",
        annotation_font_size=10
    )
    
    fig.update_layout(
        title=dict(
            text='MONTE CARLO STRESS TEST: HEDGED VS UNHEDGED',
            font=dict(color='#ffffff', size=16, family='Alliance No.2'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Days',
        yaxis_title='Portfolio Value ($)',
        height=500,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cccccc', size=11, family='Cousine'),
        xaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc'
        ),
        yaxis=dict(
            gridcolor='#1a1a1a',
            zerolinecolor='#1a1a1a',
            color='#cccccc'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#1a1a1a',
            borderwidth=0,
            font=dict(color='#cccccc', size=10)
        )
    )
    
    return fig
