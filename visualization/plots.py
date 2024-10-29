import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List

def plot_strategy_comparison(results: Dict[str, Dict[str, float]], 
                           metric: str,
                           title: str = None):
    """Create a bar plot comparing different strategies."""
    df = pd.DataFrame(results).reset_index()
    df.columns = ['Strategy'] + list(results.keys())
    
    fig = go.Figure(data=[
        go.Bar(name=dataset, x=df['Strategy'], y=df[dataset])
        for dataset in results.keys()
    ])
    
    fig.update_layout(
        title=title or f'Strategy Comparison - {metric}',
        xaxis_title="Prompt Strategy",
        yaxis_title=metric.capitalize(),
        barmode='group'
    )
    return fig

def plot_performance_heatmap(results: Dict[str, Dict[str, float]],
                           title: str = None):
    """Create a heatmap of strategy performance."""
    df = pd.DataFrame(results)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=title or 'Strategy Performance Heatmap',
        xaxis_title="Dataset",
        yaxis_title="Strategy"
    )
    return fig