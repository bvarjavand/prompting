import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_strategy_comparison(metric_data: dict, metric_name: str, title: str):
    """Create bar plot comparing strategy performance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = list(metric_data.keys())
    values = list(metric_data.values())
    # Create bar plot
    bars = ax.bar(strategies, values)
    
    # Customize appearance
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.1)  # Add 10% padding above highest bar
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_performance_heatmap(data: pd.DataFrame):
    """Create heatmap of model performance across different strategies."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        data,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Score'},
        ax=ax
    )
    
    # Customize appearance
    plt.title('Performance Across Datasets and Strategies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig