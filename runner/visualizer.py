from pathlib import Path
from typing import Dict
from visualization import plots

class ExperimentVisualizer:
    """Generates visualizations for experiment results."""
    
    def __init__(self, save_dir: str):
        self.viz_path = Path(save_dir) / "visualizations"
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def generate_visualizations(self, results: Dict):
        """Generate all visualizations for the experiment results."""
        self._generate_metric_comparisons(results)
        self._generate_heatmap(results)
    
    def _generate_metric_comparisons(self, results: Dict):
        """Generate comparison plots for each metric."""
        for dataset_name, dataset_results in results.items():
            for strategy_results in dataset_results.values():
                for metric_name in strategy_results['metrics'].keys():
                    metric_data = {
                        strategy: results['metrics'][metric_name]['mean']
                        for strategy, results in dataset_results.items()
                    }
                    
                    fig = plots.plot_strategy_comparison(
                        metric_data,
                        metric_name,
                        f"{dataset_name} - {metric_name}"
                    )
                    
                    fig.write_html(
                        self.viz_path / f"{dataset_name}_{metric_name}_comparison.html"
                    )
    
    def _generate_heatmap(self, results: Dict):
        """Generate overall performance heatmap."""
        heatmap_data = {}
        for dataset_name, dataset_results in results.items():
            for strategy, results in dataset_results.items():
                if strategy not in heatmap_data:
                    heatmap_data[strategy] = {}
                first_metric = list(results['metrics'].keys())[0]
                heatmap_data[strategy][dataset_name] = results['metrics'][first_metric]['mean']
        
        heatmap_fig = plots.plot_performance_heatmap(heatmap_data)
        heatmap_fig.write_html(self.viz_path / "performance_heatmap.html")