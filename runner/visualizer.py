from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from visualization import plots
import logging
import traceback

class ExperimentVisualizer:
    """Generates visualizations for experiment results."""
    
    def __init__(self, save_dir: str):
        self.viz_path = Path(save_dir) / "visualizations"
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def generate_visualizations(self, results: Dict[str, Any]) -> None:
        """Generate all visualizations for the experiment results."""
        if not results:
            logging.warning("No results to visualize")
            return
            
        try:
            for dataset_name, dataset_results in results.items():
                if dataset_results:  # Check if there are results for this dataset
                    self._generate_metric_comparisons(dataset_results, dataset_name)
            self._generate_heatmap(results)
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
        finally:
            plt.close('all')  # Clean up all figures
    
    def _generate_metric_comparisons(self, dataset_results: Dict[str, Any], dataset_name: str) -> None:
        """Generate comparison plots for each metric."""
        try:
            # Verify we have valid results
            if not dataset_results:
                logging.warning(f"No results to compare for dataset {dataset_name}")
                return
                
            # Get first valid strategy result
            first_strategy = next((
                results for results in dataset_results.values()
                if results and 'metrics' in results
            ), None)
            
            if not first_strategy:
                logging.warning(f"No valid strategy results found for dataset {dataset_name}")
                return
                
            metrics = first_strategy['metrics']
                
            for metric_name in metrics:
                # Extract mean values for each strategy
                metric_data = {}
                for strategy, results in dataset_results.items():
                    # print("RESULTS", results)
                    # print("STRATEGY", strategy)
                    if results and 'metrics' in results:
                        try:
                            if isinstance(results['metrics'][metric_name], dict):
                                metric_data[strategy] = results['metrics'][metric_name]['mean']
                            else:
                                metric_data[strategy] = results['metrics'][metric_name]
                        except (KeyError, TypeError) as e:
                            logging.warning(f"Error extracting metric {metric_name} for strategy {strategy}: {e}")
                            continue
                
                if metric_data:  # Only create plot if we have data
                    # Create and save plot
                    if metric_name != "details":
                        fig = plots.plot_strategy_comparison(
                            metric_data,
                            metric_name,
                            f"{dataset_name} - {metric_name}"
                        )
                        
                        # Save as PNG
                        save_path = self.viz_path / f"{dataset_name}_{metric_name}_comparison.png"
                        fig.savefig(
                            save_path,
                            dpi=300,
                            bbox_inches='tight'
                        )
                        plt.close(fig)
                        logging.info(f"Saved comparison plot to {save_path}")
                
        except Exception as e:
            logging.error(f"Error generating metric comparisons for {dataset_name}: {e}")
            logging.error(traceback.format_exc())
    
    def _generate_heatmap(self, results: Dict[str, Any]) -> None:
        """Generate overall performance heatmap."""
        try:
            # Create lists to store data for DataFrame
            datasets = []
            strategies = []
            scores = []
            
            for dataset_name, dataset_results in results.items():
                for strategy_name, strategy_results in dataset_results.items():
                    if not strategy_results or 'metrics' not in strategy_results:
                        continue
                        
                    # Get the first metric's value
                    try:
                        metric_value = next(iter(strategy_results['metrics'].values()))
                        score = metric_value['mean'] if isinstance(metric_value, dict) else metric_value
                        
                        datasets.append(dataset_name)
                        strategies.append(strategy_name)
                        scores.append(score)
                    except (StopIteration, KeyError, TypeError) as e:
                        logging.warning(f"Error extracting score for {dataset_name}-{strategy_name}: {e}")
                        continue
            
            if not scores:
                logging.warning("No valid scores found for heatmap")
                return
                
            # Create DataFrame and pivot for heatmap
            df = pd.DataFrame({
                'Dataset': datasets,
                'Strategy': strategies,
                'Score': scores
            })
            
            heatmap_data = df.pivot(
                index='Dataset',
                columns='Strategy',
                values='Score'
            )
            
            # Create and save heatmap
            fig = plots.plot_performance_heatmap(heatmap_data)
            save_path = self.viz_path / "performance_heatmap.png"
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)
            logging.info(f"Saved heatmap to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating heatmap: {e}")