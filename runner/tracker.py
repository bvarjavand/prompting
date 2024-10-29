from typing import Dict
from pathlib import Path
import logging
import json
from datetime import datetime

from .config import ExperimentConfig

class ExperimentTracker:
    """Tracks and saves experiment results and metadata."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.experiment_name
        }
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the experiment."""
        log_path = Path(self.config.save_dir) / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / f"{self.config.experiment_name}.log"),
                logging.StreamHandler()
            ]
        )
    
    def add_result(self, dataset_name: str, strategy_name: str, result: Dict):
        """Add a result to the tracker."""
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
        self.results[dataset_name][strategy_name] = result
        logging.info(f"Added result for {dataset_name} - {strategy_name}")
    
    def save_results(self):
        """Save results and metadata to disk."""
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            'metadata': self.metadata,
            'results': self.results
        }
        
        results_path = save_path / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logging.info(f"Results saved to {results_path}")
        return results_path