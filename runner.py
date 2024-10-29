from typing import List
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

from data.base import BaseDataset
from prompts.base import BasePromptStrategy
from llm.base import BaseLLM

from runner.config import ExperimentConfig
from runner.tracker import ExperimentTracker
from runner.processor import BatchProcessor
from runner.visualizer import ExperimentVisualizer

class ExperimentRunner:
    """Main class for running prompt engineering experiments."""
    
    def __init__(self,
                 datasets: List[BaseDataset],
                 strategies: List[BasePromptStrategy],
                 llm: BaseLLM,
                 config: ExperimentConfig):
        self.datasets = datasets
        self.strategies = strategies
        self.config = config
        self.tracker = ExperimentTracker(config)
        self.processor = BatchProcessor(llm, config)
        self.visualizer = ExperimentVisualizer(config.save_dir)
    
    def run(self) -> Path:
        """Run the complete experiment suite."""
        logging.info(f"Starting experiment: {self.config.experiment_name}")
        
        for dataset in self.datasets:
            self._run_dataset(dataset)
        
        results_path = self.tracker.save_results()
        self.visualizer.generate_visualizations(self.tracker.results)
        
        logging.info("Experiment completed successfully")
        return results_path
    
    def _run_dataset(self, dataset: BaseDataset):
        """Run experiments for a single dataset."""
        logging.info(f"Processing dataset: {dataset.config.name}")
        
        for strategy in tqdm(self.strategies, 
                           desc=f"Strategies for {dataset.config.name}"):
            self._run_strategy(dataset, strategy)
    
    def _run_strategy(self, dataset: BaseDataset, strategy: BasePromptStrategy):
        """Run experiments for a single strategy on a dataset."""
        logging.info(f"Applying strategy: {strategy.name}")
        
        # Process data in batches
        responses = []
        for i in range(0, len(dataset.test_data), self.config.batch_size):
            batch = dataset.test_data[i:i + self.config.batch_size]
            batch_responses = self.processor.process_batch(
                batch,
                strategy,
                task=dataset.config.name,
                input_column=dataset.config.input_column
            )
            responses.extend(batch_responses)
        
        # Evaluate responses
        metrics = self._evaluate_responses(dataset, responses)
        
        result = {
            'metrics': metrics,
            'responses': responses if self.config.save_responses else None
        }
        
        self.tracker.add_result(dataset.config.name, strategy.name, result)
    
    def _evaluate_responses(self, dataset: BaseDataset, responses: List[str]) -> dict:
        """Evaluate model responses against dataset targets."""
        metrics = []
        for response, item in zip(responses, dataset.test_data):
            metric = dataset.evaluate_response(
                response, 
                item[dataset.config.target_column]
            )
            metrics.append(metric)
        
        # Aggregate metrics
        aggregated = {}
        for metric_name in metrics[0].keys():
            values = [m[metric_name] for m in metrics]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated

if __name__ == "__main__":
    # Example usage
    from data.datasets import EmotionDataset
    from data.gsm8k import GSM8KDataset
    from data.truthfulqa import TruthfulQADataset
    from prompts.basic import ZeroShotPrompt, FewShotPrompt
    from prompts.chain import ChainOfThoughtPrompt
    from llm.openai_llm import OpenAILLM
    
    config = ExperimentConfig(
        experiment_name="prompt_engineering_comparison",
        save_dir="experiments",
        batch_size=10,
        max_parallel_requests=5,
        cache_results=True
    )
    
    datasets = [
        EmotionDataset(...),
        GSM8KDataset(...),
        TruthfulQADataset(...)
    ]
    
    strategies = [
        ZeroShotPrompt(),
        FewShotPrompt(n_shots=3),
        ChainOfThoughtPrompt()
    ]
    
    llm = OpenAILLM(model="gpt-3.5-turbo", temperature=0.2)
    
    runner = ExperimentRunner(datasets, strategies, llm, config)
    results_path = runner.run()