from .base import BaseDataset, DatasetConfig
from typing import Dict, Any
import datasets
import re

class GSM8KDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.train_data, self.test_data = self.load_data()

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source)
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        if self.config.sample_size:
            train_df = train_df.sample(n=min(self.config.sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(n=min(self.config.sample_size // 5, len(test_df)), random_state=42)
        
        return train_df, test_df

    def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
        pred_answer = self._extract_number(response)
        true_answer = self._extract_number(target)
        
        if pred_answer is None or true_answer is None:
            return {"accuracy": 0.0}
        
        return {
            "accuracy": float(abs(pred_answer - true_answer) < 1e-6)
        }

    def _extract_number(self, text: str) -> float:
        """Extract the final number from the text."""
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return float(numbers[-1]) if numbers else None