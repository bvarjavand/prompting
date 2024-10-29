from .base import BaseDataset, DatasetConfig
from typing import Dict, Any
import datasets
from rouge_score import rouge_scorer

class TruthfulQADataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
        self.train_data, self.test_data = self.load_data()

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source)
        # TruthfulQA doesn't have a train split, so we create one
        full_df = dataset['validation'].to_pandas()
        train_df = full_df.sample(frac=0.8, random_state=42)
        test_df = full_df.drop(train_df.index)
        
        if self.config.sample_size:
            train_df = train_df.sample(n=min(self.config.sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(n=min(self.config.sample_size // 5, len(test_df)), random_state=42)
        
        return train_df, test_df

    def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
        # Split multiple correct answers
        correct_answers = target.split('|')
        
        # Calculate maximum ROUGE score against any correct answer
        max_score = max(
            self.scorer.score(response.lower(), answer.lower())['rouge2'].fmeasure
            for answer in correct_answers
        )
        
        return {
            "truth_score": max_score
        }