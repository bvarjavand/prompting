from typing import Dict
from .evaluators import BaseEvaluator, SimpleMathEvaluator, ClassificationEvaluator, QAEvaluator

class EvaluatorFactory:
    """Factory for creating appropriate evaluators."""
    
    @staticmethod
    def create_evaluator(task_type: str) -> BaseEvaluator:
        evaluators = {
            'math': SimpleMathEvaluator,
            'classification': ClassificationEvaluator,
            'qa': QAEvaluator
        }
        
        if task_type not in evaluators:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return evaluators[task_type]()