from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import pandas as pd

@dataclass
class DatasetConfig:
    name: str
    source: str
    input_column: str
    target_column: str
    metric_names: List[str]
    sample_size: int = None

class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.train_data = None
        self.test_data = None
        
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
    
    @abstractmethod
    def evaluate_response(self, response: str, target: Any) -> Dict[str, float]:
        """Evaluate model response against target"""
        pass