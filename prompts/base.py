from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

@dataclass
class DatasetConfig:
    name: str
    source: str
    input_column: str
    target_column: str
    metric_names: List[str]
    sample_size: Optional[int] = None

class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.train_data = None
        self.test_data = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def evaluate_response(self, response: str, target: Any) -> Dict[str, float]:
        pass

class SimplePromptStrategy:
    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template
    
    def generate_prompt(self, text: str, **kwargs) -> str:
        return self.template.format(text=text)