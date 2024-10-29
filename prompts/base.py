from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptStrategy(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        pass
    
    def __repr__(self):
        return f"{self.name}: {self.description}"