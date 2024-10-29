from abc import ABC, abstractmethod
from typing import Union, List, Dict

class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Generate response from prompt.
        
        Args:
            prompt: Either a string prompt or a list of message dictionaries
                   for chat-based models.
        
        Returns:
            str: The model's response
        """
        pass