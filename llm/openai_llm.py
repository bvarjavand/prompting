import os
import openai
from typing import Union, List, Dict
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    """OpenAI API implementation."""
    
    def __init__(self, 
                 model: str = "gpt-3.5-turbo", 
                 temperature: float = 0.2,
                 max_tokens: int = None,
                 top_p: float = None,
                 frequency_penalty: float = None,
                 presence_penalty: float = None):
        """Initialize OpenAI LLM.
        
        Args:
            model: Model identifier (e.g., "gpt-3.5-turbo", "gpt-4")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
        """
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
    
    def generate(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Generate response using OpenAI API.
        
        Args:
            prompt: Either a string prompt or a list of message dictionaries
        
        Returns:
            str: The model's response
        """
        # Convert string prompt to chat format if necessary
        if isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        else:
            messages = prompt
        
        # Prepare API call parameters
        params = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature
        }
        
        # Add optional parameters if specified
        if self.max_tokens is not None:
            params['max_tokens'] = self.max_tokens
        if self.top_p is not None:
            params['top_p'] = self.top_p
        if self.frequency_penalty is not None:
            params['frequency_penalty'] = self.frequency_penalty
        if self.presence_penalty is not None:
            params['presence_penalty'] = self.presence_penalty
        
        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")