from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from prompts.base import BasePromptStrategy
from llm.base import BaseLLM
from .config import ExperimentConfig
from .cache import ResultCache

class BatchProcessor:
    """Processes data in batches and handles LLM interactions."""
    
    def __init__(self, llm: BaseLLM, config: ExperimentConfig):
        self.llm = llm
        self.config = config
        self.cache = ResultCache(config.save_dir) if config.cache_results else None
    
    def process_batch(self, 
                     data: List[Dict], 
                     strategy: BasePromptStrategy,
                     **kwargs) -> List[str]:
        """Process a batch of data using the given strategy."""
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_requests) as executor:
            futures = []
            for item in data:
                future = executor.submit(
                    self._process_single_item,
                    item,
                    strategy,
                    **kwargs
                )
                futures.append(future)
            
            responses = [future.result() for future in futures]
        
        return responses
    
    def _process_single_item(self, 
                           item: Dict, 
                           strategy: BasePromptStrategy, 
                           **kwargs) -> str:
        """Process a single data item."""
        prompt = strategy.generate_prompt(
            item[kwargs.get('input_column', 'text')],
            **kwargs
        )
        
        if self.cache:
            cached_response = self.cache.get_cached_response(prompt)
            if cached_response:
                return cached_response
        
        response = self.llm.generate(prompt)
        
        if self.cache:
            self.cache.cache_response(prompt, response)
        
        return response