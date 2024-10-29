from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Dict
import time
import logging
from runner.cache import ResultCache
from llm.base import BaseLLM
from prompts.base import SimplePromptStrategy
from runner.config import ExperimentConfig

class BatchProcessor:
    def __init__(self, llm: BaseLLM, config: ExperimentConfig):
        self.llm = llm
        self.config = config
        self.cache = ResultCache(config.save_dir) if config.cache_results else None
        self.timeout = 30  # timeout in seconds
    
    def process_batch(self, 
                     data: List[Dict], 
                     strategy: SimplePromptStrategy,
                     **kwargs) -> List[str]:
        """Process a batch of data with timeout and error handling."""
        responses = []
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_requests) as executor:
                # Generate prompts first
                prompts = []
                for item in data:
                    try:
                        prompt = strategy.generate_prompt(
                            item[kwargs.get('input_column', 'text')],
                            **kwargs
                        )
                        prompts.append(prompt)
                    except Exception as e:
                        logging.error(f"Error generating prompt: {e}")
                        prompts.append(None)
                
                # Process each prompt
                futures = []
                for prompt in prompts:
                    if prompt is not None:
                        if self.cache:
                            cached = self.cache.get_cached_response(prompt)
                            if cached is not None:
                                responses.append(cached)
                                continue
                        
                        futures.append(executor.submit(self._get_response, prompt))
                    else:
                        responses.append("")
                
                # Collect responses with timeout
                for future in futures:
                    try:
                        response = future.result(timeout=self.timeout)
                        responses.append(response)
                        
                        # Cache the response if needed
                        if self.cache and response:
                            try:
                                self.cache.cache_response(
                                    prompts[len(responses)-1], 
                                    response
                                )
                            except Exception as e:
                                logging.error(f"Cache error: {e}")
                                
                    except TimeoutError:
                        logging.warning("Request timed out")
                        responses.append("")
                    except Exception as e:
                        logging.error(f"Error processing response: {e}")
                        responses.append("")
                
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
        
        # Ensure we have the right number of responses
        while len(responses) < len(data):
            responses.append("")
        
        logging.info(f"Batch processed in {time.time() - start_time:.2f} seconds")
        return responses

    def _get_response(self, prompt: str) -> str:
        """Get response with error handling."""
        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logging.error(f"LLM error: {e}")
            return ""