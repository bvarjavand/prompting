from pathlib import Path
import json
import threading
import logging
from typing import Dict, Optional

class ResultCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "response_cache.json"
        self.cache = self._load_cache()
        self.lock = threading.Lock()
        self.max_retries = 3
    
    def _load_cache(self) -> Dict:
        self.max_retries = 3
        if self.cache_file.exists():
            for attempt in range(self.max_retries):
                try:
                    with open(self.cache_file, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    logging.warning(f"Cache load error (attempt {attempt+1}): {e}")
                except Exception as e:
                    logging.error(f"Unexpected cache error: {e}")
        return {}
    
    def save_cache(self):
        with self.lock:
            for attempt in range(self.max_retries):
                try:
                    cache_copy = dict(self.cache)
                    with open(self.cache_file, 'w') as f:
                        json.dump(cache_copy, f)
                    return
                except Exception as e:
                    logging.error(f"Cache save error (attempt {attempt+1}): {e}")
                    if attempt == self.max_retries - 1:
                        logging.error("Max retries reached, cache not saved")
    
    def get_cached_response(self, prompt: str) -> Optional[str]:
        try:
            with self.lock:
                return self.cache.get(prompt)
        except Exception as e:
            logging.error(f"Cache read error: {e}")
            return None
    
    def cache_response(self, prompt: str, response: str):
        try:
            with self.lock:
                self.cache[prompt] = response
            self.save_cache()
        except Exception as e:
            logging.error(f"Cache write error: {e}")