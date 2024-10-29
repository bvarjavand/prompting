from typing import Dict, Optional
from pathlib import Path
import json

class ResultCache:
    """Caches experiment results to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        cache_file = self.cache_dir / "response_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        with open(self.cache_dir / "response_cache.json", 'w') as f:
            json.dump(self.cache, f)
    
    def get_cached_response(self, prompt: str) -> Optional[str]:
        return self.cache.get(prompt)
    
    def cache_response(self, prompt: str, response: str):
        self.cache[prompt] = response
        self.save_cache()