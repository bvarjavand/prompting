from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""
    experiment_name: str
    save_dir: str
    batch_size: int = 10
    log_level: str = "INFO"
    save_responses: bool = True
    max_parallel_requests: int = 5
    cache_results: bool = True