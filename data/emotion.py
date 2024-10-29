from .base import BaseDataset, DatasetConfig
from typing import Dict, Any
import datasets
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class EmotionDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.emotion_map = {
            0: "sadness", 1: "joy", 2: "love", 
            3: "anger", 4: "fear", 5: "surprise"
        }
        self.train_data, self.test_data = self.load_data()

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source)
        train_df = dataset['train'].to_pandas()
        val_df = dataset['validation'].to_pandas()
        
        for df in [train_df, val_df]:
            df['emotion'] = df['label'].map(self.emotion_map)
            df.drop('label', axis=1, inplace=True)
        
        if self.config.sample_size:
            train_df = train_df.sample(n=min(self.config.sample_size, len(train_df)), random_state=42)
            val_df = val_df.sample(n=min(self.config.sample_size // 5, len(val_df)), random_state=42)
        
        return train_df, val_df

    def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
        pred_emotion = self._extract_emotion(response.lower())
        return {
            "accuracy": float(pred_emotion == target),
            "f1": f1_score([target], [pred_emotion], average='micro', 
                          labels=list(self.emotion_map.values()))
        }

    def _extract_emotion(self, text: str) -> str:
        for emotion in self.emotion_map.values():
            if emotion in text:
                return emotion
        return "none"