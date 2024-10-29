import datasets
import pandas as pd
import re
import sqlparse
from typing import Dict, List, Any
from .base import BaseDataset, DatasetConfig

class EmotionDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.emotion_map = {
            0: "sadness", 1: "joy", 2: "love", 
            3: "anger", 4: "fear", 5: "surprise"
        }
        self.emotions = list(self.emotion_map.values())
        self.labels = list(self.emotion_map.values())
        self.train_data, self.test_data = self.load_data()
        print(f"\nEmotion Dataset loaded with {len(self.test_data)} test samples")
        self._print_distribution()

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source)
        train_df = dataset['train'].to_pandas()
        val_df = dataset['validation'].to_pandas()
        
        for df in [train_df, val_df]:
            df['emotion'] = df['label'].map(self.emotion_map)
            df.drop('label', axis=1, inplace=True)
        
        if self.config.sample_size:
            train_df = self._sample_balanced(train_df, self.config.sample_size)
            val_df = self._sample_balanced(val_df, self.config.sample_size // 5)
        
        return train_df.to_dict('records'), val_df.to_dict('records')

    def _sample_balanced(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        samples_per_label = max(1, n_samples // len(self.labels))
        sampled = []
        for emotion in self.labels:
            emotion_data = df[df['emotion'] == emotion]
            if len(emotion_data) > 0:
                sampled.append(emotion_data.sample(
                    n=min(samples_per_label, len(emotion_data)),
                    random_state=42
                ))
        return pd.concat(sampled).reset_index(drop=True)

    def _print_distribution(self):
        emotions = [item['emotion'] for item in self.test_data]
        print("\nEmotion Distribution in test set:")
        for emotion in self.labels:
            count = emotions.count(emotion)
            print(f"{emotion}: {count} samples")

    def _extract_emotion(self, text: str) -> str:
        """Extract emotion from model response."""
        text = text.lower().strip()
        
        # First check for exact matches
        if text in self.emotion_map.values():
            return text
        
        # Common patterns in responses
        patterns = [
            r"emotion:?\s*(\w+)",
            r"primary emotion:?\s*(\w+)",
            r"the emotion is:?\s*(\w+)",
            r"^(\w+)$"  # Single word response
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                emotion = matches[0].strip().lower()
                if emotion in self.emotion_map.values():
                    return emotion
        
        # Look for emotion words in the text
        for emotion in self.emotion_map.values():
            if emotion in text:
                return emotion
        
        return "none"

    def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
        pred = response.lower().strip()
        for emotion in self.labels:
            if emotion in pred:
                pred = emotion
                break
        return {
            "accuracy": float(pred == target)
        }

class GSM8KMathDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.train_data, self.test_data = self.load_data()
        print(f"\nGSM8K Dataset loaded with {len(self.test_data)} test samples")

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source, "main")
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        if self.config.sample_size:
            train_df = train_df.sample(n=min(self.config.sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(n=min(self.config.sample_size // 5, len(test_df)), random_state=42)
        
        return train_df.to_dict('records'), test_df.to_dict('records')

    def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
        def extract_number(text):
            matches = re.findall(r'\$?([\d,]+\.?\d*)', text)
            return float(matches[-1].replace(',', '')) if matches else None
            # numbers = re.findall(r'-?\d*\.?\d+', text)
            # return float(numbers[-1]) if numbers else None

        pred_num = extract_number(response)
        true_num = extract_number(target)
        
        if pred_num is None or true_num is None:
            return {"accuracy": 0.0}
        
        return {
            "accuracy": float(abs(pred_num - true_num) < 1e-6)
        }

class TruthfulQADataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.train_data, self.test_data = self.load_data()
        print(f"\nTruthfulQA Dataset loaded with {len(self.test_data)} test samples")

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source, 'generation')
        df = dataset['validation'].to_pandas()
        
        if self.config.sample_size:
            df = df.sample(n=min(self.config.sample_size, len(df)), random_state=42)
        
        # Split into train/test
        train_size = int(0.8 * len(df))
        return (df.iloc[:train_size].to_dict('records'), 
                df.iloc[train_size:].to_dict('records'))

    def evaluate_response(self, response: str, target: Dict) -> Dict[str, float]:
        response = response.lower()
        correct_answers = [ans.lower() for ans in target['correct_answers']]
        return {
            "accuracy": float(any(ans in response for ans in correct_answers))
        }

class SQLDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.train_data, self.test_data = self.load_data()
        print(f"\nSQL Dataset loaded with {len(self.test_data)} test samples")

    def load_data(self):
        dataset = datasets.load_dataset(self.config.source)
        # Create copies to avoid the SettingWithCopyWarning
        train_df = dataset['train'].to_pandas().copy()
        test_df = dataset['test'].to_pandas().copy()
        
        # Sample first to reduce memory usage
        if self.config.sample_size:
            train_df = self._sample_diverse(train_df, self.config.sample_size)
            test_df = self._sample_diverse(test_df, min(20, self.config.sample_size // 5))  # Reduced test size
        
        # Process the dataframes
        for df in [train_df, test_df]:
            # Format SQL queries
            df.loc[:, 'sql'] = df['sql'].apply(self._format_sql)
            
            # Create combined text field
            df.loc[:, 'text'] = df.apply(
                lambda x: (f"Question: {x['sql_prompt']}\n\n"
                         f"Schema:\n{x['sql_context']}"), 
                axis=1
            )
        
        return train_df.to_dict('records'), test_df.to_dict('records')

    def _format_sql(self, sql: str) -> str:
        return sqlparse.format(sql, reindent=True, keyword_case='upper').strip()

    def _sample_diverse(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        # Get unique domains and calculate samples per domain
        domains = df['domain'].unique()
        samples_per_domain = max(1, n_samples // len(domains))
        
        sampled_dfs = []
        for domain in domains[:len(domains)//6]:
            domain_df = df[df['domain'] == domain].copy()
            if len(domain_df) > 0:
                sampled = domain_df.sample(
                    n=min(len(domain_df), samples_per_domain),
                    random_state=42
                )
                sampled_dfs.append(sampled)
        
        # Combine all sampled dataframes
        final_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the final dataset
        return final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
        def clean_sql(sql):
            # Remove markdown SQL code blocks if present
            sql = re.sub(r'```sql\n?(.*?)\n?```', r'\1', sql, flags=re.DOTALL)
            # Format SQL for comparison
            sql = sqlparse.format(
                sql,
                reindent=True,
                keyword_case='upper'
            ).strip()
            # Remove extra whitespace
            return ' '.join(sql.split())
        
        pred_sql = clean_sql(response)
        target_sql = clean_sql(target)
        
        return {
            "exact_match": float(pred_sql == target_sql)
        }

# class EmotionDataset(BaseDataset):
#     def __init__(self, config: DatasetConfig):
#         super().__init__(config)
#         self.emotion_map = {
#             0: "sadness", 1: "joy", 2: "love", 
#             3: "anger", 4: "fear", 5: "surprise"
#         }
#         self.emotions = list(self.emotion_map.values())
#         self.train_data, self.test_data = self.load_data()
        
#         # Print dataset statistics after loading
#         print("\nDataset Statistics:")
#         print(f"Training samples: {len(self.train_data)}")
#         print(f"Test samples: {len(self.test_data)}")
#         print("\nEmotion distribution in test set:")
#         self._print_distribution(self.test_data)

#     def _print_distribution(self, data):
#         """Print emotion distribution in dataset."""
#         emotion_counts = {}
#         for item in data:
#             emotion = item['emotion']
#             emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
#         for emotion in self.emotions:
#             count = emotion_counts.get(emotion, 0)
#             print(f"{emotion}: {count} samples")

#     def load_data(self):
#         """Load and prepare the dataset with proper sampling."""
#         dataset = datasets.load_dataset("dair-ai/emotion")
#         train_df = dataset['train'].to_pandas()
#         val_df = dataset['validation'].to_pandas()
        
#         # Map emotions
#         for df in [train_df, val_df]:
#             df['emotion'] = df['label'].map(self.emotion_map)
#             df.drop('label', axis=1, inplace=True)
        
#         if self.config.sample_size:
#             # Calculate samples per emotion
#             train_samples_per_emotion = max(1, self.config.sample_size // len(self.emotions))
#             test_samples_per_emotion = max(1, train_samples_per_emotion // 5)
            
#             # Sample with stratification
#             train_sampled = []
#             test_sampled = []
            
#             for emotion in self.emotions:
#                 # Sample training data
#                 emotion_data = train_df[train_df['emotion'] == emotion]
#                 if len(emotion_data) > 0:
#                     sampled = emotion_data.sample(
#                         n=min(train_samples_per_emotion, len(emotion_data)),
#                         random_state=42
#                     )
#                     train_sampled.append(sampled)
                
#                 # Sample test data
#                 emotion_data = val_df[val_df['emotion'] == emotion]
#                 if len(emotion_data) > 0:
#                     sampled = emotion_data.sample(
#                         n=min(test_samples_per_emotion, len(emotion_data)),
#                         random_state=42
#                     )
#                     test_sampled.append(sampled)
            
#             train_df = pd.concat(train_sampled).reset_index(drop=True)
#             val_df = pd.concat(test_sampled).reset_index(drop=True)
        
#         return train_df.to_dict('records'), val_df.to_dict('records')

#     def evaluate_response(self, response: str, target: str) -> Dict[str, float]:
#         """Evaluate a single response against its target."""
#         pred_emotion = self._extract_emotion(response.lower())
#         correct = pred_emotion == target
        
#         # For single examples, accuracy and F1 are the same
#         return {
#             "accuracy": float(correct),
#             "f1": float(correct)
#         }

#     def batch_evaluate(self, responses: List[str], targets: List[str]) -> Dict[str, float]:
#         """Evaluate a batch of responses."""
#         predictions = [self._extract_emotion(r.lower()) for r in responses]
        
#         # Calculate metrics
#         return {
#             "accuracy": accuracy_score(targets, predictions),
#             "f1": f1_score(targets, predictions, average='weighted',
#                           labels=self.emotions, zero_division=0)
#         }

#     def _extract_emotion(self, text: str) -> str:
#         """Extract emotion from model response."""
#         text = text.lower().strip()
        
#         # First check for exact matches
#         if text in self.emotions:
#             return text
        
#         # Common patterns in responses
#         patterns = [
#             r"emotion:?\s*(\w+)",
#             r"primary emotion:?\s*(\w+)",
#             r"the emotion is:?\s*(\w+)",
#             r"^(\w+)$"  # Single word response
#         ]
        
#         import re
#         for pattern in patterns:
#             matches = re.findall(pattern, text)
#             if matches:
#                 emotion = matches[0].strip().lower()
#                 if emotion in self.emotions:
#                     return emotion
        
#         # Look for emotion words in the text
#         for emotion in self.emotions:
#             if emotion in text:
#                 return emotion
        
#         return "none"
    