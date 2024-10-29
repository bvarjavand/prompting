import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ResultsAnalyzer:
    def __init__(self, dataset, results_path: str = None):
        self.dataset = dataset
        self.results = {}
        if results_path:
            self.load_results(results_path)
    
    def analyze_strategy(self, 
                        strategy_name: str, 
                        responses: List[str], 
                        true_labels: List[str]):
        """Analyze performance of a specific strategy."""
        predictions = [self.dataset._extract_emotion(r) for r in responses]
        
        # Create confusion matrix
        cm = confusion_matrix(
            true_labels, 
            predictions, 
            labels=self.dataset.emotions
        )
        
        # Calculate per-emotion metrics
        per_emotion_metrics = {}
        for i, emotion in enumerate(self.dataset.emotions):
            true_pos = cm[i, i]
            false_pos = cm[:, i].sum() - true_pos
            false_neg = cm[i, :].sum() - true_pos
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_emotion_metrics[emotion] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Analyze error patterns
        error_patterns = []
        for i, true_emotion in enumerate(self.dataset.emotions):
            for j, pred_emotion in enumerate(self.dataset.emotions):
                if i != j and cm[i, j] > 0:
                    error_patterns.append({
                        'true': true_emotion,
                        'predicted': pred_emotion,
                        'count': cm[i, j]
                    })
        
        return {
            'confusion_matrix': cm,
            'per_emotion_metrics': per_emotion_metrics,
            'error_patterns': sorted(error_patterns, key=lambda x: x['count'], reverse=True)
        }
    
    def plot_confusion_matrices(self, save_dir: str = None):
        """Plot confusion matrices for all strategies."""
        for strategy_name, results in self.results.items():
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                results['confusion_matrix'],
                annot=True,
                fmt='d',
                xticklabels=self.dataset.emotions,
                yticklabels=self.dataset.emotions
            )
            plt.title(f'Confusion Matrix - {strategy_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            if save_dir:
                plt.savefig(f"{save_dir}/confusion_matrix_{strategy_name}.png",
                           bbox_inches='tight')
            plt.close()
    
    def print_detailed_analysis(self):
        """Print detailed analysis for all strategies."""
        for strategy_name, results in self.results.items():
            print(f"\nDetailed Analysis for {strategy_name}")
            print("=" * 50)
            
            print("\nPer-emotion Performance:")
            metrics_df = pd.DataFrame(results['per_emotion_metrics']).T
            print(metrics_df.round(3))
            
            print("\nTop Error Patterns:")
            for error in results['error_patterns'][:5]:
                print(f"{error['true']} → {error['predicted']}: {error['count']} times")
            
            print("\nStrengths:")
            top_emotions = metrics_df.sort_values('f1', ascending=False).head(2)
            for emotion in top_emotions.index:
                print(f"- Good at identifying {emotion} "
                      f"(F1: {top_emotions.loc[emotion, 'f1']:.3f})")
            
            print("\nWeaknesses:")
            bottom_emotions = metrics_df.sort_values('f1').head(2)
            for emotion in bottom_emotions.index:
                print(f"- Struggles with {emotion} "
                      f"(F1: {bottom_emotions.loc[emotion, 'f1']:.3f})")
            
            print("-" * 50)
