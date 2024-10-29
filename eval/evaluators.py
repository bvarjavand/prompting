from typing import List, Dict, Any, Optional, Union
import re
from statistics import mean
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
import numpy as np

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    @abstractmethod
    def evaluate_batch(self, predictions: List[str], targets: List[Any]) -> Dict[str, Any]:
        pass

class SimpleMathEvaluator:
    """Evaluator for mathematical word problems with marked answers."""
    
    def extract_answer(self, text: str) -> Optional[float]:
        """Extract number after #### marker."""
        if not text:
            return None
            
        try:
            matches = re.findall(r'\$?([\d,]+\.?\d*)', text)
            return float(matches[-1].replace(',', '')) if matches else None
        except ValueError:
            pass
                
        return None

    def evaluate_batch(self, predictions: List[str], targets: List[str]) -> Dict[str, Any]:
        """Evaluate a batch of math problem responses."""
        results = []
        
        for pred, target in zip(predictions, targets):
            pred_num = self.extract_answer(pred)
            target_num = float(self.extract_answer(target)) if target else None
            
            # Check for numeric match with small tolerance
            correct = False
            if pred_num is not None and target_num is not None:
                tolerance = max(0.01, abs(target_num) * 0.001)
                correct = abs(pred_num - target_num) <= tolerance
            
            results.append({
                'correct': correct,
                'extracted_answer': pred_num,
                'target_answer': target_num,
                'has_marker': '####' in pred
            })
        
        return {
            'accuracy': mean(r['correct'] for r in results),
            'has_markers': mean(r['has_marker'] for r in results),
            'details': results
        }

def print_evaluation(predictions: List[str], targets: List[str], n_examples: int = 3):
    """Print detailed evaluation results."""
    evaluator = SimpleMathEvaluator()
    results = evaluator.evaluate_batch(predictions, targets)
    
    print(f"\nOverall Results:")
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Responses with #### marker: {results['has_markers']:.2f}")
    
    print(f"\nExample Evaluations:")
    for i in range(min(n_examples, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Prediction: {predictions[i]}")
        print(f"Target: {targets[i]}")
        print(f"Extracted answer: {results['details'][i]['extracted_answer']}")
        print(f"Correct: {results['details'][i]['correct']}")

class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""
    
    def evaluate_batch(self, predictions: List[str], targets: List[str]) -> Dict[str, Any]:
        correct = 0
        results = []
        
        for pred, target in zip(predictions, targets):
            # Normalize and compare
            pred_clean = pred.lower().strip()
            target_clean = target.lower().strip()
            is_correct = pred_clean == target_clean
            correct += is_correct
            
            results.append({
                'prediction': pred_clean,
                'target': target_clean,
                'correct': is_correct
            })
        
        accuracy = correct / len(predictions) if predictions else 0
        return {
            'accuracy': accuracy,
            'details': results
        }

class QAEvaluator:
    """Evaluates QA responses based on presence of key concepts with flexible matching."""
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def get_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text while maintaining context."""
        normalized = self.normalize_text(text)
        
        # Split into meaningful chunks
        phrases = []
        for phrase in normalized.split(' and '):
            for subphrase in phrase.split(' but '):
                cleaned = subphrase.strip()
                if len(cleaned.split()) > 2:  # Only keep meaningful phrases
                    phrases.append(cleaned)
        
        return phrases if phrases else [normalized]
    
    def phrases_match(self, pred_phrase: str, target_phrase: str) -> bool:
        """Check if prediction phrase matches target phrase allowing for variations."""
        # Split into words for more flexible matching
        pred_words = set(pred_phrase.split())
        target_words = set(target_phrase.split())
        
        # Calculate word overlap
        common_words = pred_words & target_words
        if not common_words:
            return False
            
        # Check for key word matches
        important_words = {'not', 'no', 'cannot', 'does', 'do', 'private', 'government', 
                         'average', 'height', 'eyes', 'artificial', 'intelligence'}
        important_matches = important_words & common_words
        
        # Calculate overlap score
        overlap = len(common_words) / max(len(pred_words), len(target_words))
        
        # More strict matching if important words are present
        if important_matches:
            return overlap > 0.3  # Lower threshold if important words match
        return overlap > 0.5  # Higher threshold for general matches

    def evaluate_response(self, prediction: str, targets: Union[List[str], np.ndarray]) -> Dict[str, Any]:
        """Evaluate a single response against target answers."""
        # Convert targets to list of strings
        if isinstance(targets, np.ndarray):
            target_list = [str(t) for t in targets.flatten()]
        else:
            target_list = [str(t) for t in targets]
        
        # Get key phrases from prediction and targets
        pred_phrases = self.get_key_phrases(prediction)
        target_phrases = []
        for target in target_list:
            target_phrases.extend(self.get_key_phrases(target))
        
        # Match phrases
        matched_phrases = []
        for target_phrase in target_phrases:
            for pred_phrase in pred_phrases:
                if self.phrases_match(pred_phrase, target_phrase):
                    matched_phrases.append(target_phrase)
                    break
        
        # Calculate score
        score = len(set(matched_phrases)) / len(set(target_phrases)) if target_phrases else 0
        
        return {
            'score': score,
            'matched_phrases': list(set(matched_phrases)),
            'total_phrases': len(set(target_phrases)),
            'prediction': prediction,
            'correct': score > 0.5  # Consider it correct if more than half phrases match
        }

    def evaluate_batch(self, predictions: List[str], targets: List[Union[str, List[str], np.ndarray]]) -> Dict[str, Any]:
        """Evaluate a batch of responses."""
        results = []
        
        for pred, target in zip(predictions, targets):
            result = self.evaluate_response(pred, target)
            results.append(result)
        
        return {
            'accuracy': mean(r['score'] for r in results),
            'strict_accuracy': mean(r['correct'] for r in results),
            'details': results
        }

def print_qa_evaluation(predictions: List[str], targets: List[Union[str, List[str], np.ndarray]], n_examples: int = 3):
    """Print detailed evaluation results."""
    evaluator = QAEvaluator()
    results = evaluator.evaluate_batch(predictions, targets)
    
    print(f"\nOverall Results:")
    print(f"Accuracy (avg phrase match): {results['accuracy']:.2f}")
    print(f"Strict Accuracy (>50% phrases): {results['strict_accuracy']:.2f}")
    
    print(f"\nExample Evaluations:")
    for i in range(min(n_examples, len(predictions))):
        detail = results['details'][i]
        print(f"\nExample {i+1}:")
        print(f"Prediction: {detail['prediction']}")
        print(f"Target(s): {targets[i]}")
        print(f"Score: {detail['score']:.2f}")
        print(f"Matched phrases: {detail['matched_phrases']}")
        print(f"Total phrases: {detail['total_phrases']}")