import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Tuple, Dict
import numpy as np

from prompts import *
import visualizations as vis

class PromptStrategy:
    def __init__(self, name: str, prompt_func: callable):
        self.name = name
        self.prompt_func = prompt_func
        self.strategies = [(self.name, self.prompt_func)]
    def __repr__(self):
        return self.name

class CombinedStrategy(PromptStrategy):
    def __init__(self, strategies: List[PromptStrategy]):
        super().__init__(f"Combined: {' + '.join([s.name for s in strategies])}", None)
        self.strategies = strategies

    def generate_prompt(self, text: str) -> str:
        return "\n\n".join([f"{s.name}:\n{s.prompt_func(text)}" for s in self.strategies])

class EmotionClassificationHarness:
    def __init__(self, llm_model, dataset):
        self.llm_model = llm_model
        self.dataset = dataset
        self.train_data_pd, self.test_data_pd = train_test_split(self.dataset, test_size=0.2, random_state=42)
        self.train_data = self.train_data_pd.to_dict('records')
        self.test_data = self.test_data_pd.to_dict('records')
        self.emotions = ["joy", "sadness", "anger", "fear", "surprise", "love"]
        self.strategies = self._initialize_strategies()
        self.optimization_history = []
        self.logs = {}

    def _initialize_strategies(self) -> List[PromptStrategy]:
        return [
            PromptStrategy("Zero-shot", zero_shot_prompt),
            PromptStrategy("Few-shot", lambda text: few_shot_prompt(text, self.train_data_pd)),
            PromptStrategy("Chain-of-Thought", chain_of_thought_prompt),
            PromptStrategy("Emotion Definitions", emotion_definition_prompt),
            PromptStrategy("Persona-based", persona_based_prompt),
            PromptStrategy("Multitask", multitask_prompt),
            PromptStrategy("Structured Output", structured_output_prompt),
            PromptStrategy("Contrastive", lambda text: contrastive_prompt(text, self.emotions))
        ]

    def _find_emotion_in_string(self, string):
        # gets the last mentioned emotion
        idxs = [string.lower().rfind(e) for e in self.emotions]
        if max(idxs) > -1:
            return self.emotions[idxs.index(max(idxs))]
        else:
            return "none"

    def evaluate_strategy(self, strategy: PromptStrategy, raw_out=False) -> Tuple[float, float, np.ndarray]:
        raw_predictions = []
        predictions = []
        true_labels = []

        print("Evaluating", strategy)
        for item in self.test_data:
            prompt = strategy.prompt_func(item['text']) if isinstance(strategy, PromptStrategy) else strategy.generate_prompt(item['text'])
            prediction = self.llm_model.generate(prompt)
            raw_predictions.append(prediction)
            predictions.append(self._find_emotion_in_string(prediction))
            true_labels.append(item['emotion'])

        accuracy = accuracy_score(true_labels, predictions)
        print("Accuracy:", f'{accuracy:.4f}')
        f1 = f1_score(true_labels, predictions, average='weighted')
        cm = confusion_matrix(true_labels, predictions, labels=self.emotions)

        if raw_out:
            return accuracy, f1, cm, raw_predictions
        else:
            return accuracy, f1, cm

    def first_order_evaluations(self):
        strats = self.strategies
        for s in strats:
            accuracy, f1, cm, raw_predictions = self.evaluate_strategy(s, self.test_data, raw_out=True)
            self.logs[s.name] = (accuracy, f1, cm, raw_predictions)


    def find_best_combination(self, max_iterations: int = 10, improvement_threshold: float = 0.001):
        best_strategy = None
        best_performance = 0

        for iteration in range(max_iterations):
            if best_strategy is None:
                performances = [(s, *self.evaluate_strategy(s, self.test_data)) for s in self.strategies]
                print(max(performances, key=lambda x: x[1]))
                best_strategy, best_performance = max(performances, key=lambda x: x[1])
                self.optimization_history.append((iteration, best_strategy, best_performance))
            else:
                unused_strategies = [s for s in self.strategies if s not in best_strategy.strategies]
                if not unused_strategies:
                    break

                candidate_strategies = [
                    CombinedStrategy(best_strategy.strategies + [s])
                    for s in unused_strategies
                ]

                performances = [(s, *self.evaluate_strategy(s, self.test_data)) for s in candidate_strategies]
                best_candidate, best_candidate_performance, _, _ = max(performances, key=lambda x: x[1])

                if best_candidate_performance > best_performance + improvement_threshold:
                    best_strategy = best_candidate
                    best_performance = best_candidate_performance
                    self.optimization_history.append((iteration, best_strategy, best_performance))
                else:
                    break

        return best_strategy, best_performance

    def run_optimization(self):
        print("Starting strategy optimization...")
        best_strategy, best_performance = self.find_best_combination()
        print(f"\nBest strategy combination found: {best_strategy.name}")
        print(f"Best performance (Accuracy): {best_performance:.4f}")

        final_accuracy, final_f1, final_cm = self.evaluate_strategy(best_strategy, self.dataset)
        print(f"\nFinal evaluation on entire dataset:")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"F1-score: {final_f1:.4f}")

        self.generate_visualizations(best_strategy, final_accuracy, final_f1, final_cm)

    def generate_visualizations(self, best_strategy, final_accuracy, final_f1, final_cm):
        vis.plot_optimization_history(self.optimization_history)
        individual_performances = [(s.name, self.evaluate_strategy(s, self.dataset)[0]) for s in self.strategies]
        best_combined = self.optimization_history[-1]
        vis.plot_strategy_comparison(individual_performances, best_combined)
        vis.plot_confusion_matrix(final_cm, best_strategy.name, self.emotions)
        vis.plot_emotion_performance(final_cm, best_strategy.name, self.emotions)
