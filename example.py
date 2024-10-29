import os
from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Any
from runner.config import ExperimentConfig
from runner.tracker import ExperimentTracker
from runner.processor import BatchProcessor
from runner.visualizer import ExperimentVisualizer
from llm.openai_llm import OpenAILLM
from data.base import DatasetConfig
from data.datasets import EmotionDataset, GSM8KMathDataset, TruthfulQADataset, SQLDataset
import matplotlib
from sklearn.metrics import accuracy_score, f1_score
from analysis import ResultsAnalyzer
from prompts.strategies import PROMPT_TEMPLATES
from prompts.base import SimplePromptStrategy
from eval import EvaluatorFactory
import logging, time
import traceback

matplotlib.use('Agg')

def evaluate_dataset(dataset: Any, responses: List[str], dataset_type: str) -> Dict[str, Any]:
    """Evaluate responses for a specific dataset."""
    evaluator = EvaluatorFactory.create_evaluator(dataset_type)
    targets = [item[dataset.config.target_column] for item in dataset.test_data]
    
    results = evaluator.evaluate_batch(responses, targets)
    
    # Print some example evaluations
    print("\nExample Evaluations:")
    for i in range(min(3, len(responses))):
        print(f"\nExample {i+1}:")
        print(f"Input: {dataset.test_data[i][dataset.config.input_column]}")
        print(f"Prediction: {responses[i]}")
        print(f"Target: {targets[i]}")
        print(f"Details: {results['details'][i]}")
    
    return results

# class SimplePromptStrategy:
#     def __init__(self, name: str, template: str):
#         self.name = name
#         self.template = template
    
#     def generate_prompt(self, text: str, **kwargs) -> str:
#         if 'theme_hint' in kwargs and '{theme_hint}' in self.template:
#             return self.template.format(text=text, theme_hint=kwargs['theme_hint'])
#         return self.template.format(text=text)

# def get_theme_hint(text):
#     if 'feel like' in text.lower() or 'passionate' in text.lower():
#         return "Notice expressions of passion or appreciation vs general happiness"
#     elif 'agitated' in text.lower() or 'anxiety' in text.lower():
#         return "Notice expressions of worry/anxiety vs frustration"
#     elif 'awful' in text.lower() or 'wasted' in text.lower():
#         return "Notice expressions of disappointment vs anxiety"
#     else:
#         return "Consider the depth and nature of the emotional expression"


def main():
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    # Add logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="quick_test",
        save_dir="quick_test_results",
        batch_size=5,
        max_parallel_requests=2,
        cache_results=True
    )
    
    # Initialize LLM
    llm = OpenAILLM(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=200
    )
    
    # Initialize dataset
    dataset_configs = {
        "emotion": {
            "config": DatasetConfig(
            name="emotion",
            source="dair-ai/emotion",
            input_column="text",
            target_column="emotion",
            metric_names=["accuracy"],
                sample_size=600
            ),
            "type": "classification"
        },
        "math": {
            "config": DatasetConfig(
                name="math",
                source="gsm8k",
                input_column="question",
                target_column="answer",
                metric_names=["accuracy"],
                sample_size=600
            ),
            "type": "math"
        },
        "truthful": {
            "config": DatasetConfig(
            name="truthful_qa",
            source="truthfulqa/truthful_qa",
            input_column="question",
            target_column="correct_answers",
            metric_names=["accuracy"],
                sample_size=600
            ),
            "type": "qa"
        },
        # "sql": DatasetConfig(
        #     name="sql",
        #     source="gretelai/synthetic_text_to_sql",
        #     input_column="text",
        #     target_column="sql",
        #     metric_names=["exact_match"],
        #     sample_size=60
        # )
    }

    datasets = {}
    for key, cfg in dataset_configs.items():
        if key == "emotion":
            datasets[key] = EmotionDataset(cfg["config"])
        elif key == "math":
            datasets[key] = GSM8KMathDataset(cfg["config"])
        elif key == "truthful":
            datasets[key] = TruthfulQADataset(cfg["config"])
        elif key == "sql":
            datasets[key] = SQLDataset(cfg["config"])
    
    # Initialize runner components
    tracker = ExperimentTracker(config)
    processor = BatchProcessor(llm, config)
    visualizer = ExperimentVisualizer(config.save_dir)

 
#     # Define prompt strategies
#     strategies = [
#         SimplePromptStrategy(
#             name="zero-shot",
#             template="""Classify the primary emotion in this text, choosing EXACTLY ONE of these emotions:
# - joy (happiness, excitement)
# - sadness (sorrow, disappointment)
# - anger (frustration, irritation)
# - fear (anxiety, worry, uncertainty)
# - surprise (astonishment, shock)
# - love (affection, deep appreciation, caring)

# Important: 
# - If someone expresses passion or deep appreciation for something, that's love, not joy
# - If someone expresses worry or anxiety, that's fear, not anger
# - Choose the most fundamental emotion, not its manifestation

# Text: {text}

# Emotion:"""
#         ),
#         SimplePromptStrategy(
#             name="detailed",
#             template="""Carefully distinguish between similar emotions in this text. Choose EXACTLY ONE emotion from these options:

# LOVE vs JOY:
# - Love: Deep affection, caring about something/someone, passionate appreciation
# - Joy: General happiness, pleasure, excitement without deep attachment

# FEAR vs ANGER:
# - Fear: Worry, anxiety, uncertainty about outcomes
# - Anger: Frustration, irritation, feeling wronged

# SADNESS vs FEAR:
# - Sadness: Disappointment, feeling down, loss
# - Fear: Concern about future events, anxiety

# SURPRISE remains distinct as sudden astonishment or shock.

# Text: {text}

# First identify the emotional theme: {theme_hint}

# The single primary emotion is:"""
#         ),
#         SimplePromptStrategy(
#             name="structured",
#             template="""Analyze this text step by step to identify the primary emotion:

# Text: {text}

# Step 1: Key Questions
# - Is there deep appreciation/caring (love) or just happiness (joy)?
# - Is there anxiety/worry (fear) or frustration/irritation (anger)?
# - Is there disappointment/loss (sadness) or concern about future (fear)?
# - Is there sudden astonishment (surprise)?

# Step 2: Emotion Definitions
# love = deep appreciation, passion, caring
# joy = general happiness, excitement
# fear = worry, anxiety, uncertainty
# anger = frustration, irritation
# sadness = disappointment, loss
# surprise = astonishment, shock

# Your task: Output ONLY ONE of these exact words: love, joy, fear, anger, sadness, surprise

# Emotion:"""
#         )
#     ]
# Process each dataset

    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")

        # Process each strategy for this dataset
        for strategy_name in PROMPT_TEMPLATES[dataset_name]:
            print(f"\nTesting strategy: {strategy_name}")
            
            strategy = SimplePromptStrategy(
                f"{dataset_name}-{strategy_name}",
                PROMPT_TEMPLATES[dataset_name][strategy_name]
            )
            
            # Process responses
            responses = processor.process_batch(
                dataset.test_data,
                strategy,
                input_column=dataset.config.input_column
            )
            
            # Evaluate responses
            try:
                results = evaluate_dataset(
                    dataset, 
                    responses, 
                    dataset_configs[dataset_name]["type"]
                )
                
                # Store results
                tracker.add_result(dataset_name, strategy.name, {
                    'metrics': results,
                    'responses': responses if config.save_responses else None
                })
                
            except Exception as e:
                logging.error(f"Evaluation error: {e}")
                logging.error(traceback.format_exc())
    
    # Save results and generate visualizations
    results_path = tracker.save_results()
    visualizer.generate_visualizations(tracker.results)
    print(f"\nResults saved to: {results_path}")
        
        # # Print detailed analysis
        # analyzer.print_detailed_analysis()
        
        # # Plot confusion matrices
        # analyzer.plot_confusion_matrices(save_dir="quick_test_results/visualizations")
        
        # # Save results and generate visualizations
        # results_path = tracker.save_results()
        # visualizer.generate_visualizations(tracker.results)
        # print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("quick_test_results", exist_ok=True)
    os.makedirs("quick_test_results/logs", exist_ok=True)
    os.makedirs("quick_test_results/visualizations", exist_ok=True)
    
    main()