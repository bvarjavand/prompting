import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_optimization_history(optimization_history):
    plt.figure(figsize=(12, 6))
    iterations, strategies, performances = zip(*optimization_history)
    plt.plot(iterations, performances, marker='o')
    plt.title('Optimization History')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    for i, (iteration, strategy, performance) in enumerate(optimization_history):
        plt.annotate(strategy.name, (iteration, performance), textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.show()

def plot_strategy_comparison(individual_performances, best_combined):
    names, accuracies = zip(*individual_performances)
    names = list(names) + [best_combined[1].name]
    accuracies = list(accuracies) + [best_combined[2]]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accuracies)
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, strategy_name, emotions):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title(f'Confusion Matrix - {strategy_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_emotion_performance(cm, strategy_name, emotions):
    emotion_accuracy = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, emotion_accuracy)
    plt.title(f'Emotion-wise Accuracy - {strategy_name}')
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
