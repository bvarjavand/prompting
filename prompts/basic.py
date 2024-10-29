from .base import BasePromptStrategy
from typing import List

class ZeroShotPrompt(BasePromptStrategy):
    def __init__(self):
        super().__init__(
            name="Zero-shot",
            description="Direct instruction without examples"
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        task_instructions = {
            "emotion": "Classify the emotion in the following text as either joy, sadness, anger, fear, surprise, or love.",
            "gsm8k": "Solve this math problem and provide the final numerical answer.",
            "truthfulqa": "Provide a truthful answer to this question based on verified facts."
        }
        
        task = kwargs.get('task', 'emotion')
        return f"{task_instructions[task]}\n\nText: {input_text}\n\nAnswer:"

class FewShotPrompt(BasePromptStrategy):
    def __init__(self, n_shots: int = 3):
        super().__init__(
            name=f"Few-shot-{n_shots}",
            description=f"Instruction with {n_shots} examples"
        )
        self.n_shots = n_shots
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        examples = kwargs.get('examples', [])
        task = kwargs.get('task', 'emotion')
        
        prompt = f"Task: {self._get_task_description(task)}\n\nExamples:\n\n"
        
        for example in examples[:self.n_shots]:
            prompt += self._format_example(example, task)
        
        prompt += f"\nNow classify this:\n{input_text}\n\nAnswer:"
        return prompt
    
    def _get_task_description(self, task: str) -> str:
        descriptions = {
            "emotion": "Classify the emotion as joy, sadness, anger, fear, surprise, or love.",
            "gsm8k": "Solve math problems and provide the final numerical answer.",
            "truthfulqa": "Provide truthful answers based on verified facts."
        }
        return descriptions.get(task, "")
    
    def _format_example(self, example: dict, task: str) -> str:
        if task == "gsm8k":
            return f"Problem: {example['question']}\nAnswer: {example['answer']}\n\n"
        elif task == "emotion":
            return f"Text: {example['text']}\nEmotion: {example['emotion']}\n\n"
        else:
            return f"Q: {example['question']}\nA: {example['correct_answers'].split('|')[0]}\n\n"