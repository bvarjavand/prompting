from .base import BasePromptStrategy

class ChainOfThoughtPrompt(BasePromptStrategy):
    def __init__(self):
        super().__init__(
            name="Chain-of-thought",
            description="Explicit step-by-step reasoning"
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        task = kwargs.get('task', 'emotion')
        
        prompts = {
            "emotion": self._emotion_cot_prompt,
            "gsm8k": self._math_cot_prompt,
            "truthfulqa": self._truthfulness_cot_prompt
        }
        
        return prompts[task](input_text)
    
    def _emotion_cot_prompt(self, text: str) -> str:
        return f"""Let's analyze the emotion in this text step by step:

Text: {text}

1. First, let's identify key emotional words and phrases
2. Then, consider the overall tone
3. Next, examine the context and situation
4. Finally, determine the primary emotion (joy, sadness, anger, fear, surprise, or love)

Analysis:"""

    def _math_cot_prompt(self, question: str) -> str:
        return f"""Let's solve this math problem step by step:

Problem: {question}

Let's approach this:
1) First, let's identify the key information
2) Then, break down the problem into steps
3) Solve each step
4) Finally, provide the numerical answer

Solution:"""

    def _truthfulness_cot_prompt(self, question: str) -> str:
        return f"""Let's answer this question truthfully using a step-by-step approach:

Question: {question}

1) First, let's identify any assumptions or potential misconceptions
2) Then, consider verified facts and evidence
3) Next, evaluate different perspectives
4) Finally, formulate a truthful answer

Analysis:"""