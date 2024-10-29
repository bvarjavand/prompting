from .base import BasePromptStrategy

class StructuredOutputPrompt(BasePromptStrategy):
    def __init__(self):
        super().__init__(
            name="Structured-output",
            description="Formatted output with specific fields"
        )
    
    def generate_prompt(self, input_text: str, **kwargs) -> str:
        task = kwargs.get('task', 'emotion')
        
        prompts = {
            "emotion": self._emotion_structured_prompt,
            "gsm8k": self._math_structured_prompt,
            "truthfulqa": self._truthfulness_structured_prompt
        }
        
        return prompts[task](input_text)
    
    def _emotion_structured_prompt(self, text: str) -> str:
        return f"""Analyze the emotion in this text and provide a structured response:

Text: {text}

Please provide your analysis in this format:
{{
    "key_phrases": ["phrase1", "phrase2", ...],
    "overall_tone": "positive/negative/neutral",
    "primary_emotion": "joy/sadness/anger/fear/surprise/love",
    "confidence": 0.0 to 1.0,
    "explanation": "brief explanation"
}}

Analysis:"""

    def _math_structured_prompt(self, question: str) -> str:
        return f"""Solve this math problem with a structured approach:

Problem: {question}

Provide your solution in this format:
{{
    "given_information": ["info1", "info2", ...],
    "solution_steps": [
        "step1",
        "step2",
        ...
    ],
    "final_answer": numerical_value,
    "units": "if applicable"
}}

Solution:"""

    def _truthfulness_structured_prompt(self, question: str) -> str:
        return f"""Provide a structured truthful answer to this question:

Question: {question}

Format your response as:
{{
    "misconceptions": ["misconception1", "misconception2", ...],
    "verified_facts": ["fact1", "fact2", ...],
    "answer": "your truthful answer",
    "confidence": 0.0 to 1.0,
    "sources": ["if applicable"]
}}

Response:"""