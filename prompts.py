import pandas as pd

INSTRUCTION = """Classify the emotion in the following text delimited by triple backticks as either joy, sadness, anger, fear, surprise, or love."""

def zero_shot_prompt(text: str) -> str:
    return f"{INSTRUCTION}\nClassify this text: ```{text}```"

def few_shot_prompt(text: str, dataset, shots=3) -> str:
    composite = []
    for gid, g in dataset.groupby('emotion'):
        composite.append(g.sample(shots))
    examples = pd.concat(composite)
    examples = examples.sample(frac=1).to_dict('records')
    prompt = f"{INSTRUCTION}\n\nExamples:\n"
    for example in examples:
        prompt += f"Text: ```{example['text']}```\nEmotion: {example['emotion']}\n\n"
    prompt += f"Now classify this text: ```{text}```"
    return prompt

def chain_of_thought_prompt(text: str) -> str:
    return f"""{INSTRUCTION} Let's approach this step-by-step:

1. Read the text carefully.
2. Identify key words or phrases that indicate emotion.
3. Consider the overall tone and context of the message.
4. Match the identified emotional cues to the most appropriate emotion from the list.
5. Provide your final classification.

Based on this approach, classify this text: ```{text}```"""

def emotion_definition_prompt(text: str) -> str:
    definitions = {
        "joy": "a feeling of happiness, pleasure, or contentment",
        "sadness": "a feeling of sorrow, unhappiness, or depression",
        "anger": "a strong feeling of annoyance, displeasure, or hostility",
        "fear": "an unpleasant emotion caused by the threat of danger, pain, or harm",
        "surprise": "a feeling of mild astonishment or shock caused by something unexpected",
        "love": "a feeling of strong approval, often romantic or longing"
    }
    definitions_text = "\n".join([f"{emotion.capitalize()}: {definition}" for emotion, definition in definitions.items()])
    return f"""{INSTRUCTION} based on these definitions:

{definitions_text}

Text to classify: ```{text}```

The emotion expressed in this text is:"""

def persona_based_prompt(text: str) -> str:
    return f"""As an expert psychologist specializing in emotions, analyze and {INSTRUCTION.lower()}

Text: ```{text}```

Based on your expert knowledge, what is the primary emotion expressed in this text?"""

def multitask_prompt(text: str) -> str:
    return f"""Perform the following tasks for the given text delimited by triple backticks:

1. Identify key emotional words or phrases.
2. Determine the overall tone (positive, negative, or neutral).
3. Classify the primary emotion (joy, sadness, anger, fear, surprise, or love).
4. Explain your reasoning for the classification.

Text: ```{text}```

Provide your analysis in the following format:
Key emotional words/phrases:
Overall tone:
Reasoning:
Primary emotion:"""

def structured_output_prompt(text: str) -> str:
    return f"""Classify the primary emotion in the following text delimited by triple backticks. The possible emotions are joy, sadness, anger, fear, surprise, or love.

Text: ```{text}```

Provide your answer in the following JSON format:
{{
  "secondary_emotion": "emotion_name" (if applicable),
  "key_phrases": ["phrase1", "phrase2", ...],
  "confidence": 0.0 to 1.0,
  "primary_emotion": "emotion_name"
}}"""

def contrastive_prompt(text: str, emotions) -> str:
    selected_emotions = random.sample(emotions, 3)
    emotion_prompts = "\n".join([f"{emotion}:" for emotion in selected_emotions])
    return f"""Consider the following text delimited by triple backticks and compare it against these three emotions: {', '.join(selected_emotions)}. 

Text: ```{text}```

For each emotion, provide a brief argument for why it might be the primary emotion expressed in the text. Then, make a final decision on which emotion is most likely and explain why.

{emotion_prompts}

Final classification and explanation:"""
