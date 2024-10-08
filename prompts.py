import pandas as pd
import random

INSTRUCTION = """Classify the emotion in the following text delimited by triple backticks as either joy, sadness, anger, fear, surprise, or love."""

def zero_shot_prompt(text: str) -> str:
    return f"{INSTRUCTION}\nClassify this text: ```{text}```"

def refined_emotion_prompt_v2(text: str) -> list:
    system_content = """You are an expert in emotion analysis. Your task is to classify the primary emotion in given texts as joy, sadness, anger, fear, surprise, or love. Pay close attention to subtle cues, context, and the overall emotional tone. Even if multiple emotions are present, identify the most dominant one.

Emotion definitions and key aspects:
- Joy: Feelings of happiness, contentment, or relief. Can be subtle, like appreciation of beauty or small victories. Key aspects: positivity, satisfaction, pleasure.
- Sadness: Feelings of sorrow, unhappiness, or melancholy. Can involve nostalgia or feeling unfulfilled. Key aspects: loss, disappointment, emptiness.
- Anger: Feelings of annoyance, frustration, or hostility. Can be expressed through dissatisfaction or resentment. Key aspects: irritation, indignation, feeling wronged.
- Fear: Feelings of apprehension, worry, or threat. Includes feeling vulnerable or unsafe. Key aspects: anxiety, insecurity, dread.
- Surprise: Feelings of astonishment or shock caused by unexpected events. Key aspects: amazement, disbelief.
- Love: Feelings of deep affection or attachment. Includes romantic love, platonic love, or general appreciation. Key aspects: warmth, fondness, care.

Remember:
1. Emotions can be mixed or complex. Focus on the primary, most dominant emotion.
2. Consider the overall context and tone, not just individual words.
3. Some emotions might seem similar (e.g., love and joy, or fear and sadness). Look for distinguishing features.

Examples:
1. Text: "I feel terrified, like I'm on the edge of a precipice."
   Emotion: Fear (expresses a clear sense of dread and insecurity)

2. Text: "I can finally stop feeling listless and like a waste of space."
   Emotion: Sadness (despite the change, it reflects a period of feeling unfulfilled)

3. Text: "I recall those high school feelings and the longing with which I watched the Olympic runners."
   Emotion: Love (expresses a deep fondness and admiration, even if tinged with nostalgia)

4. Text: "I find every body beautiful and only want people to feel vital in their bodies."
   Emotion: Joy (expresses appreciation and a positive outlook, rather than personal affection)

5. Text: "I love him but I feel threatened with him around a little."
   Emotion: Fear (despite expressing love, the dominant emotion is a sense of threat)

Classify the primary emotion in the given text, explaining your reasoning briefly. You must respond with a primary emotion of either joy, love, fear, anger, suprise, or sadness - not any other emotions."""

    user_content = f"Text: {text}\n\nWhat is the primary emotion expressed in this text? Explain your reasoning briefly, considering the overall context and dominant feeling."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def combined_prompt(text, dataset, shots=2):
    intro = """You are an expert in emotion analysis. Classify the primary emotion in the given text as joy, sadness, anger, fear, love, or surprise.
Emotion Categories:

Joy: Great pleasure, happiness, or contentment.
Sadness: Sorrow, unhappiness, or disappointment.
Anger: Annoyance, displeasure, or hostility.
Fear: Unpleasant emotion caused by perceived danger or threat.
Love: Strong affection, fondness, or romantic attraction.
Surprise: Astonishment or shock caused by something unexpected.

Examples:"""
    composite = []
    for gid, g in dataset.groupby('emotion'):
        composite.append(g.sample(shots))
    examples = pd.concat(composite)
    examples = examples.sample(frac=1).to_dict('records')
    for example in examples:
        intro = intro + "\n\nText: " + example['text']
        intro = intro + "\nEmotion: " + example['emotion']

    intro += """

Instructions:

1. Read the given text carefully.
2. Identify key phrases and context which indicate emotion.
3. Classify the primary emotion based on the six categories.

Input Format:
Text: [input_text]
Output Format:
Emotion: [classified_emotion]"""
    
    messages = [
        {'role': 'system', 'content': intro},
        {'role':'user', 'content':f"""Now, please classify the emotion in the following text: 
Text: {text}"""}
    ]
    return messages

def few_shot_prompt(text: str, dataset, shots=3) -> str:
    composite = []
    for gid, g in dataset.groupby('emotion'):
        composite.append(g.sample(shots))
    examples = pd.concat(composite)
    examples = examples.sample(frac=1).to_dict('records')
    prompt = [{'role':'system', 'content':f"{INSTRUCTION}"}]
    for example in examples:
        prompt += [{'role':'user', 'content':f"Classify this text: ```{example['text']}```"}]
        prompt += [{'role':'assistant', 'content':example['emotion']}]
    prompt += [{'role':'user', 'content':f"Classify this text: ```{text}```"}]
    return prompt

def chain_of_thought_prompt(text: str) -> str:
    prompt = [{'role':'system', 'content':f"""{INSTRUCTION} Let's approach this step-by-step:

1. Read the text carefully.
2. Identify key words or phrases that indicate emotion.
3. Consider the overall tone and context of the message.
4. Match the identified emotional cues to the most appropriate emotion from the list.
5. Provide your final classification."""}]
    prompt += [{'role':'user', 'content': f"Classify this text: ```{text}```"}]
    return prompt

def refined_emotion_prompt(text):
    system_content = """You are an expert in emotion analysis. Your task is to classify the primary emotion in given texts as joy, sadness, anger, fear, surprise, or love. Pay close attention to subtle cues and context. Even if multiple emotions are present, identify the most dominant one.

Emotion definitions:
- Joy: A feeling of happiness or contentment, even in challenging situations. Can involve restraint or understanding.
- Sadness: A feeling of sorrow or unhappiness. Can be expressed through words like "tears" or "devastating".
- Anger: A feeling of annoyance or hostility, which can be subtle and expressed through frustration or dissatisfaction.
- Fear: An unpleasant emotion caused by perceived danger or threat. Often involves words like "terrified" or "scared".
- Surprise: A feeling of astonishment caused by something unexpected.
- Love: A feeling of deep affection, which can be romantic or general appreciation for others.

Examples:
1. Text: "I didn't cry when they left because I understood they needed space."
   Emotion: Joy (shows restraint and understanding despite a potentially sad situation)

2. Text: "Everything feels dull and meaningless lately."
   Emotion: Sadness (expresses a lack of positive feelings)

3. Text: "I'm tired of always being overlooked and undervalued."
   Emotion: Anger (shows frustration and dissatisfaction)

4. Text: "The world seems so beautiful and full of possibilities."
   Emotion: Joy (expresses a positive outlook and appreciation)

Classify the primary emotion in the given text, explaining your reasoning briefly."""

    user_content = f"Text: {text}\n\nWhat is the primary emotion expressed in this text? Explain your reasoning briefly."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

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
    prompt = [{'role':'system', 'content':f"""{INSTRUCTION} based on these definitions:

{definitions_text}"""}]
    prompt += [{'role':'user', 'content': f"""Text to classify: ```{text}```
    
The emotion expressed in this text is:"""}]
    return prompt

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
