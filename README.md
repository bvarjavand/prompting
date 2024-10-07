## Testing different prompting techniques 
using openai api (can be applied to any models)

## Instructions
Prompts located in `prompts.py`, you can easily add additional prompts to this file
Evaluation harness can load any number of chosen prompt strategies in the init method.

## Example use case: (dair-ai/emotion dataset)
```python
import json
import openai
import pandas as pd
from pprint import pprint
from emotion_classification_harness import EmotionClassificationHarness
import datasets
import os
os.environ["OPENAI_API_KEY"] = <your-openai-api-key>

dataset = datasets.load_dataset("dair-ai/emotion")
labelmap = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
train_df = dataset['train'].to_pandas()
train_df['emotion'] = train_df.apply(lambda x: labelmap[x['label']], axis=1)
train_df.drop('label', axis=1, inplace=True)

class openai_llm:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model
    
    def generate(self, input):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'content': input, 'role': 'user'}]
        )

        return response.choices[0].message.content
llm_model = openai_llm()

harness = EmotionClassificationHarness(llm_model, train_df[:500])
harness.run_optimization()
```

Output from running on my machine (for the first pass)
```
Starting strategy optimization...
Evaluating Zero-shot
Accuracy: 0.5600
Evaluating Few-shot
Accuracy: 0.5500
Evaluating Chain-of-Thought
Accuracy: 0.5400
Evaluating Emotion Definitions
Accuracy: 0.6000
...
```
