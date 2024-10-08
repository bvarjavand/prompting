from eval_harness import EmotionClassificationHarness
import datasets
import os
import json
import openai
import pandas as pd
from pprint import pprint
import itertools

os.environ["OPENAI_API_KEY"] = '<your api key>'

class openai_llm:
    def __init__(self, model="gpt-3.5-turbo", temperature=None):
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        self.model = model
        self.temperature = temperature
    
    def generate(self, input):
        if type(input) is str:
            messages = [{'content': input, 'role': 'user'}]
        else:
            messages = input
        if self.temperature is None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )

        return response.choices[0].message.content

def prep_data():
    dataset = datasets.load_dataset("dair-ai/emotion")
    labelmap = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas()
    train_df['emotion'] = train_df.apply(lambda x: labelmap[x['label']], axis=1)
    train_df.drop('label', axis=1, inplace=True)
    val_df['emotion'] = val_df.apply(lambda x: labelmap[x['label']], axis=1)
    val_df.drop('label', axis=1, inplace=True)
    return train_df, val_df

def main(sweep):
    train_df, val_df = prep_data()

    # sweep across hparams
    for v in itertools.product(*sweep.values()):
        print(v)
        llm_model = openai_llm(v[0], v[1])
        harness = EmotionClassificationHarness(llm_model, train_df[:750])
        harness.run_optimization(max_iter=1)
        print('-'*10)

if __name__ == "__main__":
    sweep = {
        "model" : ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o-mini'],
        "temperature": [0.2, 0.5, 0.8]
    }
    main(sweep)
