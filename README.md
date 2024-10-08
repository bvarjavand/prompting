## Testing different prompting techniques 
using openai api (can be applied to any models)

## Instructions
Prompts located in `prompts.py`, you can easily add additional prompts to this file
Evaluation harness can load any number of chosen prompt strategies in the init method.

## Example use case: (dair-ai/emotion dataset)
I've set up a quick example with classifying text as having emotions of either `{joy, sadness, love, surprise, anger, fear}`.
After running the script `python run.py` to sweep across a few models and temperature values, I get the following results.

Output from running on my machine (for the first pass)

