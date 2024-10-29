PROMPT_TEMPLATES = {
    "emotion": {
        "zero-shot": """Classify the emotion in this text by choosing EXACTLY ONE of these emotions: joy, sadness, anger, fear, surprise, or love.
Respond with just the emotion word.

Text: {text}

Emotion:""",

        "detailed": """Analyze the emotion in this text carefully. Choose EXACTLY ONE emotion from this list:
- joy: happiness, pleasure, excitement
- sadness: disappointment, sorrow, feeling down
- anger: frustration, irritation, annoyance
- fear: worry, anxiety, uncertainty
- surprise: astonishment, shock, unexpected reaction
- love: deep appreciation, caring, passionate interest

Text: {text}

Respond with just the single emotion word (e.g., "joy", "sadness", etc.).

Emotion:""",

        "structured": """Analyze this text step by step:
1. Identify emotional words and tone
2. Consider the overall context
3. Choose EXACTLY ONE emotion from: joy, sadness, anger, fear, surprise, love

Text: {text}

Respond with only the emotion word.

Emotion:"""
    },
    
    "math": {
        "zero-shot": """Solve this math problem and provide the final numerical answer after #### marker.

Problem: {text}

Final answer:""",

        "detailed": """Let's solve this math problem carefully. Show your work and provide the final numerical answer after #### marker.

Problem: {text}

Solution:""",

        "structured": """Let's solve this step by step:
1. Identify the key information
2. Break down the calculations needed
3. Solve each step
4. Provide the final numerical answer after #### marker

Problem: {text}

Let's solve:"""
    },
    
    "truthful": {
        "zero-shot": """Answer this question truthfully, based only on verified facts.

Question: {text}

Answer:""",

        "detailed": """Provide a truthful answer to this question. If you're unsure about something, say so rather than making assumptions.
Base your answer only on verified facts.

Question: {text}

Truthful answer:""",

        "structured": """Answer this question step by step:
1. Identify any potential misconceptions
2. Consider verified facts only
3. Provide a truthful answer

Question: {text}

Answer:"""
    },
    
    "sql": {
        "zero-shot": """Write a SQL query to answer this question.

{text}

SQL query:""",

        "detailed": """Write a SQL query to answer this question. The query should be clear and efficient.

{text}

Write the SQL query in this format:
```sql
YOUR QUERY HERE
```""",

        "structured": """Let's write a SQL query step by step:
1. SELECT: Identify the columns needed
2. FROM/JOIN: Determine the required tables
3. WHERE: Add necessary conditions
4. GROUP/ORDER: Add any grouping or sorting

{text}

Write the SQL query:
```sql
```"""
    }
}