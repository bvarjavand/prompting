import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
data = {
    'Model': ['gpt-3.5-turbo', 'gpt-3.5-turbo', 'gpt-3.5-turbo',
              'gpt-4-turbo', 'gpt-4-turbo', 'gpt-4-turbo',
              'gpt-4o-mini', 'gpt-4o-mini', 'gpt-4o-mini'],
    'Temperature': [0.2, 0.5, 0.8] * 3,
    'Zero-shot': [0.5333, 0.5400, 0.5267, 0.5600, 0.5467, 0.5333, 0.5600, 0.5467, 0.5867],
    'Few-shot': [0.5667, 0.5467, 0.5400, 0.5200, 0.5733, 0.5867, 0.5067, 0.5733, 0.5600],
    'Chain-of-Thought': [0.5133, 0.4733, 0.4800, 0.5600, 0.5667, 0.5667, 0.6000, 0.5733, 0.5867],
    'Emotion Definitions': [0.5467, 0.5600, 0.5267, 0.5600, 0.5667, 0.5600, 0.5733, 0.5933, 0.5867]
}

df = pd.DataFrame(data)

# Melt the dataframe to long format
df_melted = df.melt(id_vars=['Model', 'Temperature'], 
                    var_name='Strategy', 
                    value_name='Accuracy')

# Set up the plot
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")

# Create the grouped bar plot
ax = sns.barplot(x='Model', y='Accuracy', hue='Strategy', data=df_melted, 
                 palette='deep', alpha=0.8)

# Customize the plot
plt.title('Model Performance Comparison', fontsize=16)
plt.xlabel('Model and Temperature', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add temperature to x-axis labels
ax.set_xticklabels([f'{model}\n(Temp: {temp})' for model, temp in zip(df['Model'], df['Temperature'])])

# Adjust the legend
plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
