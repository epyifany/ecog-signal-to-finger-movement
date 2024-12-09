import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
data = pd.read_csv('results.csv')

# Create figure for average performance across subjects
plt.figure(figsize=(10, 6))
model_means = data.groupby('Model')['Avg Corr'].mean().sort_values(ascending=False)
sns.barplot(x=model_means.index, y=model_means.values)
plt.xticks(rotation=45)
plt.title('Average Correlation Coefficient Across All Subjects and Lags')
plt.tight_layout()
plt.savefig('model_comparison.pdf')
plt.close()

# Create lag analysis plot
plt.figure(figsize=(12, 6))
for model in data['Model'].unique():
    model_data = data[data['Model'] == model]
    plt.plot(model_data['Lag'], model_data['Avg Corr'], label=model, marker='o')
plt.xlabel('Lag (ms)')
plt.ylabel('Average Correlation')
plt.title('Model Performance vs. Lag Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lag_analysis.pdf')
plt.close()

# Create subject-wise comparison
for subject in data['Patient'].unique():
    subject_data = data[data['Patient'] == subject]
    best_results = subject_data.groupby('Model')['Avg Corr'].mean().sort_values(ascending=False)
    print(f"\n{subject} Best Results:")
    print(best_results)