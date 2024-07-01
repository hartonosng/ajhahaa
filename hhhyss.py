import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import KBinsDiscretizer

# Sample data
data = {
    'numeric_feature1': [100, 200, 150, 250, 300, 350, 400, 450],
    'numeric_feature2': [10, 20, 15, 25, 30, 35, 40, 45],
    'categorical_feature1': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'categorical_feature2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
    'categorical_feature3': ['M', 'N', 'M', 'N', 'M', 'N', 'M', 'N'],
    'label': ['good', 'bad', 'good', 'bad', 'bad', 'good', 'bad', 'good']
}

df = pd.DataFrame(data)

# Convert label to binary for calculation
df['label'] = df['label'].map({'good': 0, 'bad': 1})

# List of features to analyze
numeric_features = ['numeric_feature1', 'numeric_feature2']
categorical_features = ['categorical_feature1', 'categorical_feature2', 'categorical_feature3']

# Bin numeric features and get bin intervals
def bin_numeric_features(df, numeric_features, bins=2, strategy='uniform'):
    bin_intervals = {}
    for feature in numeric_features:
        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
        df[feature + '_binned'] = est.fit_transform(df[[feature]]).astype(int)
        bin_edges = est.bin_edges_[0]
        bin_intervals[feature] = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges) - 1)]
    return df, bin_intervals

df, bin_intervals = bin_numeric_features(df, numeric_features)

# Combine binned numeric and categorical features
all_features = [feature + '_binned' for feature in numeric_features] + categorical_features

# Function to calculate recall and precision for each feature based on frequency
def calculate_metrics(df, features, label):
    results = {}
    crosstabs = {}
    
    for feature in features:
        # Create crosstab with frequency
        crosstab_freq = pd.crosstab(df[feature], df[label], margins=True, margins_name='Total')
        
        # Calculate recall and precision based on frequency
        if 'Total' in crosstab_freq.index:
            total_instances = crosstab_freq.loc['Total', 'Total']
            true_positives = crosstab_freq.loc[1, 1] if 1 in crosstab_freq.index else 0  # Predicted 1 (bad) and actually 1 (bad)
            false_negatives = crosstab_freq.loc[1, 0] if 1 in crosstab_freq.index else 0  # Predicted 1 (bad) but actually 0 (good)
            false_positives = crosstab_freq.loc[0, 1] if 0 in crosstab_freq.index else 0  # Predicted 0 (good) but actually 1 (bad)
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0
            
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0.0
        else:
            recall = 0.0
            precision = 0.0
        
        # Store results
        results[feature] = {'recall': recall, 'precision': precision}
        crosstabs[feature] = crosstab_freq
        
    return results, crosstabs

# Calculate metrics and get crosstabs
metrics, crosstabs = calculate_metrics(df, all_features, 'label')

# Display bin intervals
for feature, intervals in bin_intervals.items():
    print(f"Bin intervals for {feature}:")
    for i, interval in enumerate(intervals):
        print(f"  Bin {i}: {interval[0]:.2f} to {interval[1]:.2f}")
    print()

# Display the results
for feature, scores in metrics.items():
    print(f"Feature: {feature}")
    print(f"Recall: {scores['recall']:.2f}")
    print(f"Precision: {scores['precision']:.2f}\n")

# Display crosstabs as dataframes
for feature, crosstab in crosstabs.items():
    print(f"Crosstab for {feature}:\n{crosstab}\n")
