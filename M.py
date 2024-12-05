import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'customerid': [1, 1, 2, 2, 3, 3, 4, 4],
    'product_code': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'labels': [1, 0, 0, 1, 0, 1, 0, 0],
    'probability': [0.9, 0.5, 0.3, 0.4, 0.1, 0.2, 0, 0]
}
df = pd.DataFrame(data)

# Function for Normalization
def normalize_probabilities(group):
    if group['probability'].sum() == 0:
        group['normalized_prob'] = 1 / len(group)
    else:
        group['normalized_prob'] = group['probability'] / group['probability'].sum()
    return group

# Function for Softmax Scaling
def softmax_probabilities(group):
    if group['probability'].sum() == 0:
        group['softmax_prob'] = 1 / len(group)
    else:
        exp_probs = np.exp(group['probability'])
        group['softmax_prob'] = exp_probs / exp_probs.sum()
    return group

# Apply Normalization and Softmax Scaling
df = df.groupby('customerid').apply(normalize_probabilities)
df = df.groupby('customerid').apply(softmax_probabilities)

# Display resulting DataFrame
print(df)
