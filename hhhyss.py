import pandas as pd

# Sample DataFrame
data = {
    'feature1': ['A', 'A', 'B', 'B', 'C', 'C'],
    'feature2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'label': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Frequency crosstab
frequency_crosstab = pd.crosstab(df['feature1'], df['label'])

# Percentage crosstab (normalize='index' to normalize over rows)
percentage_crosstab = pd.crosstab(df['feature1'], df['label'], normalize='index') * 100

# Rename columns for clarity
frequency_crosstab.columns = [f'count_{col}' for col in frequency_crosstab.columns]
percentage_crosstab.columns = [f'percent_{col}' for col in percentage_crosstab.columns]

# Concatenate frequency and percentage crosstabs
combined_crosstab = pd.concat([frequency_crosstab, percentage_crosstab], axis=1)

print("Combined Crosstab (Frequency and Percentage):")
print(combined_crosstab)
