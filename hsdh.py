import pandas as pd
import matplotlib.pyplot as plt

# Example DataFrame creation
data = {
    'customerid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'period': ['202101', '202102', '202102', '202103', '202103', '202103', '202104', '202104', '202105', '202106'],
    'label': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# Convert 'period' column to datetime format for easier manipulation
df['period'] = pd.to_datetime(df['period'], format='%Y%m')

# Filter data for bad customers (label 1)
df_bad = df[df['label'] == 1]

# Create a new column to represent the end period of 3 months later
df_bad['end_period'] = df_bad['period'] + pd.DateOffset(months=2)

# Group by 'period' and count the number of bad customers within the next 3 months
bad_customer_counts = df_bad.groupby(df_bad['period'].dt.to_period('M'))['customerid'].count()

# Plotting using Matplotlib
plt.figure(figsize=(10, 6))
plt.bar(bad_customer_counts.index.astype(str), bad_customer_counts.values, color='skyblue')
plt.title('Distribution of Bad Customers Count per Month')
plt.xlabel('Month (YYYYMM)')
plt.ylabel('Number of Bad Customers')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
