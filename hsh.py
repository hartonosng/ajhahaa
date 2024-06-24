import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame
data = {
    'customerid': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'period': ['20230101', '20230102', '20230201', '20230202', '20230301', '20230302', '20230401', '20230402', '20230501', '20230502'],
    'collectibility': [1, 2, 1, 3, 1, 2, 1, 1, 3, 2],
    'collectibility_next_1': [2, 2, 1, 3, 2, 2, 2, 1, 3, 2],
    'collectibility_next_2': [3, 2, 1, 3, 3, 2, 3, 1, 3, 3],
    'collectibility_next_3': [4, 2, 1, 3, 4, 2, 4, 1, 3, 4]
}
df = pd.DataFrame(data)

# Generate the 'label' column
df['label'] = df.apply(lambda row: 1 if row['collectibility'] == 1 and 
                       (row['collectibility_next_1'] > 1 or 
                        row['collectibility_next_2'] > 1 or 
                        row['collectibility_next_3'] > 1) else 0, axis=1)

# Convert the 'period' column to datetime
df['period'] = pd.to_datetime(df['period'], format='%Y%m%d')

# Visualization
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Plot all points
sns.scatterplot(data=df, x='period', y='collectibility', hue='label', style='label', palette='coolwarm', s=100)

# Highlight points where label is 1
high_risk = df[df['label'] == 1]
plt.scatter(high_risk['period'], high_risk['collectibility'], color='red', edgecolor='black', s=200, label='High Risk (label=1)', marker='X')

plt.title('Collectibility Over Periods with High Risk Cases Highlighted', fontsize=15)
plt.xlabel('Period', fontsize=12)
plt.ylabel('Collectibility', fontsize=12)
plt.legend(title='Label', loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
