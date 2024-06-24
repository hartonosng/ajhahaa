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

# Create a new DataFrame to store periods where customers become bad within 3 months
bad_periods = pd.DataFrame(columns=['period', 'count'])

# Iterate through each period
for index, row in df.iterrows():
    current_period = row['period']
    end_period = current_period + pd.DateOffset(months=2)  # End of 3 months period
    
    # Filter customers that become bad within the next 3 months
    future_bad_customers = df[(df['period'] > current_period) & (df['period'] <= end_period) & (df['label'] == 1)]
    
    # Count these customers
    count_bad_customers = future_bad_customers.shape[0]
    
    # Add to bad_periods DataFrame
    bad_periods = bad_periods.append({'period': current_period, 'count': count_bad_customers}, ignore_index=True)

# Plotting using Matplotlib
plt.figure(figsize=(10, 6))
plt.bar(bad_periods['period'].dt.strftime('%Y%m'), bad_periods['count'], color='skyblue')
plt.title('Periods where customers turn from good to bad within the next 3 months')
plt.xlabel('Period (YYYYMM)')
plt.ylabel('Number of Bad Customers')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()
