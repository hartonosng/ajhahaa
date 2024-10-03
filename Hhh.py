import polars as pl
import numpy as np

# Sample user_data
user_data = pl.DataFrame({
    'customerid': [1, 2, 3],
    'age_group': [1, 2, 3],  # Encoded features
    'income_level': [3, 2, 1],
    'gender': [1, 0, 1]  # Encoded gender
})

# Sample interaction_data
interaction_data = pl.DataFrame({
    'customerid': [1, 1, 2, 3],
    'product_code': ['A', 'B', 'A', 'C'],
    'transaction_amount': [100, 200, 300, 150]
})

# Sample product_data
product_data = pl.DataFrame({
    'product_code': ['A', 'B', 'C', 'D', 'E'],
    'product_type': [1, 2, 3, 1, 2],  # Encoded product features
    'price_level': [3, 2, 1, 3, 2]
})

# 1. Prepare positive interactions
positive_interactions = interaction_data.select(['customerid', 'product_code']).unique()

# 2. Create candidate pool for negative sampling (products customer hasn't interacted with)
def get_negative_samples(customerid, positive_products, all_products):
    # Products customer has interacted with
    interacted_products = positive_products.filter(pl.col('customerid') == customerid).select('product_code').to_series().to_list()
    # Available products for negative sampling
    available_products = all_products.filter(~pl.col('product_code').is_in(interacted_products))
    return available_products

# 3. Hard Negative Sampling based on product features
def hard_negative_sampling(customerid, positive_products, all_products, product_data):
    # Get customer's positive interactions
    interacted_products = positive_products.filter(pl.col('customerid') == customerid).select('product_code').to_series().to_list()
    # Get the features of those products
    interacted_features = product_data.filter(pl.col('product_code').is_in(interacted_products)).drop('product_code')
    
    # Calculate similarity or choose products with similar features (e.g., product_type)
    similar_products = product_data.filter(~pl.col('product_code').is_in(interacted_products))
    similar_products = similar_products.filter(pl.col('product_type').is_in(interacted_features['product_type'].to_list()))
    
    return similar_products

# 4. Negative Sampling combining random and hard negatives
def negative_sampling(customerid, positive_products, all_products, product_data, num_random=2, num_hard=2):
    # Get random negative samples
    available_random_samples = get_negative_samples(customerid, positive_products, all_products)
    random_samples = available_random_samples.sample(n=num_random)
    
    # Get hard negative samples
    hard_samples = hard_negative_sampling(customerid, positive_products, all_products, product_data)
    hard_samples = hard_samples.sample(n=min(num_hard, hard_samples.height))
    
    # Combine random and hard negatives
    negative_samples = pl.concat([random_samples, hard_samples])
    return negative_samples

# Example usage
all_products = product_data.select(['product_code'])  # All available products

customer_negative_samples = []

for customer in user_data['customerid'].to_list():
    neg_samples = negative_sampling(customer, positive_interactions, all_products, product_data)
    neg_samples = neg_samples.with_columns(pl.lit(customer).alias('customerid'))
    customer_negative_samples.append(neg_samples)

# Concatenate all negative samples
negative_samples = pl.concat(customer_negative_samples).with_columns(pl.arange(0, pl.count()).alias('id'))

# Display the results
negative_samples
