import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataframes (structure only)
user_data = pl.DataFrame({
    'customerid': [1, 2, 3],
    'demographic_feature1': [10, 20, 30],
    'demographic_feature2': [1, 0, 1]
})

interaction_data = pl.DataFrame({
    'customerid': [1, 1, 2],
    'product_code': ['A', 'B', 'A'],
    'transaction_amount': [100, 150, 200]
})

product_data = pl.DataFrame({
    'product_code': ['A', 'B', 'C', 'D'],
    'product_type': [1, 2, 1, 2]
})

# Get all products and users
all_products = set(product_data['product_code'].to_list())

# Function to get random negative samples
def random_negative_sampling(user_interactions, all_products, num_neg_samples=2):
    # Get products that have not been interacted with
    non_interacted = list(all_products - set(user_interactions))
    
    # Sample random negatives from non-interacted products
    if len(non_interacted) > 0:
        return list(np.random.choice(non_interacted, size=min(num_neg_samples, len(non_interacted)), replace=False))
    else:
        return []

# Function to get hard negative samples based on product similarity
def hard_negative_sampling(interacted_products, product_data, num_hard_neg_samples=1):
    # Convert product data to pandas for cosine similarity
    product_features = product_data.select(['product_code', 'product_type']).to_pandas().set_index('product_code')
    
    # Calculate cosine similarity between products based on 'product_type'
    product_sim = cosine_similarity(product_features)
    
    # Create a DataFrame of similarities
    product_sim_df = pl.DataFrame({
        'product_code': product_features.index,
        'similarity': [row for row in product_sim]
    })
    
    # Select hard negatives based on product similarity
    hard_negatives = []
    for product in interacted_products:
        similar_products = product_sim_df.filter(pl.col('product_code') == product)['similarity']
        
        # Find hard negatives (similar products but not interacted with)
        candidates = product_sim_df.filter(~pl.col('product_code').is_in(interacted_products)).select('product_code')
        if len(candidates) > 0:
            hard_negatives.append(candidates[0][0])  # Get the most similar product as hard negative
            
    return list(hard_negatives[:num_hard_neg_samples])

# Step 3: Combine Random and Hard Negative Sampling
def sample_negatives(interacted_products, all_products, product_data, num_random_neg=2, num_hard_neg=1):
    # Random negative samples
    random_negatives = random_negative_sampling(interacted_products, all_products, num_random_neg)
    
    # Hard negative samples
    hard_negatives = hard_negative_sampling(interacted_products, product_data, num_hard_neg)
    
    # Combine both and return as a list (Polars-compatible)
    return random_negatives + hard_negatives

# Group the interaction_data by customerid to get the list of interacted products for each user
interacted_products_per_user = interaction_data.groupby('customerid').agg(
    pl.col('product_code').apply(lambda x: x.to_list(), return_dtype=pl.List(pl.Utf8)).alias('interacted_products')
)

# Create a column with negative samples using Polars-native operations
negative_samples_df = interacted_products_per_user.with_columns(
    [
        pl.struct(['interacted_products']).apply(
            lambda row: sample_negatives(row['interacted_products'], all_products, product_data)
        ).alias('negative_samples')
    ]
)

# View the result
print(negative_samples_df)
