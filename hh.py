import polars as pl
import numpy as np

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
    # Calculate similarities based on product type
    product_types = product_data.select(['product_code', 'product_type'])
    product_types = product_types.with_columns(
        pl.col('product_type').cast(pl.Int32)  # Ensure the product type is of correct type
    )
    
    # Create a similarity matrix based on product types
    similarity_matrix = product_types.to_pandas().pivot_table(index='product_code', columns='product_type', aggfunc='size', fill_value=0).values
    
    # Compute similarity (you can use any method; here we just take product types)
    # For simplicity, let's assume products with the same type are more similar.
    # You can implement more complex similarity measures as needed.
    
    hard_negatives = []
    for product in interacted_products:
        similar_products = product_types.filter(pl.col('product_type') == product_types.filter(pl.col('product_code') == product)['product_type'][0])
        
        # Exclude products already interacted with
        candidates = similar_products.filter(~pl.col('product_code').is_in(interacted_products)).select('product_code')
        if candidates.height > 0:
            hard_negatives.append(candidates['product_code'][0])  # Get the first similar product

    return hard_negatives[:num_hard_neg_samples]

# Step 3: Combine Random and Hard Negative Sampling
def sample_negatives(interacted_products, all_products, product_data, num_random_neg=2, num_hard_neg=1):
    # Random negative samples
    random_negatives = random_negative_sampling(interacted_products, all_products, num_random_neg)
    
    # Hard negative samples
    hard_negatives = hard_negative_sampling(interacted_products, product_data, num_hard_neg)
    
    # Combine both and return as a list (Polars-compatible)
    return random_negatives + hard_negatives

# Step 4: Apply sampling for each user using 'with_columns'
# Group the interaction_data by customerid to get the list of interacted products for each user
interacted_products_per_user = interaction_data.groupby('customerid').agg(
    pl.col('product_code').apply(lambda x: x.to_list(), return_dtype=pl.List(pl.Utf8))).rename({"product_code": "interacted_products"})
)

# Perform negative sampling
negative_samples_df = interacted_products_per_user.with_columns(
    pl.struct(['interacted_products']).apply(
        lambda row: sample_negatives(row['interacted_products'], all_products, product_data)
    ).alias('negative_samples')
)

# View the result
print(negative_samples_df)
