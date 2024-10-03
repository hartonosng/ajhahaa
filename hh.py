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
    'product_type': [1, 2, 1, 2]  # Assuming product_type can be used for similarity
})

# Step 1: Get all products as a set for quick lookup
all_products = set(product_data['product_code'].to_list())

# Step 2: Function to get random negative samples
def random_negative_sampling(interacted_products, all_products, num_neg_samples=2):
    non_interacted = list(all_products - set(interacted_products))
    if len(non_interacted) == 0:
        return []
    return list(np.random.choice(non_interacted, size=min(num_neg_samples, len(non_interacted)), replace=False))

# Step 3: Function to get hard negative samples based on product similarity
def hard_negative_sampling(interacted_products, product_data, num_hard_neg_samples=1):
    hard_negatives = []
    for product in interacted_products:
        product_type = product_data.filter(pl.col('product_code') == product)['product_type'][0]
        similar_candidates = product_data.filter((pl.col('product_type') == product_type) &
                                                  (~pl.col('product_code').is_in(interacted_products)))
        
        # Select the first candidate if available
        if similar_candidates.height > 0:
            hard_negatives.append(similar_candidates['product_code'][0])

    return hard_negatives[:num_hard_neg_samples]

# Step 4: Combine Random and Hard Negative Sampling
def sample_negatives(interacted_products, all_products, product_data, num_random_neg=2, num_hard_neg=1):
    random_negatives = random_negative_sampling(interacted_products, all_products, num_random_neg)
    hard_negatives = hard_negative_sampling(interacted_products, product_data, num_hard_neg)
    return random_negatives + hard_negatives

# Step 5: Prepare interacted products for each user
interacted_products_per_user = interaction_data.groupby('customerid').agg(
    pl.col('product_code').list().alias('interacted_products')
)

# Step 6: Generate negative samples for each user without using apply
negative_samples = []
for row in interacted_products_per_user.to_dicts():
    customer_id = row['customerid']
    interacted_products = row['interacted_products']
    neg_samples = sample_negatives(interacted_products, all_products, product_data)
    negative_samples.append((customer_id, neg_samples))

# Convert to a DataFrame
negative_samples_df = pl.DataFrame(negative_samples, schema=["customerid", "negative_samples"])

# View the result
print(negative_samples_df)
