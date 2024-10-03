import polars as pl
import numpy as np

# Sample DataFrames
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

# Step 1: Get all products as a list for quick lookup
all_products = product_data['product_code'].to_list()

# Step 2: Function to get random negative samples
def random_negative_sampling(interacted_products, all_products, num_neg_samples=2):
    non_interacted = list(set(all_products) - set(interacted_products))
    if len(non_interacted) == 0:
        return []
    return np.random.choice(non_interacted, size=min(num_neg_samples, len(non_interacted)), replace=False).tolist()

# Step 3: Function to get hard negative samples based on product similarity
def hard_negative_sampling(interacted_products, product_data, num_hard_neg_samples=1):
    hard_negatives = []
    interacted_product_types = product_data.filter(pl.col('product_code').is_in(interacted_products))

    for product in interacted_products:
        product_type = interacted_product_types.filter(pl.col('product_code') == product)['product_type'].to_numpy()[0]

        # Get candidates that are of the same product type but not interacted with
        similar_candidates = product_data.filter((pl.col('product_type') == product_type) & 
                                                  (~pl.col('product_code').is_in(interacted_products)))
        
        if similar_candidates.height > 0:
            hard_negatives.append(similar_candidates['product_code'].to_list()[0])

    return hard_negatives[:num_hard_neg_samples]

# Step 4: Combine Random and Hard Negative Sampling
def sample_negatives(interacted_products, all_products, product_data, num_random_neg=2, num_hard_neg=1):
    random_negatives = random_negative_sampling(interacted_products, all_products, num_random_neg)
    hard_negatives = hard_negative_sampling(interacted_products, product_data, num_hard_neg)
    return random_negatives + hard_negatives

# Step 5: Prepare interacted products for each user using the new method
interacted_products_per_user = (
    interaction_data
    .groupby('customerid')
    .agg(pl.col('product_code').list().alias('interacted_products'))
)

# Step 6: Generate negative samples for each user
negative_samples = []
for user in interacted_products_per_user.rows():
    customer_id = user[0]  # customerid
    interacted_products = user[1]  # interacted_products
    neg_samples = sample_negatives(interacted_products, all_products, product_data)
    negative_samples.append((customer_id, neg_samples))

# Convert to a DataFrame
negative_samples_df = pl.DataFrame(negative_samples, schema=["customerid", "negative_samples"])

# View the result
print(negative_samples_df)
