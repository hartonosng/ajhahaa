import pandas as pd
import random
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st

# Sample customer data generation
customer_ids = [f'CUST{i:05}' for i in range(1, 10001)]
customer_ages = [random.randint(18, 70) for _ in range(10000)]
incomes = [random.randint(3000, 100000) for _ in range(10000)]
segments = random.choices(['Mass', 'Affluent', 'High Net Worth'], weights=[0.6, 0.3, 0.1], k=10000)

customer_data = pd.DataFrame({
    'customerid': customer_ids,
    'customerage': customer_ages,
    'income': incomes,
    'segment': segments
})

# Sample program data generation
program_ids = [f'PROG{i:03}' for i in range(1, 21)]
program_names = [
    'Travel Reward', 'Dining Delight', 'Shopping Extravaganza', 'Fuel Saver', 'Gadget Mania',
    'Cashback Carnival', 'Luxury Lounge', 'Hotel Stay Offer', 'Two Towers Dining', 'Movie Magic',
    'Spa Retreat', 'Adventure Fun', 'Grocery Gold', 'Fitness First', 'Wellness Reward',
    'Electronics Edge', 'Home Needs', 'Pet Lover', 'Weekend Getaway', 'Festive Special'
]
terms_conditions = [
    "Reward points applicable for every $100 spent. Points redeemable at selected partners.",
    "Get up to 20% off on dining at participating restaurants. Valid on weekends.",
    "Earn 5% cashback on purchases above $500 during promotional periods.",
    "Save 3% on fuel surcharge at participating petrol stations.",
    "Exclusive offers on gadgets with partnered retailers. Available while stocks last.",
    "Flat 5% cashback on all online purchases capped at $50 per month.",
    "Access to premium lounges at selected airports. Limit: 4 visits per year.",
    "Enjoy up to 2 complimentary nights at partnered hotels once a year.",
    "Special dining offers at Two Towers restaurants. Booking required.",
    "Up to 50% off on movie tickets every Friday at selected theatres.",
    "20% discount on spa treatments at partnered luxury spas.",
    "Adventure sports vouchers worth up to $100 on spends above $1000.",
    "Earn extra reward points on grocery purchases. Points expire in 3 months.",
    "Access to top-tier gyms and fitness studios with discounts up to 30%.",
    "Redeem wellness rewards for spa, health checkups, and more.",
    "Exclusive discounts on electronics at selected retail partners.",
    "Flat 10% off on home needs and improvement products.",
    "10% off on pet products at partner stores. Valid only on weekends.",
    "Discount vouchers for weekend trips and getaways. T&Cs apply.",
    "Up to 25% off on purchases during festive season at selected outlets."
]

program_data = pd.DataFrame({
    'program_id': program_ids,
    'program_name': program_names,
    'terms_conditions': terms_conditions
})

# Preparing data for TensorFlow model
label_encoder = LabelEncoder()
customer_data['segment_encoded'] = label_encoder.fit_transform(customer_data['segment'])
scaler = StandardScaler()
customer_data[['customerage', 'income']] = scaler.fit_transform(customer_data[['customerage', 'income']])

# Define the User and Candidate models
class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Dense(32, activation='relu')

    def call(self, inputs):
        return self.embedding_layer(inputs)

class CandidateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.program_embedding_layer = tf.keras.layers.StringLookup(vocabulary=program_names, mask_token=None)
        self.program_embedding_vector = tf.keras.layers.Embedding(input_dim=len(program_names) + 1, output_dim=32)
        self.terms_embedding_layer = tf.keras.layers.TextVectorization(output_mode='int', output_sequence_length=50)
        self.terms_embedding_layer.adapt(program_data['terms_conditions'].values)
        self.terms_embedding_vector = tf.keras.layers.Embedding(input_dim=1000, output_dim=32)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.dense_layer = tf.keras.layers.Dense(32, activation='relu')

    def call(self, inputs, training=False):
        program_input, terms_input = inputs
        program_embedded = self.program_embedding_layer(tf.expand_dims(program_input, -1))
        program_embedded = self.program_embedding_vector(program_embedded)
        program_embedded = tf.reshape(program_embedded, (-1, 32))
        terms_embedded = self.terms_embedding_layer(tf.expand_dims(terms_input, -1))
        terms_embedded = self.terms_embedding_vector(terms_embedded)
        terms_embedded = self.flatten_layer(terms_embedded)
        concatenated = self.concat_layer([program_embedded, terms_embedded])
        return self.dense_layer(concatenated)

# Define the Two Towers Model
class TwoTowersModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.user_model = UserModel()
        self.candidate_model = CandidateModel()
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices((program_data['program_name'], program_data['terms_conditions'])).batch(128).map(lambda x, y: self.candidate_model((x, y)))
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["customer_features"])
        candidate_embeddings = self.candidate_model((features["program_name"], features["terms_conditions"]))
        return self.task(user_embeddings, candidate_embeddings)

# Prepare features for training
features = {
    "customer_features": customer_data[['customerage', 'income']].values,
    "program_name": program_data['program_name'].sample(len(customer_data), replace=True).values,
    "terms_conditions": program_data['terms_conditions'].sample(len(customer_data), replace=True).values
}

# Split the data into training, validation, and test sets
train_indices, test_indices = train_test_split(range(len(customer_data)), test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

train_features = {key: value[train_indices] for key, value in features.items()}
val_features = {key: value[val_indices] for key, value in features.items()}
test_features = {key: value[test_indices] for key, value in features.items()}

# Instantiate and compile the model
model = TwoTowersModel()
model.compile(optimizer='adam')

# Train the model
model.fit(x=train_features, validation_data=(val_features), epochs=1, batch_size=32)

# Evaluate the model
model.evaluate(test_features)

print("Model training complete.")

# Function to make predictions for a specific customer
def recommend_for_customer(customer_id):
    customer_info = customer_data[customer_data['customerid'] == customer_id]
    if customer_info.empty:
        return "Customer ID not found."
    customer_features = scaler.transform(customer_info[['customerage', 'income']].values)
    user_embedding = model.user_model(customer_features)
    candidate_dataset = tf.convert_to_tensor([model.candidate_model([program_name, terms_conditions]) for program_name, terms_conditions in zip(program_data['program_name'], program_data['terms_conditions'])])
    scores = tf.linalg.matmul(user_embedding, candidate_dataset, transpose_b=True)
    top_scores_indices = tf.argsort(tf.squeeze(scores), direction='DESCENDING')[:5]
    recommendations = program_data.iloc[top_scores_indices.numpy()]
    return recommendations[['program_name', 'terms_conditions']]

# Streamlit app
st.title("Customer Credit Card Program Recommender")
customer_id_input = st.selectbox("Select Customer ID:", options=customer_data['customerid'].tolist(), index=0)
if st.button("Recommend Programs"):
    if customer_id_input:
        recommendations = recommend_for_customer(customer_id_input)
        st.write(recommendations)
