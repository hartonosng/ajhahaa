import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# === Load and preprocess data ===
# Assume `user_data`, `interaction_data`, `product_data` are pandas DataFrames
# Replace these with actual data loading if necessary
user_data = pd.read_csv('user_data.csv')
interaction_data = pd.read_csv('interaction_data.csv')
product_data = pd.read_csv('product_data.csv')

# Merge interaction_data with user_data and product_data
data = interaction_data.merge(user_data, on='customerid', how='left')
data = data.merge(product_data, on='product_code', how='left')

# Encode product_code for multiclass prediction
label_encoder = LabelEncoder()
data['product_code_encoded'] = label_encoder.fit_transform(data['product_code'])

# Define feature sets
user_features = ['age', 'income', 'segment', 'mob', 'aum']
product_features = ['product_type', 'performance']
interaction_features = ['trx_amount_idr_l1m', 'trx_amount_idr_l2m', 'trx_amount_idr_l3m', 
                        'trx_amount_idr_l4m', 'trx_amount_idr_l5m', 'trx_amount_idr_l6m']

X_user = data[user_features]
X_product = data[product_features]
X_interaction = data[interaction_features]
y = data['product_code_encoded']

# Split into train and test sets
X_user_train, X_user_test, X_product_train, X_product_test, X_interaction_train, X_interaction_test, y_train, y_test = train_test_split(
    X_user, X_product, X_interaction, y, test_size=0.2, random_state=42)

# === Define Two Towers Model ===
# Input towers
user_input = Input(shape=(len(user_features),), name="user_input")
product_input = Input(shape=(len(product_features),), name="product_input")
interaction_input = Input(shape=(len(interaction_features),), name="interaction_input")

# User tower
user_dense = Dense(64, activation='relu')(user_input)
user_dense = Dropout(0.2)(user_dense)
user_embedding = Dense(32, activation='relu', name="user_embedding")(user_dense)

# Product tower
product_dense = Dense(64, activation='relu')(product_input)
product_dense = Dropout(0.2)(product_dense)
product_embedding = Dense(32, activation='relu', name="product_embedding")(product_dense)

# Interaction tower
interaction_dense = Dense(64, activation='relu')(interaction_input)
interaction_dense = Dropout(0.2)(interaction_dense)
interaction_embedding = Dense(32, activation='relu', name="interaction_embedding")(interaction_dense)

# Concatenate embeddings
combined_embedding = Concatenate()([user_embedding, product_embedding, interaction_embedding])
output = Dense(len(label_encoder.classes_), activation='softmax', name="output")(combined_embedding)

# Build and compile model
model = Model(inputs=[user_input, product_input, interaction_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train the model ===
history = model.fit(
    [X_user_train, X_product_train, X_interaction_train],
    y_train,
    validation_data=([X_user_test, X_product_test, X_interaction_test], y_test),
    epochs=10,
    batch_size=64
)

# === Evaluate the model ===
loss, accuracy = model.evaluate([X_user_test, X_product_test, X_interaction_test], y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# === Predict top product for a new customer ===
predictions = model.predict([X_user_test, X_product_test, X_interaction_test])
predicted_classes = np.argmax(predictions, axis=1)
predicted_product_codes = label_encoder.inverse_transform(predicted_classes)

# Save the model
model.save('two_towers_model.h5')
