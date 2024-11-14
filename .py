import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import tensorflow_recommenders as tfrs
import streamlit as st

# Load user data
user_data = pd.read_csv("/home/hartonosng/data-science/sample_user_data_with_numeric.csv")
user_data["AUM"] = pd.to_numeric(user_data["AUM"], errors="coerce").fillna(0)
user_data["income"] = pd.to_numeric(user_data["income"], errors="coerce").fillna(0)

# Encode categorical features
label_encoders = {
    "gender": LabelEncoder(),
    "location": LabelEncoder(),
    "product_holding": LabelEncoder(),
    "card_type": LabelEncoder()
}
for col, encoder in label_encoders.items():
    user_data[col] = encoder.fit_transform(user_data[col].astype(str))

# Load program data
program_data = pd.read_csv("/home/hartonosng/data-science/sample_program_data_filtered.csv")
program_data["program_name"] = program_data["program_name"].astype(str)
program_data["terms_conditions"] = program_data["terms_conditions"].astype(str)

# Adapt TextVectorization layers before using in the model
program_name_vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000)
tnc_vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000)

program_name_vectorizer.adapt(program_data["program_name"])
tnc_vectorizer.adapt(program_data["terms_conditions"])

# Convert to TensorFlow datasets, flattening structure directly
user_dataset = tf.data.Dataset.from_tensor_slices({
    "user_id": user_data["user_id"].values,
    "gender": user_data["gender"].values,
    "age": user_data["age"].values,
    "location": user_data["location"].values,
    "product_holding": user_data["product_holding"].values,
    "card_type": user_data["card_type"].values,
    "AUM": user_data["AUM"].values,
    "income": user_data["income"].values
})

program_dataset = tf.data.Dataset.from_tensor_slices({
    "program_id": program_data["program_id"].values,
    "program_name": program_data["program_name"].values,
    "valid_date": program_data["valid_date"].astype(str).values,
    "terms_conditions": program_data["terms_conditions"].values
})

# Define the UserModel with embeddings for categorical data and dense layers for numerical features
class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gender_embedding = tf.keras.layers.Embedding(input_dim=user_data["gender"].nunique() + 1, output_dim=4)
        self.location_embedding = tf.keras.layers.Embedding(input_dim=user_data["location"].nunique() + 1, output_dim=4)
        self.card_type_embedding = tf.keras.layers.Embedding(input_dim=user_data["card_type"].nunique() + 1, output_dim=4)
        self.aum_layer = tf.keras.layers.Dense(8, activation="relu")
        self.income_layer = tf.keras.layers.Dense(8, activation="relu")
        self.concat_dense = tf.keras.layers.Dense(32, activation='relu')

    def call(self, inputs):
        gender_embedded = self.gender_embedding(inputs["gender"])
        location_embedded = self.location_embedding(inputs["location"])
        card_type_embedded = self.card_type_embedding(inputs["card_type"])
        aum_processed = self.aum_layer(tf.expand_dims(inputs["AUM"], -1))
        income_processed = self.income_layer(tf.expand_dims(inputs["income"], -1))
        concat_output = tf.concat([
            tf.reshape(gender_embedded, (-1, 4)),
            tf.reshape(location_embedded, (-1, 4)),
            tf.reshape(card_type_embedded, (-1, 4)),
            aum_processed,
            income_processed
        ], axis=1)
        return self.concat_dense(concat_output)

# Define the ProgramModel with text vectorization and embedding layers
class ProgramModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.program_name_vectorizer = program_name_vectorizer
        self.program_name_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=8)
        self.tnc_vectorizer = tnc_vectorizer
        self.tnc_embedding_layer = tf.keras.layers.Embedding(input_dim=10000, output_dim=8)
        self.concat_dense = tf.keras.layers.Dense(32, activation='relu')

    def call(self, inputs):
        program_name_tokens = self.program_name_vectorizer(inputs["program_name"])
        program_name_embedded = self.program_name_embedding(program_name_tokens)
        tnc_tokens = self.tnc_vectorizer(inputs["terms_conditions"])
        tnc_embedded = self.tnc_embedding_layer(tnc_tokens)
        program_name_embedded_avg = tf.reduce_mean(program_name_embedded, axis=1)
        tnc_embedded_avg = tf.reduce_mean(tnc_embedded, axis=1)
        concat_output = tf.concat([program_name_embedded_avg, tnc_embedded_avg], axis=1)
        return self.concat_dense(concat_output)

# Multi-task model combining UserModel and ProgramModel for recommendation
class MultiTaskModel(tfrs.models.Model):
    def __init__(self, user_model, program_model):
        super().__init__()
        self.user_model = user_model
        self.program_model = program_model
        self.query_layer = tf.keras.layers.Dense(32, activation="relu")
        self.candidate_layer = tf.keras.layers.Dense(32, activation="relu")
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=program_dataset.batch(128).map(self.program_model)))

    def compute_loss(self, features, training=False):
        user_features = features["user"]
        program_features = features["program"]
        
        # Directly pass flattened user features and program features to respective models
        user_embeddings = self.query_layer(self.user_model(user_features))
        program_embeddings = self.candidate_layer(self.program_model(program_features))
        
        return self.task(user_embeddings, program_embeddings)

# Create the dataset as a tuple to ensure proper mapping
train = tf.data.Dataset.zip((user_dataset, program_dataset))
train = train.map(lambda user, program: {"user": user, "program": program})

# Shuffle and split the dataset manually
cardinality = tf.data.experimental.cardinality(train).numpy()
train_size = int(0.7 * cardinality)
val_size = int(0.15 * cardinality)
test_size = cardinality - train_size - val_size

# Ensure that train, validation, and test sizes are valid
test_size = max(test_size, 1)  # Make sure test size is at least 1
val_size = max(val_size, 1)  # Make sure validation size is at least 1

shuffled = train.shuffle(buffer_size=cardinality, reshuffle_each_iteration=True)
train_dataset = shuffled.take(train_size).batch(128).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = shuffled.skip(train_size).take(val_size).batch(128).cache().prefetch(tf.data.AUTOTUNE)
test_dataset = shuffled.skip(train_size + val_size).take(test_size).batch(128).cache().prefetch(tf.data.AUTOTUNE)

# Initialize models and compile
user_model = UserModel()
program_model = ProgramModel()
multi_task_model = MultiTaskModel(user_model=user_model, program_model=program_model)


multi_task_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Train the model and save training history
history = multi_task_model.fit(train_dataset, epochs=max(2, int(10)), validation_data=val_dataset, steps_per_epoch=max(1, len(user_data) // 128), validation_steps=max(1, val_size // 128), verbose=2)

# Evaluate the model on test data
evaluation_results = multi_task_model.evaluate(test_dataset, return_dict=True)
print("Evaluation results on test data:", evaluation_results)

# Get user and program embeddings for prediction
user_embeddings = []
program_embeddings = []

# Generate embeddings for all users in the test dataset
for batch in test_dataset:
    user_embeddings.append(user_model(batch['user']))
user_embeddings = tf.concat(user_embeddings, axis=0)
user_embeddings = tf.reshape(user_embeddings, [user_embeddings.shape[0], -1])

# Generate embeddings for all programs
for program in program_dataset.batch(128):
    program_embeddings.append(program_model(program))
program_embeddings = tf.concat(program_embeddings, axis=0)
program_embeddings = tf.reshape(program_embeddings, [program_embeddings.shape[0], -1])

# Calculate similarity scores between all users and all programs
scores = tf.linalg.matmul(user_embeddings, program_embeddings, transpose_b=True)

# Streamlit application to show predictions filtered by user_id
st.set_page_config(page_title='User Program Recommendation System', layout='wide')

st.title('User Program Recommendation System')
user_ids = []
for batch in test_dataset:
    user_ids.extend(batch['user']['user_id'].numpy())
user_id_input = st.selectbox('Select User ID:', options=list(set(user_ids)))

st.write('### User Information:')
original_user_data = pd.read_csv("/home/hartonosng/data-science/sample_user_data_with_numeric.csv")
user_info = original_user_data[original_user_data['user_id'] == user_id_input]
if not user_info.empty:
    st.write(f"**User ID:** {user_info['user_id'].values[0]}")
    st.write(f"**Gender:** {user_info['gender'].values[0]}")
    st.write(f"**Age:** {user_info['age'].values[0]}")
    st.write(f"**Location:** {user_info['location'].values[0]}")
    st.write(f"**Product Holding:** {user_info['product_holding'].values[0]}")
    st.write(f"**Card Type:** {user_info['card_type'].values[0]}")
    st.write(f"**AUM:** {user_info['AUM'].values[0]}")
    st.write(f"**Income:** {user_info['income'].values[0]}")
else:
    st.write('User information not found.')

if st.button('Get Recommendations'):
    if user_id_input:
        user_id = int(user_id_input)
        # Find the index of the selected user in the test dataset
        user_index = next((i for i, batch in enumerate(test_dataset) if user_id in batch['user']['user_id'].numpy()), None)
        if user_index is not None:
            # Get similarity scores for the selected user
            user_scores = scores[user_index]
            top_programs_indices = tf.argsort(user_scores, direction='DESCENDING')[:10]
            top_programs = program_data.iloc[top_programs_indices.numpy()]
            for _, program in top_programs.iterrows():
                st.markdown(f"### {program['program_name']}")
                st.image('https://via.placeholder.com/400x200', caption='Program Banner')  # Replace with actual image URL if available
                st.write(f"**Valid Date:** {program['valid_date']}")
                st.write(f"**Terms & Conditions:** {program['terms_conditions']}")
                st.markdown('---')
        else:
            st.write('User ID not found in the test dataset.')
    else:
        st.write('Please enter a valid User ID.')

