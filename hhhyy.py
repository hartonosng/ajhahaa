# Import library yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Simpan data yang sudah di-load
# Misalnya: X_train, X_test, y_train, y_test

# Contoh split data train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Scaling untuk data numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['numeric_column1', 'numeric_column2']])
X_test_scaled = scaler.transform(X_test[['numeric_column1', 'numeric_column2']])

# Preprocessing: One-hot encoding untuk data kategorikal
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train[['categorical_column']])
X_test_encoded = encoder.transform(X_test[['categorical_column']])

# Gabungkan kembali data yang sudah di-preprocess
X_train_preprocessed = np.hstack((X_train_scaled, X_train_encoded.toarray()))
X_test_preprocessed = np.hstack((X_test_scaled, X_test_encoded.toarray()))

# Definisikan model-model yang akan digunakan
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'XGBoost': XGBClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Multi-Layer Perceptron': MLPClassifier()
}

# Inisialisasi dictionary untuk menyimpan hasil evaluasi
results = {'Model': [], 'Precision (label 1)': [], 'Precision (label 0)': [],
           'Recall (label 1)': [], 'Recall (label 0)': [], 'Accuracy': [], 'F1-score': []}

# Inisialisasi variabel untuk menyimpan feature importances
best_model = None
best_f1_score = 0.0
best_features_importances = None

# Loop untuk melatih, menguji, dan menghitung metrik untuk setiap model
for model_name, model in models.items():
    # Training model
    model.fit(X_train_preprocessed, y_train)

    # Prediksi
    y_pred = model.predict(X_test_preprocessed)

    # Hitung metrik evaluasi
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Simpan hasil evaluasi ke dalam dictionary
    results['Model'].append(model_name)
    results['Precision (label 1)'].append(precision_1)
    results['Precision (label 0)'].append(precision_0)
    results['Recall (label 1)'].append(recall_1)
    results['Recall (label 0)'].append(recall_0)
    results['Accuracy'].append(accuracy)
    results['F1-score'].append(f1)

    # Simpan feature importances jika model memiliki F1-score terbaik
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model_name
        if isinstance(model, RandomForestClassifier) or isinstance(model, DecisionTreeClassifier):
            best_features_importances = model.feature_importances_

# Buat dataframe dari hasil evaluasi
results_df = pd.DataFrame(results)

# Tampilkan dataframe hasil evaluasi
print("Summary of Evaluation Metrics:")
print(results_df)

# Tampilkan feature importances untuk best model
if best_features_importances is not None:
    feature_names = list(X_train.columns)
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': best_features_importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)
    print(f"\nFeature Importances for Best Model ({best_model}):")
    print(importances_df)
else:
    print("\nNo feature importances available for the best model.")
