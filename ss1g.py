# Impor library yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Misalkan kita memiliki dataset dalam bentuk dataframe
# Gantilah dengan cara memuat dataset sesuai kebutuhan Anda
# Misalnya:
# df = pd.read_csv('nama_file.csv')

# Contoh data
data = {
    'Fitur1': [10, 20, 30, 40, 50],
    'Fitur2': [5, 15, 25, 35, 45],
    'Label': [0, 1, 0, 1, 0]  # Misalnya, 1 untuk 'bad', 0 untuk 'customer'
}

df = pd.DataFrame(data)

# Memisahkan fitur dan label
X = df[['Fitur1', 'Fitur2']]
y = df['Label']

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi fitur-fitur dengan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Melatih model Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Memprediksi label untuk data uji
y_pred = model.predict(X_test_scaled)

# Evaluasi model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
