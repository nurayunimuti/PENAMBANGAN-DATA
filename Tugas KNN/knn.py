# Import libraries yang diperlukan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Load dataset
data = pd.read_excel('Social_Network_Ads_Modified_Numeric.xlsx')

# Tampilkan beberapa data untuk melihat formatnya
print(data.head())

# Pisahkan fitur (X) dan target (y)
X = data.iloc[:, :-1].values  # Fitur: semua kolom kecuali yang terakhir
y = data.iloc[:, -1].values   # Target: kolom terakhir

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Fungsi untuk melakukan prediksi K-NN
def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Hitung jarak antara test_point dengan semua data latih
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        
        # Urutkan berdasarkan jarak dan pilih k terdekat
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        
        # Voting untuk menentukan kelas mayoritas
        majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(majority_vote)
    return predictions

# Menentukan nilai k
k = 4  # Anda bisa mencoba beberapa nilai k untuk hasil optimal

# Lakukan prediksi pada data uji
y_pred = knn_predict(X_train, y_train, X_test, k)

# Evaluasi akurasi prediksi
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Akurasi model K-NN dengan k={k}: {accuracy * 100:.2f}%')
