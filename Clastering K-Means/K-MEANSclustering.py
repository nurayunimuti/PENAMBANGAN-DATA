import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# === 1. Membuat Dataset Konsumen ===
data = {
    'Age': [25, 34, 28, 45, 35, 52, 23, 40, 60, 48, 33, 50, 21, 41, 55, 39, 30, 47, 36, 38],
    'Annual Income (k$)': [15, 20, 22, 40, 35, 80, 18, 50, 100, 60, 25, 70, 16, 55, 90, 45, 32, 65, 30, 50],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 6, 94, 3, 73, 14, 99, 15, 77, 13, 80, 82, 25, 14, 24, 19]
}
df = pd.DataFrame(data)

# Menampilkan data
print("Dataset Konsumen:")
print(df.head())

# === 2. Standarisasi Data ===
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# === 3. Menentukan Jumlah Cluster Optimal dengan Metode Elbow ===
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Visualisasi Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# === 4. Klastering dengan K-Means ===
# Menggunakan jumlah cluster optimal (misal k=3 berdasarkan elbow method)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Menambahkan hasil klaster ke dataset
df['Cluster'] = clusters
print("\nDataset dengan Klaster:")
print(df.head())

# === 5. Visualisasi Hasil Klaster ===
plt.figure(figsize=(8, 5))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (Age)')
plt.ylabel('Feature 2 (Annual Income)')
plt.legend()
plt.show()

# Menampilkan centroid
print("\nCentroid dari masing-masing cluster:")
print(kmeans.cluster_centers_)
