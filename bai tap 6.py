import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, rand_score, normalized_mutual_info_score, davies_bouldin_score
from sklearn.datasets import load_iris
import random

# Load dữ liệu IRIS
iris = load_iris()
X = iris.data
y = iris.target

# Khởi tạo số lượng cụm K
K = 3  # Bộ dữ liệu IRIS có 3 lớp

# Khởi tạo các centroid ban đầu ngẫu nhiên từ dữ liệu
def initialize_centroids(X, K):
    indices = random.sample(range(X.shape[0]), K)
    centroids = X[indices]
    return centroids

# Tính khoảng cách Euclidean từ một điểm đến một centroid
def compute_distance(point, centroid):
    return np.sqrt(((point - centroid) ** 2).sum())

# Gán điểm vào cụm gần nhất
def assign_clusters(X, centroids):
    clusters = []
    for i in range(X.shape[0]):
        distances = [compute_distance(X[i], centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Cập nhật centroids
def update_centroids(X, clusters, K):
    new_centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        if len(X[clusters == k]) > 0:
            new_centroids[k] = X[clusters == k].mean(axis=0)
    return new_centroids

# Hàm chính K-means
def kmeans(X, K, max_iters=100):
    centroids = initialize_centroids(X, K)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, K)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters

# Gán các cụm cho dữ liệu
clusters = kmeans(X, K)

# Chuyển đổi nhãn cụm thành nhãn thật để tính F1-score
def map_clusters_to_labels(clusters, y, K):
    mapped_labels = np.zeros_like(clusters)
    for k in range(K):
        mask = (clusters == k)
        true_labels = y[mask]
        if len(true_labels) > 0:
            new_label = np.bincount(true_labels).argmax()
            mapped_labels[mask] = new_label
    return mapped_labels

# Ánh xạ cụm với nhãn thật
mapped_clusters = map_clusters_to_labels(clusters, y, K)

# Tính F1-score
f1 = f1_score(y, mapped_clusters, average='macro')
print(f"F1-score: {f1:.4f}")

# Tính Rand Index
rand_index = rand_score(y, mapped_clusters)
print(f"Rand Index: {rand_index:.4f}")

# Tính Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(y, mapped_clusters)
print(f"NMI: {nmi:.4f}")

# Tính Davies-Bouldin Index
db_index = davies_bouldin_score(X, clusters)
print(f"Davies-Bouldin Index: {db_index:.4f}")
