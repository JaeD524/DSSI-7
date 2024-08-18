import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import numpy as np

# 데이터 로드
cluster_df = pd.read_excel("C:\\Users\\dove8\\OneDrive\\바탕 화면\\DSSI\\my_data.xlsx")
print(cluster_df.head(3))

# 위도와 경도 데이터만 추출
df = cluster_df[['LatitudeRound', 'LongitudeRound']].values

# KMeans 하이퍼파라미터 튜닝
kmeans_param_grid = {
    'n_clusters': [3, 4, 5, 6, 7],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30, 40, 50],
    'random_state': [42]
}

best_kmeans_score = -1
best_kmeans_params = {}

for params in ParameterGrid(kmeans_param_grid):
    kmeans = KMeans(n_clusters=params['n_clusters'], init=params['init'], n_init=params['n_init'], random_state=params['random_state'])
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    if score > best_kmeans_score:
        best_kmeans_score = score
        best_kmeans_params = params

print(f"Best KMeans Params: {best_kmeans_params}")
print(f"Best KMeans Silhouette Score: {best_kmeans_score}")

# 최적의 KMeans 모델로 클러스터링 수행
best_kmeans = KMeans(n_clusters=best_kmeans_params['n_clusters'], init=best_kmeans_params['init'], n_init=best_kmeans_params['n_init'], random_state=best_kmeans_params['random_state'])
kmeans_labels = best_kmeans.fit_predict(df)

# DBSCAN 하이퍼파라미터 튜닝
dbscan_param_grid = {
    'eps': [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'min_samples': [3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
}

best_dbscan_score = -1
best_dbscan_params = {}

for params in ParameterGrid(dbscan_param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(df)
    
    if len(set(labels)) > 1:  # 모든 데이터가 한 클러스터에 있지 않다면
        score = silhouette_score(df, labels)
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_dbscan_params = params

print(f"Best DBSCAN Params: {best_dbscan_params}")
print(f"Best DBSCAN Silhouette Score: {best_dbscan_score}")

# 최적의 DBSCAN 모델로 클러스터링 수행
dbscan = DBSCAN(eps=best_dbscan_params['eps'], min_samples=best_dbscan_params['min_samples'])
dbscan_labels = dbscan.fit_predict(df)

# 시각화
plt.figure(figsize=(14, 6))

# KMeans 시각화
plt.subplot(1, 2, 1)
plt.scatter(df[:, 0], df[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('KMeans Clustering Results')
plt.xlabel('Latitude')
plt.ylabel('Longitude')

# DBSCAN 시각화
plt.subplot(1, 2, 2)
plt.scatter(df[:, 0], df[:, 1], c=dbscan_labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering Results')
plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.tight_layout()
plt.show()
