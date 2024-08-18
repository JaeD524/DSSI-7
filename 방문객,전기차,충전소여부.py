import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 데이터 로드
cluster_df = pd.read_excel("C:\\Users\\dove8\\OneDrive\\바탕 화면\\DSSI\\my_data.xlsx")
print(cluster_df.head(3))

# 방문객 수, 전기차 대수, 충전소 여부 데이터만 추출
df = cluster_df[['방문객수', 'electornic_car', 'y_n']].values

# 표준화
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# KMeans 하이퍼파라미터 튜닝
kmeans_param_grid = {
    'n_clusters': list(range(2, 11)),  # 2부터 10까지의 클러스터 수
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30, 40],  # n_init 값을 더 많이 시도
    'random_state': [42]
}

best_kmeans_score = -1
best_kmeans_params = {}

for params in ParameterGrid(kmeans_param_grid):
    kmeans = KMeans(n_clusters=params['n_clusters'], init=params['init'], n_init=params['n_init'], random_state=params['random_state'])
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    if score > best_kmeans_score:
        best_kmeans_score = score
        best_kmeans_params = params

print(f"Best KMeans Params: {best_kmeans_params}")
print(f"Best KMeans Silhouette Score: {best_kmeans_score}")

# 최적의 KMeans 모델로 클러스터링 수행
best_kmeans = KMeans(n_clusters=best_kmeans_params['n_clusters'], init=best_kmeans_params['init'], n_init=best_kmeans_params['n_init'], random_state=best_kmeans_params['random_state'])
kmeans_labels = best_kmeans.fit_predict(df_scaled)

# DBSCAN 하이퍼파라미터 튜닝
dbscan_param_grid = {
    'eps': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],  # eps 값을 더 넓은 범위로 시도
    'min_samples': [3, 5, 10, 15, 20, 30]   # min_samples 값을 더 넓은 범위로 시도
}

best_dbscan_score = -1
best_dbscan_params = {}

for params in ParameterGrid(dbscan_param_grid):
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(df_scaled)
    
    # 유효한 클러스터가 있는지 확인
    if len(set(labels)) > 1 and len(set(labels)) < len(df_scaled):  # 클러스터가 1개 이상, 모든 데이터가 노이즈가 아님
        score = silhouette_score(df_scaled, labels)
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_dbscan_params = params

print(f"Best DBSCAN Params: {best_dbscan_params}")
print(f"Best DBSCAN Silhouette Score: {best_dbscan_score}")

# 최적의 DBSCAN 모델로 클러스터링 수행
if best_dbscan_params:
    dbscan = DBSCAN(eps=best_dbscan_params['eps'], min_samples=best_dbscan_params['min_samples'])
    dbscan_labels = dbscan.fit_predict(df_scaled)
else:
    dbscan_labels = np.full(df_scaled.shape[0], -1)  # 모든 데이터를 노이즈로 간주

# 시각화
fig = plt.figure(figsize=(14, 6))

# KMeans 시각화 (3D 플롯)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter = ax1.scatter(df_scaled[:, 0], df_scaled[:, 1], df_scaled[:, 2], c=kmeans_labels, cmap='viridis')
ax1.set_title('KMeans Clustering Results (Standardized)')
ax1.set_xlabel('Standardized Visitor Count')
ax1.set_ylabel('Standardized EV Count')
ax1.set_zlabel('Charging Station')

# DBSCAN 시각화 (3D 플롯)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter = ax2.scatter(df_scaled[:, 0], df_scaled[:, 1], df_scaled[:, 2], c=dbscan_labels, cmap='viridis')
ax2.set_title('DBSCAN Clustering Results (Standardized)')
ax2.set_xlabel('Standardized Visitor Count')
ax2.set_ylabel('Standardized EV Count')
ax2.set_zlabel('Charging Station')

plt.tight_layout()
plt.show()
