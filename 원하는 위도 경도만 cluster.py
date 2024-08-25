import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import ParameterGrid
import numpy as np

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 경우: 맑은 고딕 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 데이터 로드
cluster_df = pd.read_excel("C:\\Users\\dove8\\OneDrive\\바탕 화면\\DSSI\\my_data.xlsx")
print(cluster_df.head(3))

# y_n이 0일 때
cluster_df = cluster_df[cluster_df['y_n'] == 0]

# 필터링할 위도와 경도 값 목록
lat_long_values = [
    (36.37887, 127.3049),
    (36.36181, 127.3529),
    (36.35451, 127.2905),
    (36.33931, 127.3875),
    (36.34833, 127.3863),
    (36.21494, 127.4410),
    (36.35468, 127.4443),
    (36.41318, 127.3383),
    (36.35279, 127.3451),
    (36.37887, 127.3049),
    (36.32492, 127.4276),
    (36.32762, 127.4276),
    (36.37538, 127.3919),
    (36.45025, 127.4665),
    (36.35995, 127.3083),
    (36.37362, 127.5045),
    (36.32672, 127.4209),
    (36.36718, 127.3194),
    (36.35553, 127.3774),
    (36.32582, 127.4242)
]

# 위도와 경도 값으로 데이터 필터링
filtered_df = cluster_df[cluster_df[['LatitudeRound', 'LongitudeRound']].apply(tuple, axis=1).isin(lat_long_values)]

# 위도와 경도 데이터만 추출
df = filtered_df[['LatitudeRound', 'LongitudeRound']].values

# KMeans 하이퍼파라미터 튜닝 및 실루엣 계수 저장
kmeans_param_grid = {
    'n_clusters': range(2, 8),  # 클러스터 개수를 2~7로 설정
    'init': ['k-means++'],
    'n_init': [10],
    'random_state': [42]
}

best_kmeans_score = -1
best_kmeans_params = {}
silhouette_scores = []

for params in ParameterGrid(kmeans_param_grid):
    kmeans = KMeans(n_clusters=params['n_clusters'], init=params['init'], n_init=params['n_init'], random_state=params['random_state'])
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    silhouette_scores.append((params['n_clusters'], score))
    
    if score > best_kmeans_score:
        best_kmeans_score = score
        best_kmeans_params = params

print(f"Best KMeans Params: {best_kmeans_params}")
print(f"Best KMeans Silhouette Score: {best_kmeans_score}")

# 최적의 KMeans 모델로 클러스터링 수행
best_kmeans = KMeans(n_clusters=best_kmeans_params['n_clusters'], init=best_kmeans_params['init'], n_init=best_kmeans_params['n_init'], random_state=best_kmeans_params['random_state'])
kmeans_labels = best_kmeans.fit_predict(df)

# 실루엣 계수 시각화
plt.figure(figsize=(10, 7))

for i, (n_clusters, score) in enumerate(silhouette_scores):
    plt.subplot(2, 3, i + 1)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(df)
    
    silhouette_vals = silhouette_samples(df, cluster_labels)
    y_lower, y_upper = 0, 0
    
    for j in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[cluster_labels == j]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)
        y_lower += len(cluster_silhouette_vals)
    
    plt.axvline(x=score, color="red", linestyle="--")
    plt.title(f"Number of Cluster: {n_clusters}\nSilhouette Score: {round(score, 3)}")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.tight_layout()

plt.show()

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
plt.title('KMeans 클러스터링 결과')
plt.xlabel('위도')
plt.ylabel('경도')

# DBSCAN 시각화
plt.subplot(1, 2, 2)
plt.scatter(df[:, 0], df[:, 1], c=dbscan_labels, cmap='viridis', marker='o')
plt.title('DBSCAN 클러스터링 결과')
plt.xlabel('위도')
plt.ylabel('경도')

plt.tight_layout()
plt.show()

# 각 클러스터의 중앙값 계산 함수
def calculate_cluster_medians(data, labels):
    unique_labels = np.unique(labels)
    cluster_medians = {}
    
    for label in unique_labels:
        # 노이즈 점 (-1)은 제외
        if label == -1:
            continue
        
        # 각 클러스터에 속하는 데이터 포인트 선택
        cluster_points = data[labels == label]
        
        # 중앙값 계산
        median = np.median(cluster_points, axis=0)
        
        cluster_medians[label] = median
    
    return cluster_medians

# KMeans 클러스터의 중앙값 계산
kmeans_cluster_medians = calculate_cluster_medians(df, kmeans_labels)
print("KMeans 클러스터 중앙값:")
for label, median in kmeans_cluster_medians.items():
    print(f"Cluster {label}: {median}")

# DBSCAN 클러스터의 중앙값 계산
dbscan_cluster_medians = calculate_cluster_medians(df, dbscan_labels)
print("\nDBSCAN 클러스터 중앙값:")
for label, median in dbscan_cluster_medians.items():
    print(f"Cluster {label}: {median}")
