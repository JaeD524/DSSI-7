import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

eda_df = pd.read_csv("C:\\Users\\dove8\\OneDrive\\바탕 화면\\DSSI\\한국환경공단_전기차 충전소 위치 및 운영정보(충전소 ID 포함)_20230531.csv", encoding='cp949')
print(eda_df.head(3))

# 대전광역시만 남기기
eda_df = eda_df[eda_df['주소'].str.startswith('대전광역시')]
print(eda_df.head(3))

# 이용 불가능한 충전소 제거
print(eda_df['이용자제한'].unique())
eda_df = eda_df[eda_df['이용자제한'].isin(['이용가능', '비공개'])]

# 동일한 주소 제거
eda_df = eda_df.drop_duplicates(subset=['주소'], keep='first')

# 시설구분(소)별 빈도수 구하기
df = pd.DataFrame(eda_df)

# '시설구분(소)' 열을 기준으로 그룹화하여 빈도수 계산
grouped_df = df.groupby('시설구분(소)').size().reset_index(name='Freq')

# 빈도수 기준으로 정렬
grouped_df = grouped_df.sort_values(by='Freq', ascending=False).reset_index(drop=True)

# 데이터프레임 출력
print(grouped_df)

# 한글 글꼴 설정
font_path = 'C:/Windows/Fonts/NanumGothic.ttf' 
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

# 빈도수 계산
value_counts = eda_df['시설구분(소)'].value_counts()
print(value_counts)

# 데이터프레임으로 변환 (시각화를 위해)
value_counts_df = value_counts.reset_index()
value_counts_df.columns = ['값', '빈도수']

# 시각화 설정
plt.figure(figsize=(12, 8))
sns.barplot(x='값', y='빈도수', data=value_counts_df, palette='viridis')

# 그래프 제목 및 축 레이블 설정
plt.title('전기차 충전소 시설 구분', fontsize=16)
plt.xlabel('시설구분', fontsize=14)
plt.ylabel('빈도수', fontsize=14)

# x축의 값 레이블 회전
plt.xticks(rotation=45, ha='right')

# 값 표시
for index, value in enumerate(value_counts_df['빈도수']):
    plt.text(index, value, str(value), ha='center', va='bottom')

# 그래프 표시
plt.tight_layout()
plt.show()



# 기종이 급속인 것과 완속인 경우 구분해서 빈도수 시각화

# 급속인 경우

# '기종(대)'가 급속인 경우 필터링
eda_df_rapid = eda_df[eda_df['기종(대)'] == '급속']

# 시설구분(소)별 빈도수 계산
value_counts_rapid = eda_df_rapid['시설구분(소)'].value_counts()

# 데이터프레임으로 변환
value_counts_rapid_df = value_counts_rapid.reset_index()
value_counts_rapid_df.columns = ['값', '빈도수']

# 시각화 설정
plt.figure(figsize=(12, 8))
sns.barplot(x='값', y='빈도수', data=value_counts_rapid_df, palette='viridis')

# 그래프 제목 및 축 레이블 설정
plt.title('전기차 충전소 시설 구분 (급속)', fontsize=16)
plt.xlabel('시설구분', fontsize=14)
plt.ylabel('빈도수', fontsize=14)

# x축의 값 레이블 회전
plt.xticks(rotation=45, ha='right')

# 값 표시
for index, value in enumerate(value_counts_rapid_df['빈도수']):
    plt.text(index, value, str(value), ha='center', va='bottom')

# 그래프 표시
plt.tight_layout()
plt.show()

# 완속인 경우

# '기종(대)'가 완속인 경우 필터링
eda_df_slow = eda_df[eda_df['기종(대)'] == '완속']

# 시설구분(소)별 빈도수 계산
value_counts_slow = eda_df_slow['시설구분(소)'].value_counts()

# 데이터프레임으로 변환
value_counts_slow_df = value_counts_slow.reset_index()
value_counts_slow_df.columns = ['값', '빈도수']

# 시각화 설정
plt.figure(figsize=(12, 8))
sns.barplot(x='값', y='빈도수', data=value_counts_slow_df, palette='viridis')

# 그래프 제목 및 축 레이블 설정
plt.title('전기차 충전소 시설 구분 (완속)', fontsize=16)
plt.xlabel('시설구분', fontsize=14)
plt.ylabel('빈도수', fontsize=14)

# x축의 값 레이블 회전
plt.xticks(rotation=45, ha='right')

# 값 표시
for index, value in enumerate(value_counts_slow_df['빈도수']):
    plt.text(index, value, str(value), ha='center', va='bottom')

# 그래프 표시
plt.tight_layout()
plt.show()
