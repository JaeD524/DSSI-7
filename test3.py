import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from shapely.ops import unary_union
import webbrowser
import os
import re

# 파일 경로
charging_file_path = 'C:\\Users\\user\\Desktop\\대학교\\2024 여름계절학기\\DSSI\\대회 준비\\ChargingStationData.xlsx'
boundary_file_path = 'C:\\Users\\user\\Desktop\\대학교\\2024 여름계절학기\\DSSI\\대회 준비\\boundary.geojson' # 지도파일

# 엑셀 파일을 읽어들여 데이터프레임으로 변환합니다.
df = pd.read_excel(charging_file_path)
# print(df)
# df = df[(df['시설구분(소)'] == '공영주차장')]
df = df[df['시설구분(소)'].isin(['공영주차장', '공원', '공원주차장', '마트(쇼핑몰)', '백화점', '생태공원', '음식점', '주유소', '카페'])]
df = df[(df['기종(대)'] == '급속')]

# '위도경도' 컬럼을 쉼표를 기준으로 분리하여 'Latitude'와 'Longitude' 컬럼 생성
if '위도경도' in df.columns:
    df[['Latitude', 'Longitude']] = df['위도경도'].str.split(',', expand=True)

    # 데이터 타입을 float으로 변환
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

# '동' 열의 결측치를 '주소'로 채우기
# df.loc[df['동'].isna(), '동'] = df['주소']
# '동' 열의 결측치를 '주소'로 채우기
df['동'] = df['동'].fillna(df['주소'])

# 정규 표현식 패턴 (동을 추출)
pattern = r'\b(\w+동)\b'

# 정규 표현식 패턴을 사용하여 '동' 열에서 동 이름을 추출
df['Dong'] = df['동'].str.extract(pattern, expand=False)

# 데이터 준비
data_set_column = {
    #'InstallYear': df['설치년도'],
    'Address': df['주소'],
    #'City': df['시도'],
    #'District': df['군구'],
    'Dong': df['Dong'],
    #'Division': df['시설구분(소)'],
    'Latitude': df['Latitude'],
    'Longitude': df['Longitude']
}

df_set_column = pd.DataFrame(data_set_column)

# 모든 행을 보기 위한 설정
pd.set_option('display.max_rows', None)

# 동 없는 행들만 보기
# print(df_set_column[df_set_column['Dong'].isna()])

# GeoDataFrame 생성
geometry = [Point(xy) for xy in zip(df_set_column['Longitude'], df_set_column['Latitude'])]
geo_df = gpd.GeoDataFrame(df_set_column, geometry=geometry)

# 대전광역시 경계 데이터 읽기
daejeon = gpd.read_file(boundary_file_path)

# 법정동 코드 범위를 사용하여 대전광역시 경계 선택
daejeon_boundaries = daejeon[(daejeon['adm_cd2'] >= '30110515') & (daejeon['adm_cd2'] <= '30230610')].copy()
# print(daejeon_boundaries)

# print(df_set_column.shape)
# print(df_set_column)

# 충전소 개수를 동별로 집계
charging_station_count = df_set_column.groupby('Dong').size().reset_index(name='Count')
print(charging_station_count)

# 합계 구한 거 맞는지 확인
column_sums = charging_station_count['Count'].sum()
print(column_sums)

# adm_nm에서 동 이름 추출
#daejeon_boundaries['Dong'] = daejeon_boundaries['adm_nm'].str.extract(pattern, expand=False)

# 동 이름 통합 함수 정의
#def normalize_dong(dong_name):
    #if pd.isna(dong_name):
        #return None
    # '판암1동', '판암2동'과 같은 동을 '판암동'으로 통합
    # '유천1동', '유천2동'을 '유천동'으로 통합
    #normalized_name = re.sub(r'(판암|유천|문화|도마|월평|갈마|관저|법)\d*동', r'\1동', dong_name)
    #normalized_name = re.sub(r'은행선화동', '선화동', normalized_name)
    #return normalized_name

# 동 이름 통합 적용
#daejeon_boundaries['Dong'] = daejeon_boundaries['Dong'].apply(normalize_dong)
# print(daejeon_boundaries['Dong'])

# 경계 데이터와 충전소 개수 데이터 병합
#daejeon_boundaries = daejeon_boundaries.merge(charging_station_count, on='Dong', how='left')
# print(daejeon_boundaries)

# 동별로 중복된 데이터를 제거하고, 하나의 레코드만 남기기
# 가장 큰 'Count' 값을 가진 동만 남깁니다
#daejeon_boundaries = daejeon_boundaries.sort_values(by='Count', ascending=False).drop_duplicates(subset='Dong')
# print(daejeon_boundaries)

# 결측치 0으로 수정
#daejeon_boundaries['Count'] = daejeon_boundaries['Count'].fillna(0)
# print(daejeon_boundaries)

# 지도 생성 (대전광역시 경계를 중심으로 설정)
#bounds = daejeon_boundaries.total_bounds  # [minx, miny, maxx, maxy]
#m = folium.Map(
    #location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
    #zoom_start=11,
    #tiles='cartodbpositron'  # OpenStreetMap, cartodbpositron
#)

# 대전광역시 경계 표시 (Choropleth 사용)
#folium.Choropleth(
    #geo_data=daejeon_boundaries,
    #name='choropleth',
    #data=daejeon_boundaries,
    #columns=['adm_cd2', 'Count'],
    #key_on='feature.properties.adm_cd2',
    #fill_color='YlOrRd',
    #fill_opacity=0.7,
    #line_opacity=0.2,
    #legend_name='전기차 충전소 수'
#).add_to(m)

# 지도 저장 및 자동 열기
#output_file = 'ChargingMap.html'
#m.save(output_file)

#file_path = os.path.abspath(output_file)
#webbrowser.open(f'file://{file_path}')

#print(f"지도 파일이 '{output_file}'로 저장되었습니다. 웹 브라우저에서 열립니다.")

