import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from shapely.ops import unary_union
import webbrowser
import os
import chardet

# 가상환경 활성화
# python -m venv venv
# venv\Scripts\activate

# Git push
# git init
# git add [파일명]
# git commit -m "[commit 내역 문자]"
# git branch -M main
# git remote add origin [git url]
# git push -u origin main

# 파일 경로
charging_file_path = 'C:\\Users\\user\\Desktop\\대학교\\2024 여름계절학기\\DSSI\\대회 준비\\ChargingStationData.csv'
boundary_file_path = 'C:\\Users\\user\\Desktop\\대학교\\2024 여름계절학기\\DSSI\\대회 준비\\gyeongsangbukdo_boundary.geojson' # 지도파일

# 파일의 인코딩 감지
with open(charging_file_path, 'rb') as file:
    raw_data = file.read(10000)  # 파일의 처음 10,000 바이트만 읽어서 인코딩 감지
    result = chardet.detect(raw_data)
    encoding = result['encoding']

print(f"Detected encoding: {encoding}")

# 감지된 인코딩을 사용하여 파일 읽기
try:
    data = pd.read_csv(charging_file_path, encoding=encoding, low_memory=False)
except UnicodeDecodeError as e:
    print(f"Error reading file with detected encoding {encoding}: {e}")
    print("Trying with 'CP949' encoding...")
    try:
        data = pd.read_csv(charging_file_path, encoding='CP949', low_memory=False)
    except Exception as e:
        print(f"Error reading file with 'CP949' encoding: {e}")

try:
    df = pd.DataFrame(data)
    df = df[(df['시도'] == '대전광역시') & (df['시설구분(소)'] == '공영주차장')]

    # '위도경도' 컬럼을 쉼표를 기준으로 분리하여 'Latitude'와 'Longitude' 컬럼 생성
    if '위도경도' in df.columns:
        df[['Latitude', 'Longitude']] = df['위도경도'].str.split(',', expand=True)

        # 데이터 타입을 float으로 변환
        df['Latitude'] = df['Latitude'].astype(float)
        df['Longitude'] = df['Longitude'].astype(float)

    # 데이터 준비
    data_set_column = {
        'InstallYear': df['설치년도'],
        'City': df['시도'],
        'District': df['군구'],
        'Division': df['시설구분(소)'],
        'Latitude': df['Latitude'],
        'Longitude': df['Longitude']
    }

    df_set_column = pd.DataFrame(data_set_column)

    # GeoDataFrame 생성
    geometry = [Point(xy) for xy in zip(df_set_column['Longitude'], df_set_column['Latitude'])]
    geo_df = gpd.GeoDataFrame(df_set_column, geometry=geometry)

    # 대전광역시 경계 데이터 읽기
    daejeon = gpd.read_file(boundary_file_path)

    # 대전광역시 GeoDataFrame에 맞게 시각화
    daejeon_boundaries = daejeon[(daejeon['adm_cd2'] >= '30110515') & (daejeon['adm_cd2'] <= '30230610')]

    # GeoDataFrame의 모든 지오메트리를 결합 (deprecated warning 해결)
    combined_geometry = unary_union(daejeon_boundaries.geometry)

    # 지도 생성 (대전광역시 경계를 중심으로 설정)
    bounds = daejeon_boundaries.total_bounds  # [minx, miny, maxx, maxy]
    m = folium.Map(
        location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        zoom_start=11,
        tiles='cartodb positron'
    )

    # 대전광역시 경계 표시
    folium.GeoJson(combined_geometry).add_to(m)

    # 전기차충전소 데이터 시각화
    for _, row in geo_df.iterrows():
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=8,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.5,
        ).add_to(m)

    # 지도 저장 및 자동 열기
    output_file = 'ChargingMap.html'
    m.save(output_file)

    file_path = os.path.abspath(output_file)
    webbrowser.open(f'file://{file_path}')

    print(f"지도 파일이 '{output_file}'로 저장되었습니다. 웹 브라우저에서 열립니다.")

except Exception as e:
    print(f"Error processing the DataFrame: {e}")
