import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from shapely.ops import unary_union
import webbrowser
import os
import chardet


# 파일 경로
charging_file_path = 'C:\\Users\\user\\Desktop\\대학교\\2024 여름계절학기\\DSSI\\대회 준비\\ChargingStationData.csv'

# 시도할 인코딩 목록
encodings = ['cp949', 'utf-8', 'iso-8859-1']

for enc in encodings:
    try:
        data = pd.read_csv(charging_file_path, encoding=enc, low_memory=False)
        df = pd.DataFrame(data).head(100) # 우선 100행만
        # print(f"Successfully read the file with encoding {enc}")
        # print(df.head(10))

        # '위도경도' 컬럼을 쉼표를 기준으로 분리하여 'Latitude'와 'Longitude' 컬럼 생성
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
        print(df_set_column)

        break
    except Exception as e:
        print(f"Error reading file with encoding {enc}: {e}")
