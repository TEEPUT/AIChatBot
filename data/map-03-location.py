#==============
# 지도에 매장 위치 표기
#==============
import folium
import pandas as pd

CB_geoData = pd.read_csv('./(정리)SHP-CSV변환(2023년 12월 22일 14시 41분 55초).csv')
map_CB = folium.Map(location = [37.560284, 126.975334], zoom_start = 15)

for i, place in CB_geoData.iterrows():
    mk = folium.Marker(location = [place['_Y'], place['_X']], popup = place['place'],
        icon = folium.Icon(color = 'red', icon = 'star'))
    mk.add_to(map_CB)

map_CB.save('./map-location.html')