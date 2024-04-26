#==============
# Map 지도 객체 활용
#==============
import folium
import pandas as pd

map_osm = folium.Map(location = [37.56020, 126.9753], zoom_start = 16)

map_osm.save('./map.html')
