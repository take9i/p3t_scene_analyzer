import json
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium

@st.cache_data
def load_data():
    df = pd.read_json('http://localhost:5000/analyzed/whole-json', dtype={'name': 'str'})
    df['geometry'] = df.apply(lambda r: Point(r.lon, r.lat), axis=1)
    classes = json.load(open('cityscapes_labels.json'))['classes'][:-1]
    return gpd.GeoDataFrame(df, geometry='geometry', crs=4326), classes

gdf, classes = load_data()

h_min, h_max = st.sidebar.slider('Height', 800, 1500, (800, 1500))
p_min, p_max = st.sidebar.slider('Pitch', -90, 10, (-90, 0))
adf = gdf[(h_min <= gdf.height) & (gdf.height < h_max) & (p_min <= gdf.pitch) & (gdf.pitch < p_max)]

map = folium.Map(location=[gdf.iloc[0].lat, gdf.iloc[0].lon], zoom_start=13)
fg = folium.FeatureGroup(name="Pika")
for n, row in adf[[u'geometry', u'name', u'height']].iterrows():
    p, n, h = row[u'geometry'], row[u'name'], row[u'height']
    html = f'<img src="http://localhost:5000/analyzed/image/{n}.png" width="400" />'
    marker = folium.Marker(location=(p.y, p.x), popup=html, tooltip=n)
    fg.add_child(marker)
map.add_child(fg)

st_map = st_folium(map, width=700, height=450)
st.write(st_map['last_object_clicked_tooltip'])

# st.write(adf)

adf.sort_values('name', inplace=True)
# names = sorted(adf.name.to_list())
# images = [f"_analyzed/image/{n}.png" for n in names]
images = adf.name.apply(lambda n: f"_analyzed/image/{n}.png").to_list()
st.image(images, caption=adf.name.to_list(), width=200)


cols = 3
rows = len(adf)//cols+1 if len(adf) % cols else len(adf)//cols
fig, axes = plt.subplots(rows, cols, figsize=(6, rows))
bdf = adf.reset_index(drop=True)
for i in bdf.index:
    axes[i//cols, i%cols].bar(classes, bdf.iloc[i].histogram)
st.pyplot(fig)
