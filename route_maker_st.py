# pylint: disable=missing-docstring
import math
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import ops
from shapely.geometry import Point
import folium
from streamlit_folium import st_folium
import momepy
import networkx as nx

@st.cache_data
def load_data():
    df = gpd.read_file('./data/tokyo_west.gpkg')[['rID', 'ftCode', 'type', 'rdCtg', 'rnkWidth', 'geometry']]
    df = df[(df['type'] == '通常部') & (df['rnkWidth'] != '3m未満')]
    return df

def get_near_edge(p, geom):
    f = lambda p1, p2: math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    hp, tp = geom.coords[0], geom.coords[-1] 
    return hp if f(hp, p) < f(tp, p) else tp

df = load_data()

# h_min, h_max = st.sidebar.slider('Height', 800, 1500, (800, 1500))
# p_min, p_max = st.sidebar.slider('Pitch', -90, 10, (-90, 0))
# adf = gdf[(h_min <= gdf.height) & (gdf.height < h_max) & (p_min <= gdf.pitch) & (gdf.pitch < p_max)]

center = st.session_state['center'] if 'center' in st.session_state else [35.6814, 139.7671]
zoom = st.session_state['zoom'] if 'zoom' in st.session_state else 13
# m = folium.Map(location=[35.6814, 139.7671], zoom_start=13)
m = folium.Map(location=center, zoom_start=zoom)

folium.plugins.Draw(
    export=False,
    draw_options={
        'polygon': False,
        'rectangle': False,
        'circle': False,
        'marker': False,
        'circlemarker': False,
    }
).add_to(m)

if 'path_line' in st.session_state:
    path_line = st.session_state['path_line']
    poly_line = folium.PolyLine(locations=path_line, weight=5)
    poly_line.add_to(m)

st_map = st_folium(m, width=700, height=450)
# st.write(st_map)

sw, ne = st_map['bounds']['_southWest'], st_map['bounds']['_northEast']
adf = df.cx[sw['lng']:ne['lng'], sw['lat']:ne['lat']]

if st_map['all_drawings'] and len(st_map['all_drawings']):
    line = st_map['all_drawings'][0]['geometry']['coordinates']
    edge_df = gpd.GeoDataFrame(geometry=[Point(line[0]), Point(line[1])], crs="epsg:6668")
    # st.write(edge_df)

    nearest_df = gpd.sjoin_nearest(adf, edge_df, distance_col='distance', how='inner')
    # st.write(nearest_df)
    head_row = nearest_df[nearest_df.index_right == 0].sort_values('distance').iloc[0]
    head_p = get_near_edge(line[0], head_row.geometry)
    tail_row = nearest_df[nearest_df.index_right == 1].sort_values('distance').iloc[0]
    tail_p = get_near_edge(line[-1], tail_row.geometry)
    G = momepy.gdf_to_nx(adf, approach="primal")
    nodes = nx.shortest_path(G, source=head_p, target=tail_p)

    path_df = pd.DataFrame([list(G[hn][tn].values())[0] for hn, tn in zip(nodes[:-1], nodes[1:])]).pipe(gpd.GeoDataFrame)
    path_line = ops.linemerge(path_df.geometry.unary_union)
    path_line = ops.transform(lambda x, y: (y, x), path_line)
    # st.write(list(path_line.coords))
    st.session_state['path_line'] = list(path_line.coords)
    st.session_state['center'] = [st_map['center']['lat'], st_map['center']['lng']]
    st.session_state['zoom'] = st_map['zoom']
    st.rerun()
