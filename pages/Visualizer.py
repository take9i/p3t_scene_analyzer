# pylint: disable=missing-docstring
from os import path
from glob import glob
import json
import base64

import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

BASE_DIR = "_route_analyzed"

st.set_page_config(page_title="Visualizer")


def get_bname(f):
    return path.basename(f).split(".")[0]


st.markdown("# Visualizer")

scenes = [path.basename(f) for f in glob(f"{BASE_DIR}/*")]
scene = st.selectbox("select scene", scenes)

jsons = [
    json.load(open(f, encoding="utf-8"))
    for f in glob(f"{BASE_DIR}/{scene}/jsons/*.json")
]

p_min, p_max = st.sidebar.slider("Pitch", -90, 10, (-90, 0))
r_min, r_max = st.sidebar.slider("Range", 1, 500, (1, 500))

jsons = sorted(
    [
        js
        for js in jsons
        if js["pitch"] >= p_min
        and js["pitch"] < p_max
        and js["range"] >= r_min
        and js["range"] < r_max
    ],
    key=lambda js: js["name"],
)

names = [js["name"] for js in jsons]
loca = [jsons[0]["lat"], jsons[0]["lon"]] if len(jsons) else [35.6895, 139.6917]
m = folium.Map(location=loca, zoom_start=14)
fg = folium.FeatureGroup(name="Pika")
# for n, row in adf[["geometry", "name", "height"]].iterrows():
for js in jsons:
    lat, lon, name = js["lat"], js["lon"], js["name"]
    with open(f"{BASE_DIR}/{scene}/overlayed_images/{js['name']}.png", "rb") as f:
        data = f.read()
    img_str = base64.b64encode(data).decode()
    # html = f'<img src="./{BASE_DIR}/{scene}/overlayed_images/{name}.png" width="400" />'
    html = f'<img src="data:image/png;base64,{img_str}" width="400" />'
    marker = folium.Marker(location=(lat, lon), popup=html, tooltip=js["name"])
    fg.add_child(marker)
m.add_child(fg)
st_map = st_folium(m, width=700, height=450)

images = [f"{BASE_DIR}/{scene}/overlayed_images/{js['name']}.png" for js in jsons]
st.image(images, caption=names, width=200)

COLS = 3
rows = len(names) // COLS + 1 if len(names) % COLS else len(names) // COLS
classes = json.load(open("cityscapes_labels.json", encoding="utf-8"))["classes"][:-1]
fig, axes = plt.subplots(rows, COLS, figsize=(6, rows))
for i, js in enumerate(jsons):
    axes[i // COLS, i % COLS].bar(classes, js["histogram"])
st.pyplot(fig)
