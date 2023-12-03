# pylint: disable=missing-docstring,disable=redefined-outer-name
from os import makedirs
import math
import json
import time
from datetime import datetime as dt
from functools import partial
from io import BytesIO

import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely import ops
from shapely.geometry import Point, LineString
import folium
from streamlit_folium import st_folium
import momepy
import networkx as nx
from pyproj import Transformer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from PIL import Image
import numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch import nn

CHROME_DRIVER_PATH = (
    r"/Users/suika/Projects/GPTxGIS/codes/capturer/chromedriver-mac-arm64/chromedriver"
    # r"C:\Users\t_osada23053\Documents\gpt\p3t_scene_analyzer\chromedriver-win64\chromedriver.exe"
)
BASE_DIR = "_route_analyzed"

st.set_page_config(page_title="Route Maker")


@st.cache_data
def load_nw_data():
    df = gpd.read_file("./data/tokyo_west.gpkg")[
        ["rID", "ftCode", "type", "rdCtg", "rnkWidth", "geometry"]
    ].to_crs(4612)
    df = df[(df["type"] == "通常部") & (df["rnkWidth"] != "3m未満")]
    return df


def get_camera_params(route, pitch=-40, the_range=300, step_meter=100, front_pad=10):
    def get_heading(p, q):
        p, q = p.coords[0], q.coords[0]
        dx, dy = q[0] - p[0], q[1] - p[1]
        r = math.atan2(dy, dx) * 180 / math.pi
        return int(round(90 - r))

    transformer = Transformer.from_crs("EPSG:4612", "EPSG:2451", always_xy=True)
    rtransformer = Transformer.from_crs("EPSG:2451", "EPSG:4612", always_xy=True)

    m_route = LineString(transformer.itransform(route.coords))
    distances = [l for l in range(20, int(m_route.length), step_meter)]
    locations = list(
        rtransformer.itransform([m_route.interpolate(l).coords[0] for l in distances])
    )
    headings = [
        get_heading(m_route.interpolate(l), m_route.interpolate(l + front_pad))
        for l in distances
    ]
    return [
        (lon, lat, heading, pitch, the_range)
        for (lon, lat), heading in zip(locations, headings)
    ]


# ---


def predict_with_model(feature_extractor, model, image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(image.size[1], image.size[0]),  # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
    return predicted_mask[0]


def get_mask_and_overlayed_images(palette, image, prediction):
    color_map = {i: k for i, k in enumerate(palette)}
    vis = np.zeros(prediction.shape + (3,))
    for i, _c in color_map.items():
        vis[prediction == i] = color_map[i]
    mask = Image.fromarray(vis.astype(np.uint8))
    overlayed = Image.blend(image.convert("RGB"), mask.convert("RGB"), 0.5)
    return mask, overlayed


@st.cache_resource
def setup_analyzer(model, label):
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model)
    model = SegformerForSemanticSegmentation.from_pretrained(model)
    predict = partial(predict_with_model, feature_extractor, model)
    labels = json.load(open(label, encoding="utf-8"))
    decode = partial(get_mask_and_overlayed_images, labels["palette"])
    return predict, labels, decode


predict, labels, decode = setup_analyzer(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", "cityscapes_labels.json"
)


# ---


def get_path_line(line, df):
    def get_near_edge(p, geom):
        f = lambda p1, p2: math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        hp, tp = geom.coords[0], geom.coords[-1]
        return hp if f(hp, p) < f(tp, p) else tp

    edge_df = gpd.GeoDataFrame(
        geometry=[Point(line[0]), Point(line[1])], crs="EPSG:4612"
    )

    sw, ne = st_map["bounds"]["_southWest"], st_map["bounds"]["_northEast"]
    adf = df.cx[sw["lng"] : ne["lng"], sw["lat"] : ne["lat"]]
    nearest_df = gpd.sjoin_nearest(adf, edge_df, distance_col="distance", how="inner")

    head_row = nearest_df[nearest_df.index_right == 0].sort_values("distance").iloc[0]
    head_p = get_near_edge(line[0], head_row.geometry)
    tail_row = nearest_df[nearest_df.index_right == 1].sort_values("distance").iloc[0]
    tail_p = get_near_edge(line[-1], tail_row.geometry)
    G = momepy.gdf_to_nx(adf, approach="primal")
    nodes = nx.shortest_path(G, source=head_p, target=tail_p)
    path_df = pd.DataFrame(
        [list(G[hn][tn].values())[0] for hn, tn in zip(nodes[:-1], nodes[1:])]
    ).pipe(gpd.GeoDataFrame)
    path_line = ops.linemerge(path_df.geometry.unary_union)
    return list(path_line.coords)


# ---

st.markdown("# Route Maker")

nw_df = load_nw_data()

center = st.session_state.get("map_center", [35.6895, 139.6917])
zoom = st.session_state.get("map_zoom", 14)
m = folium.Map(location=center, zoom_start=zoom)
folium.plugins.Draw(
    export=False,
    draw_options={
        "polygon": False,
        "rectangle": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
    },
).add_to(m)
if "src_line" in st.session_state:
    coords = [(y, x) for x, y in st.session_state["src_line"]]
    folium.PolyLine(locations=coords, color="red", weight=5).add_to(m)
if "path_line" in st.session_state:
    coords = [(y, x) for x, y in st.session_state["path_line"]]
    folium.PolyLine(locations=coords, color="blue", weight=5).add_to(m)
st_map = st_folium(m, width=700, height=450)
# st.write(st_map)

if "src_line" in st.session_state and "path_line" in st.session_state:
    if st.button("Capture"):
        scene = dt.now().strftime("%Y%m%d_%H%M%S")  # tentative
        makedirs(f"{BASE_DIR}/{scene}", exist_ok=True)
        makedirs(f"{BASE_DIR}/{scene}/src_images", exist_ok=True)
        makedirs(f"{BASE_DIR}/{scene}/jsons", exist_ok=True)
        makedirs(f"{BASE_DIR}/{scene}/overlayed_images", exist_ok=True)

        camera_params = get_camera_params(
            LineString(st.session_state["src_line"])
        ) + get_camera_params(
            LineString(st.session_state["path_line"]), pitch=-10, the_range=25
        )
        st.write(f"capturing {len(camera_params)} 'on road view' images")

        driver = webdriver.Chrome(
            service=Service(CHROME_DRIVER_PATH), options=Options()
        )
        for i, (lon, lat, h, p, r) in enumerate(camera_params):
            driver.get(
                f"http://localhost:5173/?lat={lat}&lon={lon}&heading={h}&pitch={p}&range={r}"
            )
            time.sleep(10)
            sshot = driver.get_screenshot_as_png()
            src_img = Image.open(BytesIO(sshot)).convert("RGB")

            prediction = predict(src_img)
            _mask, overlayed_img = decode(src_img, prediction)
            histogram, _bins = np.histogram(prediction, range(len(labels["classes"])))

            src_img.save(f"{BASE_DIR}/{scene}/src_images/{i:03}.png")
            overlayed_img.save(f"{BASE_DIR}/{scene}/overlayed_images/{i:03}.png")
            result = {
                "name": f"{i:03}",
                "lon": lon,
                "lat": lat,
                "heading": h,
                "pitch": p,
                "range": r,
                "width": src_img.width,
                "height": src_img.height,
                "histogram": histogram.astype(np.int32).tolist(),
            }
            json.dump(
                result,
                open(f"{BASE_DIR}/{scene}/jsons/{i:03}.json", "w", encoding="utf-8"),
            )
            st.image(overlayed_img, caption=f"#{i:03}", width=200)
        driver.quit()


if st_map["all_drawings"] and len(st_map["all_drawings"]):
    the_line = st_map["all_drawings"][0]["geometry"]["coordinates"]
    the_path_line = get_path_line(the_line, nw_df)
    st.session_state["src_line"] = the_line
    st.session_state["path_line"] = the_path_line
    st.session_state["map_center"] = [st_map["center"]["lat"], st_map["center"]["lng"]]
    st.session_state["map_zoom"] = st_map["zoom"]
    st.rerun()
