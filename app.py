
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import zipfile
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium
import matplotlib.colors as mcolors

st.set_page_config(page_title="Texas Water Quality Dashboard", page_icon="🌊", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f4f9ff; }
    .css-18e3th9 { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #0b5394; }
    .stSelectbox>div>div { background-color: #e3f2fd; border-radius: 0.5rem; }
    .stDataFrameContainer { border-radius: 1rem; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

st.title("💧 Texas Coastal Water Quality Dashboard (1990–1991 Historical Data)")

# Load Google Sheets Data
sheet_url = "https://docs.google.com/spreadsheets/d/1JAQLzSpbU2nMVb4Pe1XUA2lxU3a1XcY4oUYS8UhIaiA/export?format=csv&gid=1208759629"
try:
    df = pd.read_csv(sheet_url)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

df = df.rename(columns={
    "Name": "StationName",
    "Latitude": "Lat",
    "Longitude": "Lon",
    "Sample Date": "ActivityStartDate"
})
df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"], errors='coerce')
df = df.dropna(subset=["Lat", "Lon", "ActivityStartDate"])

value_cols = [col for col in df.columns if col not in ["StationName", "Description", "Basin", "County", "Lat", "Lon", "TCEQ Stream Segment", "ActivityStartDate"]]
long_df = df.melt(
    id_vars=["StationName", "Lat", "Lon", "ActivityStartDate"],
    value_vars=value_cols,
    var_name="CharacteristicName",
    value_name="ResultMeasureValue"
)
long_df["ResultMeasureValue"] = pd.to_numeric(long_df["ResultMeasureValue"], errors="coerce")
long_df["StationKey"] = long_df["Lat"].astype(str) + "," + long_df["Lon"].astype(str)
long_df = long_df.dropna(subset=["ResultMeasureValue", "CharacteristicName"])

# Load shapefile
shp_folder = "shp_extracted"
shp_zip = "filtered_11_counties.zip"
if not os.path.exists(shp_folder):
    with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
        zip_ref.extractall(shp_folder)

shp_files = glob.glob(os.path.join(shp_folder, "**", "*.shp"), recursive=True)
if not shp_files:
    st.error("❌ No shapefile found.")
    st.stop()

gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)

available_params = sorted(long_df["CharacteristicName"].dropna().unique())
selected_param = st.selectbox("📌 Select a Water Quality Parameter", available_params)
filtered_df = long_df[long_df["CharacteristicName"] == selected_param]
latest_values = (
    filtered_df.sort_values("ActivityStartDate")
    .groupby("StationKey")
    .tail(1)
    .set_index("StationKey")
)

st.subheader(f"🗺️ Latest Measurements of {selected_param}")
map_center = [filtered_df["Lat"].mean(), filtered_df["Lon"].mean()]
m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

folium.GeoJson(
    gdf,
    style_function=lambda x: {
        "fillColor": "#0b5394",
        "color": "#0b5394",
        "weight": 2,
        "fillOpacity": 0.3,
    },
    name="Texas Coastal Counties"
).add_to(m)

for key, row in latest_values.iterrows():
    lat, lon = row["Lat"], row["Lon"]
    val = row["ResultMeasureValue"]
    popup_html = f"<b>Station:</b> {row['StationName']}<br><b>{selected_param}:</b> {val:.2f}<br><b>Date:</b> {row['ActivityStartDate'].strftime('%Y-%m-%d')}"
    folium.CircleMarker(
        location=[lat, lon],
        radius=5 + min(max(val, 0), 100) ** 0.5,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(m)

st_data = st_folium(m, width=1300, height=600)
