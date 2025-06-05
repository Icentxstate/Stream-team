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
from branca.colormap import linear
from streamlit_folium import st_folium
import matplotlib.colors as mcolors
import random

st.set_page_config(layout="wide")
st.title("🌊 Texas Coastal Hydrologic Monitoring Dashboard")

# --- Paths ---
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"
rivers_zip = "selected_major_rivers_shp.zip"
rivers_folder = "rivers_extracted"

# --- Caching functions ---
@st.cache_data
def load_water_quality(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')
    return df

@st.cache_resource
def load_shapefile(zip_path, folder_name):
    if not os.path.exists(folder_name):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_name)
    shp_files = glob.glob(os.path.join(folder_name, "**", "*.shp"), recursive=True)
    if not shp_files:
        return None, None
    attrs = gpd.read_file(shp_files[0], ignore_geometry=True)
    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
    return gdf, attrs

# --- Load data ---
df = load_water_quality(csv_path)
gdf, _ = load_shapefile(shp_zip, shp_folder)
gdf_rivers, gdf_rivers_attrs = load_shapefile(rivers_zip, rivers_folder)

# --- Prepare long format ---
exclude_cols = ["Name", "Description", "Basin", "County", "Latitude", "Longitude", "TCEQ Stream Segment", "Sample Date"]
value_cols = [col for col in df.columns if col not in exclude_cols]
df_long = df.melt(
    id_vars=["Name", "Latitude", "Longitude", "Sample Date"],
    value_vars=value_cols,
    var_name="CharacteristicName",
    value_name="ResultMeasureValue"
)
df_long["ActivityStartDate"] = pd.to_datetime(df_long["Sample Date"], errors='coerce')
df_long["ResultMeasureValue"] = pd.to_numeric(df_long["ResultMeasureValue"], errors="coerce")
df_long["StationKey"] = df_long["Latitude"].astype(str) + "," + df_long["Longitude"].astype(str)
df_long = df_long.dropna(subset=["ActivityStartDate", "ResultMeasureValue", "CharacteristicName"])

# --- UI ---
available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.selectbox("📌 Select a Water Quality Parameter", available_params)
filtered_df = df_long[df_long["CharacteristicName"] == selected_param]

latest_values = (
    filtered_df.sort_values("ActivityStartDate")
    .groupby("StationKey")
    .tail(1)
    .set_index("StationKey")
)

# --- Colormap ---
min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = linear.RdYlBu_11.scale(min_val, max_val)
colormap.caption = f"{selected_param} Value Range"

# --- Map ---
st.subheader(f"🗺️ Latest Measurements of {selected_param}")
map_center = [filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()]

with st.spinner("Rendering interactive map..."):
    m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

    # Add counties safely
    if gdf is not None:
        gdf_clean = gdf[[col for col in gdf.columns if gdf[col].dtype.kind in 'ifO']].copy()
        gdf_clean["geometry"] = gdf["geometry"]
        folium.GeoJson(
            gdf_clean,
            name="Coastal Counties",
            style_function=lambda x: {
                "fillColor": "#0b5394",
                "color": "#0b5394",
                "weight": 2,
                "fillOpacity": 0.3,
            },
        ).add_to(m)

    # Add rivers
    if gdf_rivers is not None and "STRM_NM" in gdf_rivers.columns:
        top_rivers = gdf_rivers_attrs["STRM_NM"].value_counts().nlargest(20).index
        gdf_rivers = gdf_rivers[gdf_rivers["STRM_NM"].isin(top_rivers)]
        color_palette = list(mcolors.CSS4_COLORS.values())
        random.shuffle(color_palette)
        river_colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(top_rivers)}
        river_group = folium.FeatureGroup(name="Major Rivers", show=True).add_to(m)

        for _, row in gdf_rivers.iterrows():
            name = row["STRM_NM"] if pd.notnull(row["STRM_NM"]) else "Unnamed River"
            color = river_colors.get(name, "#0077b6")
            if row.geometry.type == "LineString":
                segments = [row.geometry]
            elif row.geometry.type == "MultiLineString":
                segments = row.geometry.geoms
            else:
                continue
            for seg in segments:
                coords = [(lat, lon) for lon, lat in seg.coords]
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=3,
                    tooltip=name,
                    popup=folium.Popup(f"<b>{name}</b>", max_width=250)
                ).add_to(river_group)

    # Add stations
    station_group = folium.FeatureGroup(name="WQ Stations", show=True).add_to(m)
    for key, row in latest_values.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row["ResultMeasureValue"]
        color = colormap(val)
        popup_html = f"<b>Station:</b> {row['Name']}<br><b>{selected_param}:</b> {val:.2f}<br><b>Date:</b> {row['ActivityStartDate'].strftime('%Y-%m-%d')}"
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(station_group)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st_data = st_folium(m, width=1300, height=600)

# --- Click interaction (summary only) ---
if st_data and "last_object_clicked" in st_data:
    lat = st_data["last_object_clicked"].get("lat")
    lon = st_data["last_object_clicked"].get("lng")
    if lat and lon:
        st.markdown("---")
        st.markdown("### 📋 Station Summary")
        coords = f"{lat},{lon}"
        st.write(f"📍 Coordinates: `{lat:.5f}, {lon:.5f}`")
        ts_df = df_long[df_long["StationKey"] == coords]
        if ts_df.empty:
            st.warning("No data found for this station.")
        else:
            grouped = ts_df.groupby("CharacteristicName")
            summaries = []
            for param, group in grouped:
                group = group.sort_values("ActivityStartDate")
                start_date = group["ActivityStartDate"].min()
                end_date = group["ActivityStartDate"].max()
                mean_val = group["ResultMeasureValue"].mean()
                last_val = group.iloc[-1]["ResultMeasureValue"]
                last_date = group.iloc[-1]["ActivityStartDate"]
                summaries.append({
                    "Parameter": param,
                    "Start Date": start_date.date(),
                    "End Date": end_date.date(),
                    "Mean Value": round(mean_val, 2),
                    "Last Value": round(last_val, 2),
                    "Last Date": last_date.date()
                })
            summary_df = pd.DataFrame(summaries)
            st.dataframe(summary_df)
