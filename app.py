# ✅ Streamlit App: Stable Map with Popup & Detail View (No JS) — Fixed GeoJSON Error
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

# --- Session State Setup ---
if "selected_coords" not in st.session_state:
    st.session_state.selected_coords = None

# --- File Paths ---
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

# --- Load CSV ---
df = pd.read_csv(csv_path, low_memory=False)
df = df.dropna(subset=["Latitude", "Longitude"])
df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')

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

# --- Load Shapefile ---
if not os.path.exists(shp_folder):
    with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
        zip_ref.extractall(shp_folder)

shp_files = glob.glob(os.path.join(shp_folder, "**", "*.shp"), recursive=True)
gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
gdf_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

# --- Fix GeoJson serialization error ---
gdf_safe = gdf[[col for col in gdf.columns if gdf[col].dtype.kind in 'ifO']].copy()
gdf_safe["geometry"] = gdf["geometry"]

# --- UI ---
st.title("🗺️ Texas Coastal Monitoring Dashboard")
selected_param = st.sidebar.multiselect("Select Parameters", sorted(df_long["CharacteristicName"].dropna().unique())))
filtered_df = df_long[df_long["CharacteristicName"].isin(selected_param)] if selected_param else df_long.copy()
selected_dates = st.sidebar.date_input("Select Date(s)", [])
if not selected_dates:
    latest_date = df_long["ActivityStartDate"].max()
    selected_dates = [latest_date]

filtered_df = filtered_df[filtered_df["ActivityStartDate"].dt.date.isin([d if isinstance(d, pd.Timestamp) else pd.to_datetime(d) for d in selected_dates])]

latest_values = filtered_df.sort_values("ActivityStartDate").groupby("StationKey").tail(1).set_index("StationKey")

min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = linear.RdYlBu_11.scale(min_val, max_val)

# --- Map ---
m = folium.Map(location=[(gdf_bounds[1]+gdf_bounds[3])/2, (gdf_bounds[0]+gdf_bounds[2])/2], zoom_start=7)
m.fit_bounds([[gdf_bounds[1], gdf_bounds[0]], [gdf_bounds[3], gdf_bounds[2]]])

folium.GeoJson(gdf_safe, style_function=lambda x: {
    "fillColor": "#0b5394",
    "color": "#0b5394",
    "weight": 2,
    "fillOpacity": 0.3
}).add_to(m)

for key, row in latest_values.iterrows():
    lat, lon = row["Latitude"], row["Longitude"]
    val = row["ResultMeasureValue"]
    popup_content = f"""
    <b>Station:</b> {row['Name']}<br>
    <b>{selected_param}:</b> {val:.2f}<br>
    <b>Date:</b> {row['ActivityStartDate'].strftime('%Y-%m-%d')}
    """
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color=colormap(val),
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup_content, max_width=250)
    ).add_to(m)

colormap.add_to(m)
st.markdown("### Click on a station for more details below")
st_data = st_folium(m, width=1300, height=600)

# --- Handle Click ---
if st_data and st_data.get("last_object_clicked"):
    coords = st_data["last_object_clicked"]
    st.session_state.selected_coords = f"{coords['lat']},{coords['lng']}"

# --- Show Details ---
if st.session_state.selected_coords:
    lat, lon = map(float, st.session_state.selected_coords.split(","))
    st.markdown("---")
    st.subheader(f"📍 Station Details: {lat:.5f}, {lon:.5f}")
    ts_df = df_long[df_long["StationKey"] == st.session_state.selected_coords].sort_values("ActivityStartDate")
    subparams = sorted(ts_df["CharacteristicName"].dropna().unique())
    selected = st.multiselect("Select parameters to plot", subparams, default=subparams[:1])

    if selected:
        plot_df = ts_df[ts_df["CharacteristicName"].isin(selected)]
        plot_df = plot_df.pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue").dropna(how="all")
        st.line_chart(plot_df)
        st.write("### Summary Statistics")
        st.dataframe(plot_df.describe().T)
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(plot_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Please select one or more parameters.")
