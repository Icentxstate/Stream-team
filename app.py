
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

st.set_page_config(layout="wide")
st.title("🌊 Texas Coastal Hydrologic Monitoring Dashboard")

# --- Paths ---
csv_path = "Datamap.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

# --- Load CSV ---
try:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

# --- Long Format Conversion ---
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

# --- Load shapefile ---
if not os.path.exists(shp_folder):
    with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
        zip_ref.extractall(shp_folder)

shp_files = glob.glob(os.path.join(shp_folder, "**", "*.shp"), recursive=True)
if not shp_files:
    st.error("❌ No shapefile found.")
    st.stop()

gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
gdf_safe = gdf[[col for col in gdf.columns if gdf[col].dtype.kind in 'ifO']].copy()
gdf_safe["geometry"] = gdf["geometry"]

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

# --- Colormap based on parameter values ---
min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = linear.RdYlBu_11.scale(min_val, max_val)
colormap.caption = f"{selected_param} Value Range"

# --- Map ---
st.subheader(f"🗺️ Latest Measurements of {selected_param}")
map_center = [filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

# Add shapefile
folium.GeoJson(
    gdf_safe,
    style_function=lambda x: {
        "fillColor": "#0b5394",
        "color": "#0b5394",
        "weight": 2,
        "fillOpacity": 0.3,
    },
    name="Texas Coastal Counties"
).add_to(m)

# Add markers with color scale
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
    ).add_to(m)

colormap.add_to(m)
st_data = st_folium(m, width=1300, height=600)

# --- Click interaction ---
if st_data and "last_object_clicked" in st_data:
    lat = st_data["last_object_clicked"].get("lat")
    lon = st_data["last_object_clicked"].get("lng")
    if lat and lon:
        st.markdown("---")
        st.markdown("### 🧪 Selected Station")
        coords = f"{lat},{lon}"
        st.write(f"📍 Coordinates: `{lat:.5f}, {lon:.5f}`")
        ts_df = df_long[df_long["StationKey"] == coords].sort_values("ActivityStartDate")
        subparams = sorted(ts_df["CharacteristicName"].dropna().unique())

        st.markdown("**📌 Select Parameters for Time Series**")
        selected = st.multiselect("📉 Choose parameters", options=subparams, default=subparams[:1])

        if selected:
            plot_df = (
                ts_df[ts_df["CharacteristicName"].isin(selected)]
                .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
                .dropna(how='all')
            )
            st.subheader("📈 Time Series (Dot Plot)")
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in plot_df.columns:
                ax.plot(plot_df.index, plot_df[col], 'o-', label=col)
            ax.set_ylabel("Value")
            ax.set_xlabel("Date")
            ax.legend()
            st.pyplot(fig)

            st.markdown("📊 **Statistical Summary**")
            st.dataframe(plot_df.describe().T.style.format("{:.2f}"))

            st.markdown("🧮 **Correlation Heatmap**")
            corr = plot_df.corr()
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
            st.pyplot(fig2)
        else:
            st.info("Please select at least one parameter.")

