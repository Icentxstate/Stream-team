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
from streamlit.components.v1 import html

# ---------- Setup ----------
st.set_page_config(layout="wide")
if "view" not in st.session_state:
    st.session_state.view = "map"
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

# ---------- Load Data ----------
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

if not os.path.exists(shp_folder):
    with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
        zip_ref.extractall(shp_folder)

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

# ---------- Load Shapefile ----------
shp_files = glob.glob(os.path.join(shp_folder, "**", "*.shp"), recursive=True)
gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
gdf_safe = gdf[[col for col in gdf.columns if gdf[col].dtype.kind in 'ifO']].copy()
gdf_safe["geometry"] = gdf["geometry"]
bounds = gdf.total_bounds

# ---------- UI Controls ----------
available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.sidebar.selectbox("📌 Select a Parameter", available_params)
filtered_df = df_long[df_long["CharacteristicName"] == selected_param]
latest_values = (
    filtered_df.sort_values("ActivityStartDate")
    .groupby("StationKey")
    .tail(1)
    .set_index("StationKey")
)

min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = linear.RdYlBu_11.scale(min_val, max_val)

basemap_option = st.sidebar.selectbox(
    "🗺️ Basemap Style",
    ["OpenTopoMap", "Esri World Topo Map", "CartoDB Positron", "Esri Satellite Imagery"]
)
basemap_tiles = {
    "OpenTopoMap": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "OpenTopoMap"
    },
    "Esri World Topo Map": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri Topo"
    },
    "CartoDB Positron": {
        "tiles": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        "attr": "CartoDB"
    },
    "Esri Satellite Imagery": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri Imagery"
    }
}

# ---------- MAP VIEW ----------
if st.session_state.view == "map":
    st.title("🗺️ Texas Coastal Monitoring Map")

    m = folium.Map(
        location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        tiles=basemap_tiles[basemap_option]["tiles"],
        attr=basemap_tiles[basemap_option]["attr"]
    )
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    folium.GeoJson(
        gdf_safe,
        style_function=lambda x: {
            "fillColor": "#0b5394",
            "color": "#0b5394",
            "weight": 2,
            "fillOpacity": 0.3,
        }
    ).add_to(m)

    for key, row in latest_values.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row["ResultMeasureValue"]
        popup_html = f"""
        <b>Station:</b> {row['Name']}<br>
        <b>{selected_param}:</b> {val:.2f}<br>
        <b>Date:</b> {row['ActivityStartDate'].strftime('%Y-%m-%d')}<br><br>
        <button onclick="window.parent.postMessage({{'action': 'zoom', 'lat': {lat}, 'lon': {lon}}}, '*')">
        🔍 Zoom to Station</button><br>
        <button onclick="window.parent.postMessage({{'action': 'analyze', 'lat': {lat}, 'lon': {lon}}}, '*')">
        📊 Data Analysis</button>
        """
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=colormap(val),
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    # HTML JS listener for message passing from popup buttons
    html("""
    <script>
    window.addEventListener("message", (event) => {
        if (event.data.action === "analyze") {
            const coords = `${event.data.lat},${event.data.lon}`;
            localStorage.setItem("station_coords", coords);
            window.location.reload();
        }
        if (event.data.action === "zoom") {
            const mapFrames = parent.document.querySelectorAll('iframe');
            for (const frame of mapFrames) {
                const map = frame.contentWindow.map;
                if (map) {
                    map.setView([event.data.lat, event.data.lon], 15);
                    break;
                }
            }
        }
    });
    </script>
    """, height=0)

    st_data = st_folium(m, width=1400, height=650)

    # Get station selected from localStorage
    import streamlit_js_eval
    coords = streamlit_js_eval.get_local_storage("station_coords", key="coords_key")
    if coords:
        st.session_state.selected_point = coords
        st.session_state.view = "details"

# ---------- DETAILS VIEW ----------
elif st.session_state.view == "details":
    lat, lon = map(float, st.session_state.selected_point.split(","))
    st.title("📊 Station Analysis")
    st.write(f"📍 Coordinates: {lat:.5f}, {lon:.5f}")
    if st.button("🔙 Back to Map"):
        st.session_state.view = "map"
        st.session_state.selected_point = None
        st.experimental_rerun()

    ts_df = df_long[df_long["StationKey"] == f"{lat},{lon}"].sort_values("ActivityStartDate")
    subparams = sorted(ts_df["CharacteristicName"].dropna().unique())
    selected = st.multiselect("📉 Select parameters for time series", subparams, default=subparams[:1])

    if selected:
        plot_df = (
            ts_df[ts_df["CharacteristicName"].isin(selected)]
            .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
            .dropna(how='all')
        )
        st.subheader("📈 Time Series")
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], 'o-', label=col)
        ax.set_ylabel("Value")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

        st.subheader("📊 Summary Statistics")
        st.dataframe(plot_df.describe().T.style.format("{:.2f}"))

        st.subheader("🧮 Correlation Heatmap")
        corr = plot_df.corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("Please select at least one parameter.")
