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
from branca.colormap import StepColormap
from streamlit_folium import st_folium

# --- UI config ---
st.set_page_config(page_title="Cypress Creek Dashboard", page_icon="🌊", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, .stApp {
        background-color: #f9f9f9;
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }

    .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }

    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar label {
        color: #111827 !important;
        font-weight: 600;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
        font-weight: 700;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e5e7eb;
    }

    .stSelectbox, .stMultiselect, .stTextInput, .stDateInput, .stDataFrameContainer, .stForm {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border-radius: 6px;
        border: 1px solid #d1d5db;
        padding: 0.3rem;
    }

    .stButton > button {
        background-color: #2563eb !important;
        color: white !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        transition: background-color 0.2s ease-in-out;
        border: none;
    }

    .stButton > button:hover {
        background-color: #1e40af !important;
    }

    .dataframe tbody tr {
        background-color: #fdfdfd !important;
        color: #111827;
    }

    .block-container > div > h2 {
        background-color: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 0.7rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #111827 !important;
    }

    .stDataFrame, .stTable {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    iframe {
        border: none;
    }

    .stTooltip {
        background-color: #1f2937 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)
# --- Session state ---
if "view" not in st.session_state:
    st.session_state.view = "map"
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

# --- Paths ---
csv_path = "WQ.csv"
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

# --- Long Format ---
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
if not shp_files:
    st.error("❌ No shapefile found.")
    st.stop()
gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
gdf_safe = gdf[[col for col in gdf.columns if gdf[col].dtype.kind in 'ifO']].copy()
gdf_safe["geometry"] = gdf["geometry"]
bounds = gdf.total_bounds

# --- Sidebar ---
available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.sidebar.selectbox("📌 Select Parameter", available_params)
filtered_df = df_long[df_long["CharacteristicName"] == selected_param]
latest_values = (
    filtered_df.sort_values("ActivityStartDate")
    .groupby("StationKey")
    .tail(1)
    .set_index("StationKey")
)

min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = StepColormap(
    colors=['#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'],
    index=np.linspace(min_val, max_val, 6),
    vmin=min_val,
    vmax=max_val,
    caption=f"{selected_param} Value Range"
)

basemap_option = st.sidebar.selectbox("🗺️ Basemap Style", [
    "OpenTopoMap", "Esri World Topo Map", "CartoDB Positron", "Esri Satellite Imagery"
])
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

# --- Main View ---
if st.session_state.view == "map":
    st.title("🌍 Texas Coastal Monitoring Map")
    m = folium.Map(
        location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        tiles=basemap_tiles[basemap_option]["tiles"],
        attr=basemap_tiles[basemap_option]["attr"]
    )
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    folium.GeoJson(
        gdf_safe,
        style_function=lambda x: {
            "fillColor": "#338a6d",
            "color": "#338a6d",
            "weight": 2,
            "fillOpacity": 0.3,
        },
        name="Counties"
    ).add_to(m)

    for key, row in latest_values.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row["ResultMeasureValue"]
        color = colormap(val)
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(f"{row['Name']}<br>{selected_param}: {val:.2f}<br>{row['ActivityStartDate'].strftime('%Y-%m-%d')}", max_width=250),
        ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    st_data = st_folium(m, width=None, height=600)

    if st_data and st_data.get("last_object_clicked"):
        clicked = st_data["last_object_clicked"]
        lat = clicked.get("lat")
        lon = clicked.get("lng")
        if lat is not None and lon is not None:
            st.session_state.selected_point = f"{lat},{lon}"
            st.session_state.view = "details"
            st.rerun()

elif st.session_state.view == "details":
    coords = st.session_state.selected_point
    lat, lon = map(float, coords.split(","))
    st.title("📊 Station Analysis")
    st.write(f"📍 Coordinates: {lat:.5f}, {lon:.5f}")

    with st.form("back_form"):
        submitted = st.form_submit_button("🔙 Back to Map")
        if submitted:
            st.session_state.view = "map"
            st.rerun()

    ts_df = df_long[df_long["StationKey"] == coords].sort_values("ActivityStartDate")
    subparams = sorted(ts_df["CharacteristicName"].dropna().unique())
    selected = st.multiselect("📉 Select parameters", subparams, default=subparams[:1])

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

        # Scatter Plot: Parameter vs Parameter
if len(selected) == 2:
    st.subheader(f"📌 Scatter Plot: {selected[0]} vs {selected[1]}")

    scatter_df = plot_df.dropna(subset=selected)

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(scatter_df[selected[0]], scatter_df[selected[1]], alpha=0.7, color="#2563eb")
    ax3.set_xlabel(selected[0])
    ax3.set_ylabel(selected[1])
    ax3.set_title("Parameter Correlation")
    ax3.grid(True)

    st.pyplot(fig3)
elif len(selected) > 2:
    st.info("⚠️ Scatter plot is only available when exactly two parameters are selected.")
       
        st.subheader("📊 Summary Statistics")
        st.dataframe(plot_df.describe().T.style.format("{:.2f}"))

        st.subheader("🧮 Correlation Heatmap")
        corr = plot_df.corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
        st.pyplot(fig2)
    else:
        st.info("Please select at least one parameter.")
