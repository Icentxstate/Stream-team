import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import zipfile
import os
import glob
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from branca.colormap import StepColormap
from streamlit_folium import st_folium

# --- UI CONFIG ---
st.set_page_config(page_title="Cypress Creek Dashboard", page_icon="🗺️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=PT+Serif&display=swap');
    body, .stApp {
        background-color: #ffffff;
        color: #222222;
        font-family: 'PT Serif', 'Georgia', serif;
    }
    .stSidebar > div:first-child {
        background-color: rgba(255, 255, 255, 0.96);
        padding: 1rem;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label {
        color: #000000 !important;
    }
    h1, h2, h3, h4, .stMarkdown, label {
        color: #0c6e72 !important;
        font-weight: bold !important;
        background-color: #ffffff !important;
    }
    .stSelectbox, .stMultiselect, .stTextInput, .stDataFrameContainer {
        background-color: #f8fdfd !important;
        color: #3a3a3a !important;
    }
    .stButton > button {
        background-color: #cc4b00 !important;
        color: white !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #e76f00 !important;
    }
    .dataframe tbody tr {
        background-color: #fef9f3 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if "view" not in st.session_state:
    st.session_state.view = "map"
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

# --- WELCOME CARD ---
if st.session_state.view == "map":
    st.markdown("""
        <div style='background-color:#fef3e2;padding:1.5rem 2rem;border-left:5px solid #cc4b00;margin-bottom:1rem;'>
            <h2 style='color:#cc4b00;'>Welcome to the Cypress Creek Water Dashboard</h2>
            <p>Explore real-time water quality trends. Click on a station to see details.</p>
        </div>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

try:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')
except Exception as e:
    st.error(f"❌ Failed to load CSV: {e}")
    st.stop()

# --- MELT TO LONG FORMAT ---
exclude = ["Name", "Description", "Basin", "County", "Latitude", "Longitude", "TCEQ Stream Segment", "Sample Date"]
value_cols = [col for col in df.columns if col not in exclude]
df_long = df.melt(
    id_vars=["Name", "Latitude", "Longitude", "Sample Date"],
    value_vars=value_cols,
    var_name="CharacteristicName",
    value_name="ResultMeasureValue"
)
df_long["ActivityStartDate"] = pd.to_datetime(df_long["Sample Date"], errors="coerce")
df_long["ResultMeasureValue"] = pd.to_numeric(df_long["ResultMeasureValue"], errors="coerce")
df_long["StationKey"] = df_long["Latitude"].astype(str) + "," + df_long["Longitude"].astype(str)
df_long = df_long.dropna(subset=["ActivityStartDate", "ResultMeasureValue", "CharacteristicName"])

# --- LOAD SHAPEFILE ---
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
# --- SIDEBAR PARAMETER AND DATE SELECTION ---
available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.sidebar.selectbox("📌 Select Parameter", available_params)

# --- DATE FILTER ---
df_param = df_long[df_long["CharacteristicName"] == selected_param].copy()
df_param["YearMonth"] = df_param["ActivityStartDate"].dt.to_period("M").astype(str)
unique_periods = sorted(df_param["YearMonth"].dropna().unique())
selected_period = st.sidebar.selectbox("📅 Select Month-Year", ["All"] + unique_periods)

if selected_period != "All":
    df_param = df_param[df_param["YearMonth"] == selected_period]

filtered_df = df_param.copy()

latest_values = (
    filtered_df.sort_values("ActivityStartDate")
    .groupby("StationKey")
    .tail(1)
    .set_index("StationKey")
)

# --- COLOR MAP SETUP ---
min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = StepColormap(
    colors=['#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'],
    index=np.linspace(min_val, max_val, 6),
    vmin=min_val,
    vmax=max_val,
    caption=f"{selected_param} Value Range"
)

# --- BASEMAP SELECTION ---
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
# --- MAP VIEW ---
if st.session_state.view == "map":
    m = folium.Map(
        location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        tiles=basemap_tiles[basemap_option]["tiles"],
        attr=basemap_tiles[basemap_option]["attr"]
    )
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Add county layer
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

    # Add circle markers for each station
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
            popup=folium.Popup(
                f"{row['Name']}<br>{selected_param}: {val:.2f}<br>{row['ActivityStartDate'].strftime('%Y-%m-%d')}",
                max_width=250
            ),
        ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl().add_to(m)

    st_data = st_folium(m, width=None, height=600)

    # If point clicked → trigger detail view
    if st_data and st_data.get("last_object_clicked"):
        clicked = st_data["last_object_clicked"]
        lat = clicked.get("lat")
        lon = clicked.get("lng")
        if lat is not None and lon is not None:
            st.session_state.selected_point = f"{lat},{lon}"
            st.session_state.view = "details"
            st.rerun()

# --- DETAIL VIEW ---
elif st.session_state.view == "details":
    coords = st.session_state.selected_point
    lat, lon = map(float, coords.split(","))

    st.title("📊 Station Analysis")
    st.write(f"📍 Coordinates: {lat:.5f}, {lon:.5f}")

    # Back button
    with st.form("back_form"):
        submitted = st.form_submit_button("🔙 Back to Map")
        if submitted:
            st.session_state.view = "map"
            st.rerun()

    # Filter data for selected point
    ts_df = df_long[df_long["StationKey"] == coords].sort_values("ActivityStartDate")
    subparams = sorted(ts_df["CharacteristicName"].dropna().unique())
    selected = st.multiselect("📉 Select parameters", subparams, default=subparams[:1])

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Time Series", "📉 Scatter Plot", "📊 Summary Statistics", "🧮 Correlation Heatmap",
        "📦 Boxplot", "📐 Trend Analysis", "💧 WQI", "🗺️ Spatio-Temporal Heatmap",
        "🚨 Anomaly Detection", "📍 Clustering"
    ])

    if not selected:
        for tab in [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10]:
            with tab:
                st.warning("⚠️ Please select at least one parameter to display results.")


# ✅ Tab 1: Time Series
with tab1:
    st.subheader("📈 Time Series")

    # دکمه راهنما (نمایش/پنهان)
    if "show_help_tab1" not in st.session_state:
        st.session_state["show_help_tab1"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab1_button"):
            st.session_state["show_help_tab1"] = not st.session_state["show_help_tab1"]

    # جعبه راهنما اگر فعال باشد
    if st.session_state["show_help_tab1"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
                📝 **Purpose:** Visualize how selected water quality parameters change over time at the selected station.

                📊 **What it shows:**
                - Long-term and short-term variations
                - Seasonal patterns or unexpected spikes
                - Overall trends (upward, downward, or stable)

                🔍 **How to interpret:**
                - Look for consistent increases or decreases that indicate a long-term trend.
                - Identify seasonal behavior (e.g., higher temperatures in summer).
                - Spot sudden spikes or drops, which may signal pollution events or measurement errors.

                📌 **Use cases:**
                - Evaluate the effectiveness of pollution control efforts.
                - Understand environmental impacts over time.
                - Identify critical times for monitoring or interventions.
            """)

    # ایجاد نمودار تایم‌سری
    try:
        plot_df = (
            ts_df[ts_df["CharacteristicName"].isin(selected)]
            .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
            .dropna()
        )

        if plot_df.empty:
            st.info("⚠️ No valid time series data available for the selected parameters.")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in plot_df.columns:
                ax.plot(plot_df.index, plot_df[col], 'o-', label=col)
            ax.set_ylabel("Value")
            ax.set_xlabel("Date")
            ax.set_title("Time Series of Selected Parameters")
            ax.legend()
            st.pyplot(fig)

            # دکمه دانلود تصویر نمودار
            buf_ts = BytesIO()
            fig.savefig(buf_ts, format="png")
            st.download_button("💾 Download Time Series", data=buf_ts.getvalue(), file_name="time_series.png")
    except Exception as e:
        st.error(f"❌ Failed to generate time series plot: {e}")
