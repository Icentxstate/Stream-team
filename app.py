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
from io import BytesIO
import base64

# --- UI config ---
st.set_page_config(page_title="Cypress Creek Dashboard", page_icon="🌊", layout="wide")
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
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar label {
        color: #000000 !important;
    }
    h1, h2, h3, h4, .stMarkdown, .stText, label {
        color: #0c6e72 !important;
        font-weight: bold !important;
    }
    .stButton > button {
        background-color: #cc4b00 !important;
        color: white !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #e76f00 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background-color:#fef3e2;padding:1.5rem 2rem;border-left:5px solid #cc4b00;border-radius:5px;margin-bottom:1rem;'>
    <h2 style='color:#cc4b00;margin-bottom:0.5rem;'>Welcome to the Cypress Creek Water Dashboard</h2>
    <p style='color:#333333;font-size:16px;'>Explore real-time water quality trends across the region. Click on any station to view historical measurements and statistics.</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "view" not in st.session_state:
    st.session_state.view = "map"
if "selected_point" not in st.session_state:
    st.session_state.selected_point = None

# Paths
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

# Load CSV
df = pd.read_csv(csv_path, low_memory=False)
df = df.dropna(subset=["Latitude", "Longitude"])
df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')

# Long format
exclude_cols = ["Name", "Description", "Basin", "County", "Latitude", "Longitude", "TCEQ Stream Segment", "Sample Date"]
value_cols = [col for col in df.columns if col not in exclude_cols]
df_long = df.melt(id_vars=["Name", "Latitude", "Longitude", "Sample Date"], value_vars=value_cols,
                  var_name="CharacteristicName", value_name="ResultMeasureValue")
df_long["ActivityStartDate"] = pd.to_datetime(df_long["Sample Date"], errors='coerce')
df_long["ResultMeasureValue"] = pd.to_numeric(df_long["ResultMeasureValue"], errors="coerce")
df_long["StationKey"] = df_long["Latitude"].astype(str) + "," + df_long["Longitude"].astype(str)
df_long = df_long.dropna(subset=["ActivityStartDate", "ResultMeasureValue", "CharacteristicName"])

# Load Shapefile
if not os.path.exists(shp_folder):
    with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
        zip_ref.extractall(shp_folder)
shp_files = glob.glob(os.path.join(shp_folder, "**", "*.shp"), recursive=True)
gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
gdf = gdf[gdf.geometry.notnull()]
bounds = gdf.total_bounds

# Sidebar
available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.sidebar.selectbox("📌 Select Parameter", available_params)
filtered_df = df_long[df_long["CharacteristicName"] == selected_param]
latest_values = filtered_df.sort_values("ActivityStartDate").groupby("StationKey").tail(1).set_index("StationKey")

min_val = filtered_df["ResultMeasureValue"].min()
max_val = filtered_df["ResultMeasureValue"].max()
colormap = StepColormap(colors=['#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'],
                        index=np.linspace(min_val, max_val, 6), vmin=min_val, vmax=max_val,
                        caption=f"{selected_param} Value Range")

# Map view
if st.session_state.view == "map":
    st.markdown("<h1 style='color:#0c6e72;'>Texas Coastal Monitoring Map</h1>", unsafe_allow_html=True)
    m = folium.Map(location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
                   tiles="OpenStreetMap", zoom_start=7)

    folium.GeoJson(gdf.__geo_interface__,
                   style_function=lambda x: {"fillColor": "#338a6d", "color": "#338a6d", "weight": 1, "fillOpacity": 0.2},
                   name="Counties").add_to(m)

    for key, row in latest_values.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row["ResultMeasureValue"]
        color = colormap(val)
        folium.CircleMarker(location=[lat, lon], radius=6, color=color, fill=True, fill_opacity=0.8,
                            popup=folium.Popup(f"{row['Name']}<br>{selected_param}: {val:.2f}<br>{row['ActivityStartDate'].strftime('%Y-%m-%d')}", max_width=250)).add_to(m)

    colormap.add_to(m)
    st_data = st_folium(m, width=None, height=600)

    if st_data and st_data.get("last_object_clicked"):
        clicked = st_data["last_object_clicked"]
        lat, lon = clicked.get("lat"), clicked.get("lng")
        if lat is not None and lon is not None:
            st.session_state.selected_point = f"{lat},{lon}"
            st.session_state.view = "details"
            st.rerun()

# Details view
elif st.session_state.view == "details":
    coords = st.session_state.selected_point
    lat, lon = map(float, coords.split(","))
    st.markdown(f"<h2 style='background:#eef8f8;padding:0.5rem;'>Station: {lat:.4f}, {lon:.4f}</h2>", unsafe_allow_html=True)

    with st.form("back_form"):
        submitted = st.form_submit_button("🔙 Back to Map")
        if submitted:
            st.session_state.view = "map"
            st.rerun()

    ts_df = df_long[df_long["StationKey"] == coords].sort_values("ActivityStartDate")
    subparams = sorted(ts_df["CharacteristicName"].dropna().unique())
    selected = st.multiselect("📉 Select parameters", subparams, default=subparams[:2])

    if selected:
        plot_df = ts_df[ts_df["CharacteristicName"].isin(selected)].pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue").dropna(how='all')

        # Time Series
        st.markdown("<h3 style='background:#eafaf1;padding:0.3rem;'>📈 Time Series</h3>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], 'o-', label=col)
        ax.set_ylabel("Value")
        ax.set_xlabel("Date")
        ax.legend()
        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥 Download Time Series", data=buf.getvalue(), file_name="time_series.png")

        # Scatter if 2 params
        if len(selected) == 2:
            st.markdown("<h3 style='background:#fff7e6;padding:0.3rem;'>🔄 Scatter Plot</h3>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots()
            ax3.scatter(plot_df[selected[0]], plot_df[selected[1]])
            ax3.set_xlabel(selected[0])
            ax3.set_ylabel(selected[1])
            st.pyplot(fig3)

        # Summary
        st.markdown("<h3 style='background:#e9f2fd;padding:0.3rem;'>📊 Summary Statistics</h3>", unsafe_allow_html=True)
        summary_df = plot_df.describe().T
        st.dataframe(summary_df.style.format("{:.2f}"))
        st.download_button("📥 Download Summary CSV", data=summary_df.to_csv().encode("utf-8"), file_name="summary.csv")

        # Correlation
        st.markdown("<h3 style='background:#fdeff0;padding:0.3rem;'>🧮 Correlation Heatmap</h3>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(plot_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")
        st.download_button("📥 Download Heatmap", data=buf2.getvalue(), file_name="heatmap.png")
    else:
        st.info("Please select at least one parameter.")
