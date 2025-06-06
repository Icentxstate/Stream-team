# --- Full Streamlit Dashboard with Modern UI and Parameter Correlation ---
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
from io import BytesIO
from branca.colormap import StepColormap
from streamlit_folium import st_folium

# --- Page Configuration ---
st.set_page_config(page_title="Cypress Creek Dashboard", page_icon="🌊", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class^="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f9f9fb;
        color: #222;
    }

    .stSidebar {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0 !important;
    }

    h1, h2, h3, h4 {
        font-weight: 700 !important;
        color: #1f4e79 !important;
    }

    .stButton > button {
        background-color: #1f4e79 !important;
        color: white !important;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
    }

    .stButton > button:hover {
        background-color: #2c5c8a !important;
    }

    .stSelectbox, .stMultiselect, .stTextInput {
        background-color: #ffffff !important;
        border: 1px solid #ccc !important;
    }

    .block-section {
        padding: 1.5rem;
        margin-bottom: 2rem;
        border-left: 5px solid #1f4e79;
        background-color: #e7f2fa;
        border-radius: 8px;
    }

    .download-button {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Paths ---
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

# --- Load CSV ---
df = pd.read_csv(csv_path, low_memory=False)
df = df.dropna(subset=["Latitude", "Longitude"])
df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')

# --- Transform to Long Format ---
exclude_cols = ["Name", "Description", "Basin", "County", "Latitude", "Longitude", "TCEQ Stream Segment", "Sample Date"]
value_cols = [col for col in df.columns if col not in exclude_cols]
df_long = df.melt(id_vars=["Name", "Latitude", "Longitude", "Sample Date"],
                  value_vars=value_cols,
                  var_name="CharacteristicName",
                  value_name="ResultMeasureValue")
df_long["ActivityStartDate"] = pd.to_datetime(df_long["Sample Date"], errors='coerce')
df_long["ResultMeasureValue"] = pd.to_numeric(df_long["ResultMeasureValue"], errors="coerce")
df_long["StationKey"] = df_long["Latitude"].astype(str) + "," + df_long["Longitude"].astype(str)
df_long.dropna(subset=["ActivityStartDate", "ResultMeasureValue", "CharacteristicName"], inplace=True)

# --- Load Shapefile ---
if not os.path.exists(shp_folder):
    with zipfile.ZipFile(shp_zip, 'r') as zip_ref:
        zip_ref.extractall(shp_folder)
shp_files = glob.glob(os.path.join(shp_folder, "**", "*.shp"), recursive=True)
gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
bounds = gdf.total_bounds

# --- Sidebar ---
st.sidebar.title("Settings")
available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.sidebar.selectbox("Select Parameter", available_params)

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

# --- Main Map ---
st.title("\ud83c\udf0d Texas Coastal Monitoring Map")
m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=8)
m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
folium.GeoJson(gdf, name="Counties", style_function=lambda x: {"color": "#444", "weight": 1}).add_to(m)

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
        popup=f"{row['Name']}<br>{selected_param}: {val:.2f}<br>{row['ActivityStartDate'].strftime('%Y-%m-%d')}"
    ).add_to(m)

colormap.add_to(m)
st_data = st_folium(m, height=600, width=None)

if st_data and st_data.get("last_object_clicked"):
    coords = st_data["last_object_clicked"]
    lat, lon = coords["lat"], coords["lng"]
    station_key = f"{lat},{lon}"
    station_df = df_long[df_long["StationKey"] == station_key]

    st.markdown("<div class='block-section'>", unsafe_allow_html=True)
    st.subheader("\ud83d\udcca Time Series")
    subparams = sorted(station_df["CharacteristicName"].dropna().unique())
    selected = st.multiselect("Select parameters", subparams, default=subparams[:1])

    if selected:
        ts_df = station_df[station_df["CharacteristicName"].isin(selected)]
        plot_df = ts_df.pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue").dropna(how='all')

        fig, ax = plt.subplots(figsize=(10, 5))
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], marker='o', label=col)
        ax.legend()
        ax.set_title("Parameter Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        st.pyplot(fig)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Time Series Plot", data=buf.getvalue(), file_name="time_series.png", mime="image/png", key="ts_dl")

    st.markdown("</div><div class='block-section'>", unsafe_allow_html=True)
    st.subheader("\ud83d\udcca Summary Statistics")
    st.dataframe(plot_df.describe().T.style.format("{:.2f}"))

    st.markdown("</div><div class='block-section'>", unsafe_allow_html=True)
    st.subheader("📊 Correlation Heatmap")
    corr = plot_df.corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
    st.pyplot(fig2)
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    st.download_button("Download Correlation Heatmap", data=buf2.getvalue(), file_name="correlation.png", mime="image/png", key="corr_dl")

    if len(selected) == 2:
        st.markdown("</div><div class='block-section'>", unsafe_allow_html=True)
        st.subheader("🔍 Parameter Relationship")
        p1, p2 = selected[0], selected[1]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.regplot(data=plot_df, x=p1, y=p2, ax=ax3, color="#3b82f6")
        ax3.set_title(f"{p1} vs {p2} (r = {plot_df.corr().loc[p1, p2]:.2f})")
        st.pyplot(fig3)
        buf3 = BytesIO()
        fig3.savefig(buf3, format="png")
        st.download_button("Download Scatter Plot", data=buf3.getvalue(), file_name="scatter.png", mime="image/png", key="scatter_dl")

    st.markdown("</div>", unsafe_allow_html=True)

