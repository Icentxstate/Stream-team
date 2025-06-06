# --- Import Libraries ---
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

# --- Page Configuration ---
st.set_page_config(page_title="Cypress Creek Dashboard", page_icon="🌊", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st"]  {
        font-family: 'Inter', sans-serif;
        background-color: #f5f7fa;
        color: #1e1e1e;
    }

    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #d3d3d3;
    }

    .stButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.4rem 1rem;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }

    .block-container > div > h2 {
        padding: 0.5rem;
        background-color: #e0f7f4;
        border-left: 4px solid #10b981;
        color: #065f46;
    }

    .stDataFrame, .stTable {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Welcome Card ---
st.markdown("""
    <div style='background-color:#e0f2fe;padding:1rem 2rem;border-left:5px solid #3b82f6;border-radius:5px;margin-bottom:1rem;'>
        <h2 style='color:#1d4ed8;'>Welcome to the Cypress Creek Water Dashboard</h2>
        <p>Explore real-time water quality trends. Click on any station to view measurements, summaries, and charts. Use the sidebar to filter parameters.</p>
    </div>
""", unsafe_allow_html=True)

# --- Load Data ---
csv_path = "WQ.csv"
shp_zip = "filtered_11_counties.zip"
shp_folder = "shp_extracted"

try:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["ActivityStartDate"] = pd.to_datetime(df["Sample Date"], errors='coerce')
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

# --- Reshape ---
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
gdf["geometry"] = gdf["geometry"].buffer(0)
bounds = gdf.total_bounds

# --- Sidebar ---
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

# --- Map ---
st.title("Map View")
m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=9)
m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

folium.GeoJson(
    gdf,
    style_function=lambda x: {"fillColor": "#338a6d", "color": "#338a6d", "weight": 1, "fillOpacity": 0.2},
    name="Counties"
).add_to(m)

for key, row in latest_values.iterrows():
    val = row["ResultMeasureValue"]
    color = colormap(val)
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(f"{row['Name']}<br>{selected_param}: {val:.2f}<br>{row['ActivityStartDate'].date()}", max_width=250),
    ).add_to(m)

colormap.add_to(m)
st_folium(m, height=600)

# --- Station Summary ---
st.markdown("---")
st.subheader("Station Time Series Analysis")
station_coords = st.selectbox("Select a station (lat,lon)", sorted(df_long["StationKey"].unique()))
ts_df = df_long[df_long["StationKey"] == station_coords].sort_values("ActivityStartDate")
subparams = sorted(ts_df["CharacteristicName"].unique())
selected = st.multiselect("Select parameters to plot", subparams, default=subparams[:1])

if selected:
    plot_df = ts_df[ts_df["CharacteristicName"].isin(selected)].pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
    st.markdown("<div style='background-color:#ecfdf5;padding:0.5rem;'>", unsafe_allow_html=True)
    st.write("### Time Series")
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in plot_df.columns:
        ax.plot(plot_df.index, plot_df[col], marker='o', label=col)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download Time Series Image", data=buf.getvalue(), file_name="timeseries.png", mime="image/png")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='background-color:#fefce8;padding:0.5rem;'>", unsafe_allow_html=True)
    st.write("### Summary Statistics")
    st.dataframe(plot_df.describe().T.style.format("{:.2f}"))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='background-color:#eff6ff;padding:0.5rem;'>", unsafe_allow_html=True)
    st.write("### Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(plot_df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax2)
    st.pyplot(fig2)
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    st.download_button("Download Correlation Heatmap", data=buf2.getvalue(), file_name="correlation.png", mime="image/png")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(selected) == 2:
        st.markdown("<div style='background-color:#fff7ed;padding:0.5rem;'>", unsafe_allow_html=True)
        st.write("### Parameter Interaction (Scatterplot)")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.regplot(x=plot_df[selected[0]], y=plot_df[selected[1]], ax=ax3, color="#3b82f6")
        corr_val = plot_df[selected].corr().iloc[0, 1]
        ax3.set_title(f"{selected[0]} vs. {selected[1]} (r = {corr_val:.2f})")
        ax3.set_xlabel(selected[0])
        ax3.set_ylabel(selected[1])
        st.pyplot(fig3)
        buf3 = BytesIO()
        fig3.savefig(buf3, format="png")
        st.download_button("Download Scatterplot", data=buf3.getvalue(), file_name="scatterplot.png", mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Please select at least one parameter to visualize.")
