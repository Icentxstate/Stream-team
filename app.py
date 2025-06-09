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

# --- UI config ---
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
        border-radius: 0;
    }

    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar label {
        color: #000000 !important;
    }

    h1, h2, h3, h4, .stMarkdown, .stText, label, .css-10trblm, .css-1v3fvcr {
        color: #0c6e72 !important;
        font-weight: bold !important;
        background-color: #ffffff !important;
        padding: 0.5rem;
        border-radius: 0;
    }

    .stSelectbox, .stMultiselect, .stTextInput, .stDateInput, .stDataFrameContainer, .stForm {
        background-color: #f8fdfd !important;
        color: #3a3a3a !important;
        border-radius: 4px;
        border: 1px solid #cfd7d7;
        font-family: 'PT Serif', 'Georgia', serif;
    }

    .stButton > button {
        background-color: #cc4b00 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.4rem 1rem;
        transition: 0.2s;
        font-family: 'PT Serif', 'Georgia', serif;
    }

    .stButton > button:hover {
        background-color: #e76f00 !important;
    }

    .dataframe tbody tr {
        background-color: #fef9f3 !important;
        color: #000000;
    }

    .block-container > div > h2 {
        padding: 0.6rem 1rem;
        background-color: #eef8f8;
        border-left: 5px solid #0c6e72;
        border-radius: 0;
        margin-bottom: 1rem;
        color: #cc4b00 !important;
    }

    .stDataFrame, .stTable {
        background-color: #ffffff !important;
        color: #222222 !important;
        font-family: 'PT Serif', 'Georgia', serif;
    }

    iframe {
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# 📌 Orange Welcome Card (only on map view)
if "view" in st.session_state and st.session_state.view == "map":
    st.markdown("""
        <div style='background-color:#fef3e2;padding:1.5rem 2rem;border-left:5px solid #cc4b00;border-radius:5px;margin-bottom:1rem;'>
            <h2 style='color:#cc4b00;margin-bottom:0.5rem;'>Welcome to the Cypress Creek Water Dashboard</h2>
            <p style='color:#333333;font-size:16px;'>Explore real-time water quality trends across the region. Click on any station to view historical measurements and statistics. Customize the view using the sidebar on the left.</p>
        </div>
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
# --- Parameter selection ---
selected_param = st.sidebar.selectbox("📌 Select Parameter", available_params)

# --- Date filter by month and year ---
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
#########------------------------------ADD-----------------------------
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📈 Time Series", "📉 Scatter Plot", "📊 Summary Statistics", 
    "🧮 Correlation Heatmap", "📦 Boxplot", "📐 Trend Analysis", 
    "💧 WQI", "🗺️ Spatio-Temporal Heatmap", "🚨 Anomaly Detection", 
    "📍 Clustering"
])
with tab1:
    st.subheader("📈 Time Series")

    if "show_help_tab1" not in st.session_state:
        st.session_state["show_help_tab1"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab1"):
            st.session_state["show_help_tab1"] = not st.session_state["show_help_tab1"]

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

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in plot_df.columns:
        ax.plot(plot_df.index, plot_df[col], 'o-', label=col)
    ax.set_ylabel("Value")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

    buf_ts = BytesIO()
    fig.savefig(buf_ts, format="png")
    st.download_button("💾 Download Time Series", data=buf_ts.getvalue(), file_name="time_series.png")
with tab2:
    st.subheader("📉 Scatter Plot")

    if "show_help_tab2" not in st.session_state:
        st.session_state["show_help_tab2"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab2"):
            st.session_state["show_help_tab2"] = not st.session_state["show_help_tab2"]

    if st.session_state["show_help_tab2"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Explore relationships between two selected parameters.

            📊 **What it shows:**
            - Scatter plot with X and Y variables
            - Each point represents one sample

            🔍 **How to interpret:**
            - Positive slope: both variables increase together
            - Negative slope: one increases while the other decreases
            - Clustered points suggest strong relationship
            - Wide spread = weak or no relationship

            📌 **Use cases:**
            - Check if temperature affects dissolved oxygen
            - Detect parameter dependencies (e.g., TDS vs. conductivity)
            - Validate assumptions before modeling
            """)

    if not ts_df.empty:
        all_params = sorted(ts_df["CharacteristicName"].dropna().unique())
        x_var = st.selectbox("X-axis Variable", all_params, key="scatter_x")
        y_var = st.selectbox("Y-axis Variable", [p for p in all_params if p != x_var], key="scatter_y")

        df_x = ts_df[ts_df["CharacteristicName"] == x_var].rename(columns={"ResultMeasureValue": "X"})
        df_y = ts_df[ts_df["CharacteristicName"] == y_var].rename(columns={"ResultMeasureValue": "Y"})

        merged_df = pd.merge(df_x, df_y, on="ActivityStartDate", suffixes=("_x", "_y"))
        if not merged_df.empty:
            fig, ax = plt.subplots()
            ax.scatter(merged_df["X"], merged_df["Y"], alpha=0.7)
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f"{x_var} vs. {y_var}")
            st.pyplot(fig)

            buf_scatter = BytesIO()
            fig.savefig(buf_scatter, format="png")
            st.download_button("💾 Download Scatter Plot", data=buf_scatter.getvalue(), file_name="scatter_plot.png")
        else:
            st.warning("No overlapping data points between selected variables.")
    else:
        st.info("Time series data is empty.")
with tab3:
    st.subheader("📊 Summary Statistics")

    if "show_help_tab3" not in st.session_state:
        st.session_state["show_help_tab3"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab3"):
            st.session_state["show_help_tab3"] = not st.session_state["show_help_tab3"]

    if st.session_state["show_help_tab3"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Provide basic statistical summaries of selected water quality parameters.

            📊 **What it shows:**
            - Mean, standard deviation, min, max, and quartiles
            - Parameter-by-parameter overview

            🔍 **How to interpret:**
            - Use **mean** to understand average conditions
            - Use **std** (standard deviation) to see variability
            - **Min/Max** help identify extreme events
            - **25%, 50%, 75%** percentiles describe distribution shape

            📌 **Use cases:**
            - Compare variability across stations
            - Spot outliers or consistently high/low readings
            - Feed into water quality index or anomaly detection
            """)

    if not plot_df.empty:
        stats_df = plot_df.describe().T
        st.dataframe(stats_df.style.format("{:.2f}"))

        csv_stats = stats_df.to_csv().encode("utf-8")
        st.download_button("💾 Download Summary CSV", data=csv_stats, file_name="summary_statistics.csv")
    else:
        st.info("Please select at least one parameter to see summary statistics.")
with tab4:
    st.subheader("🧮 Correlation Heatmap")

    if "show_help_tab4" not in st.session_state:
        st.session_state["show_help_tab4"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab4"):
            st.session_state["show_help_tab4"] = not st.session_state["show_help_tab4"]

    if st.session_state["show_help_tab4"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Show how water quality parameters are correlated with each other.

            📊 **What it shows:**
            - Pearson correlation matrix (values from -1 to 1)
            - Color-coded heatmap

            🔍 **How to interpret:**
            - **+1**: perfect positive correlation (both increase together)
            - **-1**: perfect negative correlation (one increases while other decreases)
            - **0**: no linear correlation
            - Darker or lighter colors indicate strength and direction

            📌 **Use cases:**
            - Detect relationships (e.g., conductivity vs. TDS)
            - Reduce dimensionality for modeling
            - Support clustering or anomaly detection
            """)

    if not plot_df.empty:
        corr = plot_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

        buf_corr = BytesIO()
        fig.savefig(buf_corr, format="png")
        st.download_button("💾 Download Heatmap", data=buf_corr.getvalue(), file_name="correlation_heatmap.png")
    else:
        st.info("Please select at least one parameter.")
        
with tab5:
    st.subheader("📦 Boxplot")

    if "show_help_tab5" not in st.session_state:
        st.session_state["show_help_tab5"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab5"):
            st.session_state["show_help_tab5"] = not st.session_state["show_help_tab5"]

    if st.session_state["show_help_tab5"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Display the distribution, spread, and outliers for selected parameters using boxplots.

            📊 **What it shows:**
            - Median, quartiles (25% and 75%)
            - Whiskers showing range
            - Outliers as dots outside the box

            🔍 **How to interpret:**
            - The **box** shows the middle 50% of data
            - The **line inside** = median (middle value)
            - **Dots** far outside the box are outliers
            - Taller boxes = more variability

            📌 **Use cases:**
            - Quickly compare spread between parameters
            - Detect outliers or extreme pollution
            - Summarize water quality conditions
            """)

    if not ts_df.empty:
        selected_box_params = st.multiselect("📌 Select parameters for boxplot", sorted(ts_df["CharacteristicName"].dropna().unique()), default=selected)

        if selected_box_params:
            filtered = ts_df[ts_df["CharacteristicName"].isin(selected_box_params)]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=filtered, x="CharacteristicName", y="ResultMeasureValue", ax=ax)
            ax.set_title("Boxplot of Selected Parameters")
            ax.set_xlabel("Parameter")
            ax.set_ylabel("Value")
            st.pyplot(fig)

            buf5 = BytesIO()
            fig.savefig(buf5, format="png")
            st.download_button("💾 Download Boxplot", data=buf5.getvalue(), file_name="boxplot.png")
        else:
            st.info("Please select at least one parameter.")
    else:
        st.warning("No time series data available.")
with tab6:
    st.subheader("📐 Trend Analysis (Mann-Kendall Test)")

    if "show_help_tab6" not in st.session_state:
        st.session_state["show_help_tab6"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab6"):
            st.session_state["show_help_tab6"] = not st.session_state["show_help_tab6"]

    if st.session_state["show_help_tab6"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Assess whether there is a statistically significant increasing or decreasing trend over time in water quality data.

            📊 **What it shows:**
            - Mann-Kendall test results (Z, p-value, trend direction)
            - Plot of time series with trend line (Sen’s slope)

            🔍 **How to interpret:**
            - **Z > 0 and p < 0.05**: significant increasing trend
            - **Z < 0 and p < 0.05**: significant decreasing trend
            - **p > 0.05**: no significant trend
            - Sen’s slope shows the rate of change

            📌 **Use cases:**
            - Evaluate long-term effectiveness of interventions
            - Detect emerging pollution issues
            - Support environmental reporting and policy
            """)

    import pymannkendall as mk

    if not ts_df.empty:
        for param in selected:
            st.markdown(f"### 📈 {param}")

            series = (
                ts_df[ts_df["CharacteristicName"] == param]
                .sort_values("ActivityStartDate")
                .set_index("ActivityStartDate")["ResultMeasureValue"]
                .dropna()
            )

            if len(series) < 10:
                st.warning(f"Not enough data for {param} (needs ≥ 10 records).")
                continue

            result = mk.original_test(series)

            st.markdown(f"**Trend:** `{result.trend}`  |  **Z:** `{result.Z:.2f}`  |  **p-value:** `{result.p:.3f}`  |  **Sen's Slope:** `{result.slope:.3f}`")

            fig, ax = plt.subplots()
            ax.plot(series.index, series.values, label=param, color="gray")
            ax.plot(series.index, result.intercept + result.slope * range(len(series)), label="Trend Line", color="red")
            ax.set_title(f"Trend for {param}")
            ax.set_ylabel(param)
            ax.legend()
            st.pyplot(fig)

            buf6 = BytesIO()
            fig.savefig(buf6, format="png")
            st.download_button(f"💾 Download {param} Trend Plot", data=buf6.getvalue(), file_name=f"{param}_trend.png")
    else:
        st.info("No time series data available.")
with tab7:
    st.subheader("💧 Water Quality Index (WQI)")

    if "show_help_tab7" not in st.session_state:
        st.session_state["show_help_tab7"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab7"):
            st.session_state["show_help_tab7"] = not st.session_state["show_help_tab7"]

    if st.session_state["show_help_tab7"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Calculate a single score summarizing water quality based on selected parameters using a weighted average.

            📊 **What it shows:**
            - WQI over time (e.g., monthly)
            - Aggregated quality rating (0–100)

            🔍 **How to interpret:**
            - 90–100: Excellent
            - 70–90: Good
            - 50–70: Medium
            - 25–50: Poor
            - <25: Very Poor

            📌 **Use cases:**
            - Quickly assess water quality status
            - Compare sites or time periods
            - Support public reporting or decision-making
            """)

    # --- Example WQI Weights (adjustable if needed) ---
    weights = {
        "pH": 0.1,
        "Dissolved Oxygen": 0.2,
        "Nitrate": 0.15,
        "Phosphate": 0.15,
        "Temperature": 0.1,
        "Turbidity": 0.1,
        "Conductivity": 0.2,
    }

    wqi_data = ts_df[ts_df["CharacteristicName"].isin(weights.keys())].copy()

    if wqi_data.empty:
        st.warning("No WQI-eligible parameters available in dataset.")
    else:
        wqi_data["Weight"] = wqi_data["CharacteristicName"].map(weights)

        # Normalize values (min-max scaling per parameter)
        norm_vals = []
        for param, group in wqi_data.groupby("CharacteristicName"):
            norm = (group["ResultMeasureValue"] - group["ResultMeasureValue"].min()) / (
                group["ResultMeasureValue"].max() - group["ResultMeasureValue"].min()
            )
            norm_vals.append(norm)

        wqi_data["Normalized"] = pd.concat(norm_vals).sort_index()
        wqi_data["Weighted"] = wqi_data["Normalized"] * wqi_data["Weight"]

        wqi_scores = (
            wqi_data.groupby("ActivityStartDate")["Weighted"].sum().clip(0, 1) * 100
        ).rename("WQI").reset_index()

        # Plot
        fig, ax = plt.subplots()
        ax.plot(wqi_scores["ActivityStartDate"], wqi_scores["WQI"], marker="o", color="teal")
        ax.set_ylabel("WQI Score")
        ax.set_xlabel("Date")
        ax.set_title("Water Quality Index Over Time")
        ax.axhspan(90, 100, color="green", alpha=0.2, label="Excellent")
        ax.axhspan(70, 90, color="lightgreen", alpha=0.2, label="Good")
        ax.axhspan(50, 70, color="orange", alpha=0.2, label="Medium")
        ax.axhspan(25, 50, color="red", alpha=0.2, label="Poor")
        ax.axhspan(0, 25, color="darkred", alpha=0.2, label="Very Poor")
        ax.legend()
        st.pyplot(fig)

        buf7 = BytesIO()
        fig.savefig(buf7, format="png")
        st.download_button("💾 Download WQI Plot", data=buf7.getvalue(), file_name="wqi_plot.png")

        st.download_button("💾 Download WQI Plot", da_
with tab8:
    st.subheader("🗺️ Spatio-Temporal Heatmap")

    if "show_help_tab8" not in st.session_state:
        st.session_state["show_help_tab8"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab8"):
            st.session_state["show_help_tab8"] = not st.session_state["show_help_tab8"]

    if st.session_state["show_help_tab8"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Visualize how a parameter’s values change across both time and space using a heatmap.

            📊 **What it shows:**
            - Site vs. Time matrix
            - Color intensity based on parameter value

            🔍 **How to interpret:**
            - Darker colors = higher values
            - Horizontal patterns = temporal trends
            - Vertical patterns = site-specific behavior
            - Diagonal streaks may indicate moving plumes or coordinated change

            📌 **Use cases:**
            - Detect simultaneous changes across sites
            - Spot seasonal effects in multiple locations
            - Identify hotspots over time
            """)

    if not ts_df.empty:
        heat_param = st.selectbox("📌 Select parameter for heatmap", sorted(ts_df["CharacteristicName"].dropna().unique()), key="heat_param")

        df_hm = ts_df[ts_df["CharacteristicName"] == heat_param].copy()
        if df_hm.empty:
            st.warning("No data available for this parameter.")
        else:
            df_hm["Date"] = pd.to_datetime(df_hm["ActivityStartDate"])
            df_hm["Month"] = df_hm["Date"].dt.to_period("M").astype(str)
            df_hm["Site"] = df_hm["MonitoringLocationIdentifier"]

            heat_data = df_hm.pivot_table(index="Site", columns="Month", values="ResultMeasureValue", aggfunc="mean")

            fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(heat_data))))
            sns.heatmap(heat_data, cmap="YlOrRd", linewidths=0.1, linecolor="gray", annot=False)
            ax.set_title(f"{heat_param} — Spatio-Temporal Heatmap")
            st.pyplot(fig)

            buf8 = BytesIO()
            fig.savefig(buf8, format="png")
            st.download_button("💾 Download Heatmap", data=buf8.getvalue(), file_name=f"{heat_param}_heatmap.png")
    else:
        st.info("No time series data available.")
with tab9:
    st.subheader("🚨 Anomaly Detection")

    if "show_help_tab9" not in st.session_state:
        st.session_state["show_help_tab9"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab9"):
            st.session_state["show_help_tab9"] = not st.session_state["show_help_tab9"]

    if st.session_state["show_help_tab9"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Automatically identify unusual values in water quality data across multiple stations.

            📊 **What it shows:**
            - Anomalous data points flagged by statistical methods (e.g., Isolation Forest, IQR)
            - Visualization of normal vs. anomalous values

            🔍 **How to interpret:**
            - Red points = detected anomalies (outliers)
            - Blue/green = normal expected range
            - Review data range and sampling date for possible error, pollution, or natural event

            ⚠️ **Note:** This analysis requires data from multiple stations. Please ensure you’ve selected multiple stations to detect cross-site anomalies.

            📌 **Use cases:**
            - Detect possible measurement errors or pollution events
            - Flag outlier samples for manual review
            - Complement QA/QC and alert systems
            """)

    from sklearn.ensemble import IsolationForest

    if not ts_df.empty:
        selected_param = st.selectbox("📌 Select parameter for anomaly detection", sorted(ts_df["CharacteristicName"].dropna().unique()), key="anom_param")
        anom_df = ts_df[ts_df["CharacteristicName"] == selected_param].copy()

        if anom_df["MonitoringLocationIdentifier"].nunique() < 2:
            st.warning("Please select multiple stations to enable cross-site anomaly detection.")
        elif len(anom_df) < 20:
            st.warning("Not enough records for anomaly detection.")
        else:
            anom_df = anom_df.dropna(subset=["ResultMeasureValue"])
            anom_df["Date"] = pd.to_datetime(anom_df["ActivityStartDate"])
            anom_df = anom_df.sort_values("Date")

            # Prepare data
            X = anom_df[["ResultMeasureValue"]]
            clf = IsolationForest(contamination=0.1, random_state=42)
            anom_df["Anomaly"] = clf.fit_predict(X)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(anom_df["Date"], anom_df["ResultMeasureValue"], "bo-", label="Normal")
            ax.plot(anom_df[anom_df["Anomaly"] == -1]["Date"],
                    anom_df[anom_df["Anomaly"] == -1]["ResultMeasureValue"],
                    "ro", label="Anomaly")
            ax.set_title(f"Anomaly Detection — {selected_param}")
            ax.set_ylabel(selected_param)
            ax.set_xlabel("Date")
            ax.legend()
            st.pyplot(fig)

            buf9 = BytesIO()
            fig.savefig(buf9, format="png")
            st.download_button("💾 Download Anomaly Plot", data=buf9.getvalue(), file_name=f"{selected_param}_anomalies.png")
    else:
        st.info("Time series data is empty.")
with tab10:
    st.subheader("📍 KMeans Clustering of Selected Stations")

    if "show_help_tab10" not in st.session_state:
        st.session_state["show_help_tab10"] = False

    col1, col2 = st.columns([1, 9])
    with col1:
        if st.button("❔", key="toggle_help_tab10"):
            st.session_state["show_help_tab10"] = not st.session_state["show_help_tab10"]

    if st.session_state["show_help_tab10"]:
        with st.expander("📘 Tab Help", expanded=True):
            st.markdown("""
            📝 **Purpose:** Group monitoring stations into clusters based on water quality characteristics using KMeans.

            📊 **What it shows:**
            - A scatter plot of stations in reduced 2D space (via PCA)
            - Color-coded clusters

            🔍 **How to interpret:**
            - Points close together = similar water quality profiles
            - Different colors = different clusters
            - Use legend or hover info to identify stations

            ⚠️ **Note:** Please select multiple stations to enable clustering.

            📌 **Use cases:**
            - Group sites for similar treatment strategies
            - Identify unique or extreme stations
            - Support regional analysis or reporting
            """)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    selected_stations = st.multiselect("📍 Select stations for clustering", ts_df["MonitoringLocationIdentifier"].unique())
    n_clusters = st.slider("📌 Select number of clusters", min_value=2, max_value=10, value=3)

    if selected_stations:
        cluster_df = ts_df[ts_df["MonitoringLocationIdentifier"].isin(selected_stations)].copy()
        pivot = cluster_df.pivot_table(index="MonitoringLocationIdentifier",
                                       columns="CharacteristicName",
                                       values="ResultMeasureValue",
                                       aggfunc="mean").fillna(0)

        if pivot.shape[0] >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pivot)
            pivot["Cluster"] = kmeans.labels_

            pca = PCA(n_components=2)
            components = pca.fit_transform(pivot.drop(columns=["Cluster"]))
            pivot["PC1"] = components[:, 0]
            pivot["PC2"] = components[:, 1]

            fig, ax = plt.subplots()
            for cluster in sorted(pivot["Cluster"].unique()):
                group = pivot[pivot["Cluster"] == cluster]
                ax.scatter(group["PC1"], group["PC2"], label=f"Cluster {cluster}", s=100)
                for idx, row in group.iterrows():
                    ax.text(row["PC1"], row["PC2"], idx, fontsize=8)

            ax.set_title("Station Clustering (KMeans)")
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend()
            st.pyplot(fig)

            buf10 = BytesIO()
            fig.savefig(buf10, format="png")
            st.download_button("💾 Download Cluster Plot", data=buf10.getvalue(), file_name="clustering_plot.png")

            st.markdown("### 📋 Clustered Station Summary")
            st.dataframe(pivot[["Cluster", "PC1", "PC2"]])
        else:
            st.warning("Number of stations must be greater than or equal to number of clusters.")
    else:
        st.info("Please select at least two stations for clustering.")
