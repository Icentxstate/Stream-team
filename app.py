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
            "📈 Time Series", "📉 Scatter Plot", "📊 Summary Statistics", "🧮 Correlation Heatmap",
            "📦 Boxplot", "📐 Trend Analysis", "💧 WQI", "🗺️ Spatio-Temporal Heatmap",
            "🚨 Anomaly Detection", "📍 Clustering"
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

        # Tab 2: Scatter Plot
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
                    📝 **Purpose:** Explore the relationship between two selected parameters.

                    📊 **What it shows:**
                    - Correlations (positive, negative, or none)
                    - Outliers and unusual behaviors

                    🔍 **How to interpret:**
                    - An upward trend of points suggests a positive correlation (as one increases, so does the other).
                    - A downward trend indicates a negative correlation.
                    - Scattered or random points mean no strong relationship.
                    - Distant points may represent anomalies or extreme pollution events.

                    📌 **Use cases:**
                    - Detect potential cause-effect relationships.
                    - Refine models or select key parameters.
                    - Support hypotheses in environmental research.
                    """)

            all_params = sorted(ts_df["CharacteristicName"].dropna().unique())
            x_var = st.selectbox("X-axis Variable", all_params, key="scatter_x")
            y_var = st.selectbox("Y-axis Variable", [p for p in all_params if p != x_var], key="scatter_y")

            scatter_df = (
                ts_df[ts_df["CharacteristicName"].isin([x_var, y_var])]
                .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
                .dropna()
            )

            if not scatter_df.empty:
                fig3, ax3 = plt.subplots()
                ax3.scatter(scatter_df[x_var], scatter_df[y_var], c='steelblue', alpha=0.7)
                ax3.set_xlabel(x_var)
                ax3.set_ylabel(y_var)
                ax3.set_title(f"{y_var} vs {x_var}")
                st.pyplot(fig3)

                buf_scatter = BytesIO()
                fig3.savefig(buf_scatter, format="png")
                st.download_button("💾 Download Scatter Plot", data=buf_scatter.getvalue(), file_name="scatter_plot.png")
            else:
                st.info("Not enough data to generate scatter plot.")
        # Tab 3: Summary Statistics
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
                    📝 **Purpose:** Summarize key statistics for selected parameters, including average, minimum, maximum, standard deviation, and sample count.

                    📊 **What it shows:**
                    - General behavior of each parameter
                    - Data consistency or variability

                    🔍 **How to interpret:**
                    - Use the **mean** to understand the central tendency.
                    - Compare **mean** vs. **median** to check for skewness or outliers.
                    - A high **standard deviation** suggests high variability.
                    - **Count** tells you how much data is available — fewer than 8 samples may be too little for trend analysis.

                    📌 **Use cases:**
                    - Establish baseline values for water quality.
                    - Prepare inputs for modeling.
                    - Compare variability across time or stations.
                    """)

            stats = plot_df.describe().T
            st.dataframe(stats.style.format("{:.2f}"))
            csv_stats = stats.to_csv().encode("utf-8")
            st.download_button("💾 Download Summary CSV", data=csv_stats, file_name="summary_statistics.csv")
        # Tab 4: Correlation Heatmap
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
                    📝 **Purpose:** Show how strongly each parameter correlates with others using a color-coded matrix.

                    📊 **What it shows:**
                    - Values near **+1** = strong positive correlation
                    - Values near **-1** = strong negative correlation
                    - Values near **0** = no correlation

                    🔍 **How to interpret:**
                    - Darker or more intense colors indicate stronger relationships.
                    - Positive correlations suggest that two parameters increase or decrease together.
                    - Negative correlations mean one increases while the other decreases.
                    - Values close to zero imply no consistent relationship.

                    📌 **Use cases:**
                    - Select influential parameters for models or indexes.
                    - Reduce dimensionality in complex datasets.
                    - Reveal potential causes of water quality issues.
                    """)

            corr = plot_df.corr()
            if not corr.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
                st.pyplot(fig2)

                buf_corr = BytesIO()
                fig2.savefig(buf_corr, format="png")
                st.download_button(
                    "💾 Download Correlation Heatmap",
                    data=buf_corr.getvalue(),
                    file_name="correlation_heatmap.png"
                )
            else:
                st.info("Not enough data for correlation heatmap.")
        # Tab 5: Temporal Boxplots
        with tab5:
            st.subheader("📦 Temporal Boxplots")

            if "show_help_tab5" not in st.session_state:
                st.session_state["show_help_tab5"] = False

            col1, col2 = st.columns([1, 9])
            with col1:
                if st.button("❔", key="toggle_help_tab5"):
                    st.session_state["show_help_tab5"] = not st.session_state["show_help_tab5"]

            if st.session_state["show_help_tab5"]:
                with st.expander("📘 Tab Help", expanded=True):
                    st.markdown("""
                    📝 **Purpose:** Show parameter distributions grouped by time (month, season, or year) using box-and-whisker plots.

                    📊 **What it shows:**
                    - Median, quartiles, and data spread
                    - Seasonal variations or long-term shifts
                    - Presence of outliers

                    🔍 **How to interpret:**
                    - A longer box indicates greater variability in the data.
                    - Shifts in medians across time may indicate seasonal influence or gradual trends.
                    - Outliers (points beyond the whiskers) could suggest pollution events or measurement anomalies.
                    - Comparing different parameters on the same plot helps identify differences in behavior over time.

                    📌 **Use cases:**
                    - Identify critical time periods for management or intervention.
                    - Understand seasonal effects on water quality parameters.
                    - Support decisions about when to schedule sampling campaigns.
                    """)

            def get_season(month):
                if month in [12, 1, 2]:
                    return "Winter"
                elif month in [3, 4, 5]:
                    return "Spring"
                elif month in [6, 7, 8]:
                    return "Summer"
                else:
                    return "Fall"

            seasonal_df = ts_df[ts_df["CharacteristicName"].isin(selected)].copy()
            seasonal_df["Month"] = seasonal_df["ActivityStartDate"].dt.strftime("%b")
            seasonal_df["Year"] = seasonal_df["ActivityStartDate"].dt.year
            seasonal_df["Season"] = seasonal_df["ActivityStartDate"].dt.month.apply(get_season)

            box_type = st.radio("Select Time Grouping:", ["Season", "Month", "Year"], horizontal=True)

            if not seasonal_df.empty:
                fig5, ax5 = plt.subplots(figsize=(12, 5))

                if box_type == "Season":
                    sns.boxplot(
                        x="Season", y="ResultMeasureValue", hue="CharacteristicName",
                        data=seasonal_df, palette="Set2", ax=ax5
                    )
                elif box_type == "Month":
                    sns.boxplot(
                        x="Month", y="ResultMeasureValue", hue="CharacteristicName",
                        data=seasonal_df, palette="Set3",
                        order=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                        ax=ax5
                    )
                else:
                    sns.boxplot(
                        x="Year", y="ResultMeasureValue", hue="CharacteristicName",
                        data=seasonal_df, palette="Set1", ax=ax5
                    )

                ax5.set_ylabel("Value")
                st.pyplot(fig5)

                buf5 = BytesIO()
                fig5.savefig(buf5, format="png")
                st.download_button("💾 Download Boxplot Image", data=buf5.getvalue(), file_name=f"boxplot_{box_type.lower()}.png")
            else:
                st.info("Not enough data to generate temporal boxplots.")
        # Tab 6: Mann-Kendall Trend Test
        with tab6:
            st.subheader("📐 Mann-Kendall Trend Test")

            if "show_help_tab6" not in st.session_state:
                st.session_state["show_help_tab6"] = False

            col1, col2 = st.columns([1, 9])
            with col1:
                if st.button("❔", key="toggle_help_tab6"):
                    st.session_state["show_help_tab6"] = not st.session_state["show_help_tab6"]

            if st.session_state["show_help_tab6"]:
                with st.expander("📘 Tab Help", expanded=True):
                    st.markdown("""
                    📝 **Purpose:** Statistically test if a parameter is increasing or decreasing over time.

                    📊 **What it shows:**
                    - Presence of a monotonic trend (increasing, decreasing, or none)
                    - Significance of the trend (p-value)
                    - Strength and direction (Tau, S statistic)

                    🔍 **How to interpret:**
                    - If **p-value < 0.05** and trend is **increasing**, the parameter is likely rising over time (potential degradation).
                    - A **decreasing** trend with significance suggests improving water quality.
                    - If p-value is **not significant**, any observed trend might be random.
                    - **Tau** closer to +1 or -1 means a stronger trend.

                    📌 **Use cases:**
                    - Detect long-term environmental changes.
                    - Provide statistical evidence to support visual trends.
                    - Prioritize stations for remediation or further investigation.
                    """)

            try:
                import pymannkendall as mk
            except ImportError:
                st.error("Please install 'pymannkendall' using pip install pymannkendall.")
                st.stop()

            trend_results = []

            for param in selected:
                series = (
                    ts_df[ts_df["CharacteristicName"] == param]
                    .sort_values("ActivityStartDate")
                    .set_index("ActivityStartDate")["ResultMeasureValue"]
                    .dropna()
                )

                if len(series) >= 8:
                    try:
                        result = mk.original_test(series)
                        trend_results.append({
                            "Parameter": param,
                            "Trend": result.trend,
                            "p-value": result.p,
                            "Tau": result.Tau,
                            "S": result.S,
                            "n": result.n
                        })
                    except Exception as e:
                        trend_results.append({
                            "Parameter": param,
                            "Trend": f"Error: {e}",
                            "p-value": None,
                            "Tau": None,
                            "S": None,
                            "n": len(series)
                        })
                else:
                    trend_results.append({
                        "Parameter": param,
                        "Trend": "Insufficient Data",
                        "p-value": None,
                        "Tau": None,
                        "S": None,
                        "n": len(series)
                    })

            trend_df = pd.DataFrame(trend_results)
            trend_df["p-value"] = trend_df["p-value"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NA")
            trend_df["Tau"] = trend_df["Tau"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "NA")

            st.dataframe(trend_df)

            csv_trend = trend_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Trend Results", data=csv_trend, file_name="trend_analysis.csv")
        # Tab 7: Water Quality Index (WQI)
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
                    📝 **Purpose:** Calculate an overall water quality score based on selected parameters and their ideal ranges.

                    📊 **What it shows:**
                    - Composite index scaled from 0 to 100
                    - Higher scores = better water quality
                    - Weighted contribution of each parameter

                    🔍 **How to interpret:**
                    - **WQI ≥ 90**: Excellent water quality
                    - **WQI 70–90**: Good
                    - **WQI 50–70**: Moderate
                    - **WQI 25–50**: Poor
                    - **WQI < 25**: Very poor

                    Each parameter’s deviation from ideal values lowers the score. A single very bad parameter can significantly reduce WQI.

                    📌 **Use cases:**
                    - Communicate water quality to non-expert audiences
                    - Compare quality across stations or time
                    - Support management decisions or public awareness
                    """)

            # Define ideal values and weights for example parameters
            ideal_values = {
                "Dissolved Oxygen": (6.0, 10.0),  # min, max ideal range
                "pH": (6.5, 8.5),
                "Turbidity": (0, 5),
                "Water Temperature (°C)": (0, 25)
            }

            weights = {
                "Dissolved Oxygen": 0.3,
                "pH": 0.2,
                "Turbidity": 0.25,
                "Water Temperature (°C)": 0.25
            }

            wqi_records = []
            wqi_df = ts_df.copy()
            for date, group in wqi_df.groupby("ActivityStartDate"):
                total_score = 0
                total_weight = 0
                for param, (min_val, max_val) in ideal_values.items():
                    param_vals = group[group["CharacteristicName"] == param]["ResultMeasureValue"]
                    if not param_vals.empty:
                        value = param_vals.values[0]
                        weight = weights[param]
                        if min_val <= value <= max_val:
                            score = 100
                        else:
                            # Penalize based on distance from ideal range
                            if value < min_val:
                                score = 100 - (min_val - value) * 10
                            else:
                                score = 100 - (value - max_val) * 10
                            score = max(score, 0)
                        total_score += score * weight
                        total_weight += weight
                if total_weight > 0:
                    wqi = total_score / total_weight
                    wqi_records.append({"Date": date, "WQI": round(wqi, 2)})

            wqi_final = pd.DataFrame(wqi_records).set_index("Date")
            if not wqi_final.empty:
                st.line_chart(wqi_final)

                st.dataframe(wqi_final)

                csv_wqi = wqi_final.to_csv().encode("utf-8")
                st.download_button("💾 Download WQI", data=csv_wqi, file_name="wqi_scores.csv")
            else:
                st.info("WQI could not be calculated. Make sure required parameters are available.")

        # Tab 8: Spatio-Temporal Heatmap
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
                    📝 **Purpose:** Display how water quality changes over time and across multiple stations, using a color-coded matrix (heatmap).

                    📊 **What it shows:**
                    - Each cell = value of a parameter at a specific station and time
                    - Colors show low to high concentration or values

                    🔍 **How to interpret:**
                    - Darker or more intense colors = higher values
                    - Compare patterns across stations (rows) or time (columns)
                    - Uniform color = stability; sudden shifts = events or anomalies

                    📌 **Use cases:**
                    - Detect spatial pollution patterns
                    - Compare stations across months or years
                    - Identify when and where interventions are needed
                    """)

            if len(selected) == 0:
                st.info("Please select at least one parameter to generate heatmap.")
            else:
                # Prepare heatmap data
                heat_df = (
                    ts_df[ts_df["CharacteristicName"].isin(selected)]
                    .groupby(["Name", "ActivityStartDate", "CharacteristicName"])["ResultMeasureValue"]
                    .mean()
                    .reset_index()
                )

                for param in selected:
                    param_df = heat_df[heat_df["CharacteristicName"] == param].pivot(
                        index="Name", columns="ActivityStartDate", values="ResultMeasureValue"
                    )

                    if param_df.empty:
                        st.warning(f"No data available for {param}.")
                        continue

                    st.markdown(f"### 🔥 Heatmap for: `{param}`")
                    fig, ax = plt.subplots(figsize=(12, min(0.5 * len(param_df), 12)))
                    sns.heatmap(param_df, cmap="YlOrRd", ax=ax, linewidths=0.1, linecolor="gray")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Station Name")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    st.download_button(
                        f"💾 Download {param} Heatmap", data=buf.getvalue(),
                        file_name=f"{param}_heatmap.png"
                    )
        # Tab 9: Anomaly Detection
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
                    📝 **Purpose:** Automatically detect unusual water quality values that deviate from normal patterns.

                    📊 **What it shows:**
                    - Points that are significantly higher or lower than typical values
                    - Uses Z-score to identify anomalies

                    🔍 **How to interpret:**
                    - Red points = anomalies
                    - These may represent pollution events, sampling errors, or sensor issues
                    - Review anomalies in context (e.g., recent rainfall or discharge)

                    📌 **Use cases:**
                    - Alert environmental managers to potential contamination
                    - Improve data quality by identifying suspect values
                    - Inform real-time monitoring systems

                    ✅ **Note:** This tool analyzes one parameter at a time. Please select a parameter with sufficient data.
                    """)

            if len(selected) == 0:
                st.info("Please select at least one parameter.")
            else:
                for param in selected:
                    st.markdown(f"### 🔍 Detecting Anomalies for `{param}`")

                    series = (
                        ts_df[ts_df["CharacteristicName"] == param]
                        .sort_values("ActivityStartDate")
                        .set_index("ActivityStartDate")["ResultMeasureValue"]
                        .dropna()
                    )

                    if len(series) < 10:
                        st.warning(f"Not enough data for {param} (needs at least 10 values).")
                        continue

                    z_scores = (series - series.mean()) / series.std()
                    anomalies = z_scores[abs(z_scores) > 2]

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(series.index, series.values, label="Value", color="gray")
                    ax.scatter(anomalies.index, series.loc[anomalies.index], color="red", label="Anomaly", zorder=5)
                    ax.axhline(series.mean(), color="blue", linestyle="--", linewidth=1, label="Mean")
                    ax.set_title(f"{param} with Anomalies")
                    ax.set_ylabel(param)
                    ax.legend()
                    st.pyplot(fig)

                    buf9 = BytesIO()
                    fig.savefig(buf9, format="png")
                    st.download_button(
                        f"💾 Download {param} Anomaly Plot",
                        data=buf9.getvalue(),
                        file_name=f"{param}_anomalies.png"
                    )
        # Tab 10: Clustering
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
                    📝 **Purpose:** Group monitoring stations into clusters based on the similarity of selected water quality parameters.

                    📊 **What it shows:**
                    - Clusters of stations that behave similarly
                    - Each cluster is labeled with a number (e.g., 0, 1, 2)

                    🔍 **How to interpret:**
                    - Stations in the same cluster share similar parameter profiles
                    - Clusters can reveal spatial patterns or shared pollution sources
                    - Use the summary table to compare average parameter values per cluster

                    📌 **Use cases:**
                    - Identify regions with similar water quality trends
                    - Prioritize stations for targeted monitoring
                    - Explore hidden patterns across space and time

                    ✅ **Note:** Select at least two or more stations and relevant parameters for meaningful clustering.
                    """)

            station_options = sorted(ts_df["MonitoringLocationIdentifier"].unique())
            selected_stations = st.multiselect("📍 Select stations for clustering", station_options, default=station_options[:5])
            n_clusters = st.slider("🔢 Select number of clusters", 2, 10, 3)

            cluster_data = (
                ts_df[ts_df["MonitoringLocationIdentifier"].isin(selected_stations) & ts_df["CharacteristicName"].isin(selected)]
                .groupby(["MonitoringLocationIdentifier", "CharacteristicName"])["ResultMeasureValue"]
                .mean()
                .unstack()
                .dropna()
            )

            if not cluster_data.empty and len(cluster_data) >= n_clusters:
                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(scaled_data)

                cluster_data["Cluster"] = labels
                st.dataframe(cluster_data.reset_index())

                # 📊 Visualize clusters
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                reduced = pca.fit_transform(scaled_data)
                fig, ax = plt.subplots()
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="Set2", s=80)
                for i, label in enumerate(cluster_data.index):
                    ax.annotate(label, (reduced[i, 0], reduced[i, 1]))
                ax.set_title("Station Clusters (PCA Projection)")
                st.pyplot(fig)

                buf10 = BytesIO()
                fig.savefig(buf10, format="png")
                st.download_button("💾 Download Cluster Plot", data=buf10.getvalue(), file_name="station_clusters.png")
            else:
                st.warning("Not enough valid data to perform clustering. Select more stations or parameters.")
