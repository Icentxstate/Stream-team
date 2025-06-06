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

# --- Create Tabs for Advanced Analysis ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📈 Time Series",                     # Tab 1: نمایش سری زمانی پارامترهای انتخابی
    "📉 Scatter Plot",                   # Tab 2: نمودار پراکندگی بین دو پارامتر
    "📊 Summary Statistics",             # Tab 3: جدول آمار توصیفی
    "🧮 Correlation Heatmap",            # Tab 4: ماتریس همبستگی بین پارامترها
    "📦 Seasonal Boxplot",               # Tab 5: جعبه نمودار برای تحلیل فصلی
    "📐 Trend Analysis",                 # Tab 6: آزمون روند زمانی Mann-Kendall
    "💧 WQI",                            # Tab 7: شاخص ترکیبی کیفیت آب (Water Quality Index)
    "🗺️ Spatio-Temporal Heatmap",       # Tab 8: نقشه حرارتی زمانی-مکانی
    "🚨 Anomaly Detection",              # Tab 9: شناسایی داده‌های پرت با روش IQR
    "📍 Clustering"                      # Tab 10: خوشه‌بندی ایستگاه‌ها با KMeans
])
        # --- Tab 1: Time Series ---
        with tab1:
            st.subheader("📈 Time Series")
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

        # --- Tab 2: Scatter Plot ---
        with tab2:
            st.subheader("📉 Scatter Plot")
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

        # --- Tab 3: Summary Statistics ---
        with tab3:
            st.subheader("📊 Summary Statistics")
            stats = plot_df.describe().T
            st.dataframe(stats.style.format("{:.2f}"))

            csv_stats = stats.to_csv().encode("utf-8")
            st.download_button("💾 Download Summary CSV", data=csv_stats, file_name="summary_statistics.csv")

        # --- Tab 4: Correlation Heatmap ---
        with tab4:
            st.subheader("🧮 Correlation Heatmap")
            corr = plot_df.corr()
            if not corr.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
                st.pyplot(fig2)
            else:
                st.info("Not enough data for correlation heatmap.")
    else:
        st.warning("⚠️ Please select at least one parameter to analyze.")

        # --- Tab 5: Seasonal Boxplot ---
        with tab5:
            st.subheader("📦 Seasonal Boxplot")

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
            seasonal_df["Season"] = seasonal_df["ActivityStartDate"].dt.month.apply(get_season)

            if not seasonal_df.empty:
                fig5, ax5 = plt.subplots(figsize=(10, 5))
                sns.boxplot(
                    x="Season", 
                    y="ResultMeasureValue", 
                    hue="CharacteristicName", 
                    data=seasonal_df,
                    palette="Set2"
                )
                ax5.set_title("Seasonal Distribution")
                ax5.set_ylabel("Value")
                st.pyplot(fig5)
            else:
                st.info("Not enough data to generate seasonal boxplot.")

        # --- Tab 6: Trend Analysis (Mann-Kendall) ---
        with tab6:
            st.subheader("📐 Mann-Kendall Trend Test")

            try:
                import pymannkendall as mk
            except ImportError:
                st.error("Please install 'pymannkendall' using `pip install pymannkendall`.")
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
                    result = mk.original_test(series)
                    trend_results.append({
                        "Parameter": param,
                        "Trend": result.trend,
                        "p-value": result.p,
                        "Tau": result.Tau,
                        "S": result.S,
                        "n": result.n
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
            st.dataframe(trend_df.style.format({
                "p-value": "{:.4f}",
                "Tau": "{:.2f}"
            }))

            csv_trend = trend_df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Trend Results", data=csv_trend, file_name="trend_analysis.csv")

        # --- Tab 7: Water Quality Index (WQI) ---
        with tab7:
            st.subheader("💧 Water Quality Index (WQI)")

            wqi_df = ts_df.copy()
            parameters = sorted(wqi_df["CharacteristicName"].dropna().unique())

            selected_wqi_params = st.multiselect("🧪 Select parameters for WQI", parameters, default=parameters[:3])

            if selected_wqi_params:
                st.markdown("### ⚖️ Assign weights (total should sum to 1):")
                weights = {}
                total_weight = 0.0
                for param in selected_wqi_params:
                    w = st.slider(f"Weight for {param}", 0.0, 1.0, 1.0 / len(selected_wqi_params), 0.05, key=f"w_{param}")
                    weights[param] = w
                    total_weight += w

                if abs(total_weight - 1.0) > 0.01:
                    st.warning("⚠️ Total weights must sum to 1. Adjust sliders.")
                else:
                    norm_df = pd.DataFrame()

                    for param in selected_wqi_params:
                        sub = wqi_df[wqi_df["CharacteristicName"] == param].copy()
                        sub = sub[["ActivityStartDate", "ResultMeasureValue"]].dropna().copy()
                        sub = sub.set_index("ActivityStartDate").resample("M").mean().reset_index()
                        min_val = sub["ResultMeasureValue"].min()
                        max_val = sub["ResultMeasureValue"].max()
                        sub["Normalized"] = 100 * (sub["ResultMeasureValue"] - min_val) / (max_val - min_val + 1e-6)
                        sub["Weighted"] = sub["Normalized"] * weights[param]
                        sub["Parameter"] = param
                        norm_df = pd.concat([norm_df, sub])

                    wqi_monthly = norm_df.groupby("ActivityStartDate")["Weighted"].sum().reset_index()
                    wqi_monthly["WQI Category"] = pd.cut(
                        wqi_monthly["Weighted"],
                        bins=[0, 25, 50, 75, 100],
                        labels=["Poor", "Moderate", "Good", "Excellent"]
                    )

                    st.line_chart(wqi_monthly.set_index("ActivityStartDate")["Weighted"])

                    st.dataframe(wqi_monthly)

                    csv_wqi = wqi_monthly.to_csv(index=False).encode("utf-8")
                    st.download_button("💾 Download WQI Data", data=csv_wqi, file_name="wqi_results.csv")
            else:
                st.info("Please select at least one parameter.")
        # --- Tab 8: Spatio-Temporal Heatmap ---
        with tab8:
            st.subheader("🗺️ Spatial-Temporal Heatmap")

            param_options = sorted(ts_df["CharacteristicName"].dropna().unique())
            selected_param_ht = st.selectbox("Select Parameter", param_options, key="heatmap_param")

            heat_df = ts_df[ts_df["CharacteristicName"] == selected_param_ht].copy()
            heat_df["YearMonth"] = heat_df["ActivityStartDate"].dt.to_period("M").astype(str)
            months_avail = sorted(heat_df["YearMonth"].dropna().unique())

            selected_month = st.selectbox("Select Month", months_avail, key="heatmap_month")

            heat_month_df = heat_df[heat_df["YearMonth"] == selected_month]

            if heat_month_df.empty:
                st.warning("⚠️ No data available for this month.")
            else:
                avg_vals = heat_month_df.groupby("StationKey").agg({
                    "ResultMeasureValue": "mean",
                    "Latitude": "first",
                    "Longitude": "first"
                }).reset_index()

                min_val = avg_vals["ResultMeasureValue"].min()
                max_val = avg_vals["ResultMeasureValue"].max()
                colormap_ht = StepColormap(
                    colors=['#3288bd', '#99d8c9', '#e6f598', '#fee08b', '#f46d43', '#d53e4f'],
                    index=np.linspace(min_val, max_val, 6),
                    vmin=min_val,
                    vmax=max_val
                )

                m_ht = folium.Map(
                    location=[avg_vals["Latitude"].mean(), avg_vals["Longitude"].mean()],
                    zoom_start=10
                )

                for _, row in avg_vals.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=6,
                        color=colormap_ht(row["ResultMeasureValue"]),
                        fill=True,
                        fill_opacity=0.85,
                        popup=f"{selected_param_ht}: {row['ResultMeasureValue']:.2f}"
                    ).add_to(m_ht)

                colormap_ht.add_to(m_ht)
                st_folium(m_ht, height=600, width=None)
        # --- Tab 9: Anomaly Detection ---
        with tab9:
            st.subheader("🚨 Anomaly Detection (IQR Method)")

            selected_param_anom = st.selectbox("Select parameter for anomaly check", subparams, key="anom_param")

            anomaly_df = ts_df[ts_df["CharacteristicName"] == selected_param_anom].copy()
            anomaly_df = anomaly_df.sort_values("ActivityStartDate")
            anomaly_df = anomaly_df[["ActivityStartDate", "ResultMeasureValue"]].dropna()

            if len(anomaly_df) < 8:
                st.warning("⚠️ Not enough data points.")
            else:
                Q1 = anomaly_df["ResultMeasureValue"].quantile(0.25)
                Q3 = anomaly_df["ResultMeasureValue"].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                anomaly_df["Anomaly"] = ((anomaly_df["ResultMeasureValue"] < lower_bound) |
                                         (anomaly_df["ResultMeasureValue"] > upper_bound))

                st.markdown(f"**Anomaly Thresholds:** Lower = {lower_bound:.2f}, Upper = {upper_bound:.2f}")
                st.write(f"🔴 {anomaly_df['Anomaly'].sum()} anomalies detected out of {len(anomaly_df)} records.")

                fig_anom, ax_anom = plt.subplots(figsize=(10, 5))
                ax_anom.plot(anomaly_df["ActivityStartDate"], anomaly_df["ResultMeasureValue"], label="Values", marker='o')
                ax_anom.scatter(anomaly_df[anomaly_df["Anomaly"]]["ActivityStartDate"],
                                anomaly_df[anomaly_df["Anomaly"]]["ResultMeasureValue"],
                                color='red', label="Anomalies", zorder=5)
                ax_anom.axhline(lower_bound, color='orange', linestyle='--', label="Lower Bound")
                ax_anom.axhline(upper_bound, color='orange', linestyle='--', label="Upper Bound")
                ax_anom.legend()
                ax_anom.set_ylabel("Value")
                ax_anom.set_title(f"Anomaly Detection for {selected_param_anom}")
                st.pyplot(fig_anom)

                st.dataframe(anomaly_df)

                csv_anom = anomaly_df.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Anomaly Table", data=csv_anom, file_name="anomaly_results.csv")
        # --- Tab 10: Clustering ---
        with tab10:
            st.subheader("📍 Clustering Stations (KMeans)")

            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans

            cluster_params = st.multiselect("Select parameters for clustering", subparams, default=subparams[:3])

            if len(cluster_params) < 2:
                st.warning("Please select at least two parameters for clustering.")
            else:
                cluster_df = ts_df[ts_df["CharacteristicName"].isin(cluster_params)].copy()
                pivot_df = (
                    cluster_df
                    .groupby(["StationKey", "CharacteristicName"])["ResultMeasureValue"]
                    .mean()
                    .unstack()
                    .dropna()
                )

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(pivot_df)

                k = st.slider("Select number of clusters (k)", 2, 6, 3)
                kmeans = KMeans(n_clusters=k, random_state=0)
                labels = kmeans.fit_predict(X_scaled)
                pivot_df["Cluster"] = labels
                pivot_df["Latitude"] = pivot_df.index.map(lambda x: float(x.split(",")[0]))
                pivot_df["Longitude"] = pivot_df.index.map(lambda x: float(x.split(",")[1]))

                st.write(pivot_df[["Cluster"] + cluster_params])

                # Visualize on map
                cmap = plt.cm.get_cmap("tab10", k)
                m_cluster = folium.Map(location=[pivot_df["Latitude"].mean(), pivot_df["Longitude"].mean()], zoom_start=9)

                for _, row in pivot_df.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=7,
                        color=f"#{int(cmap(row['Cluster'])[0]*255):02x}{int(cmap(row['Cluster'])[1]*255):02x}{int(cmap(row['Cluster'])[2]*255):02x}",
                        fill=True,
                        fill_opacity=0.9,
                        popup=f"Cluster: {row['Cluster']}"
                    ).add_to(m_cluster)

                st_folium(m_cluster, height=600)

                csv_clust = pivot_df.reset_index().to_csv(index=False).enc

