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

# --- GLOBAL STYLE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=PT+Serif&display=swap');

    html, body, .stApp {
        background-color: #ffffff;
        color: #222222;
        font-family: 'PT Serif', 'Georgia', serif;
        font-size: 16px;
    }

    .stSidebar > div:first-child {
        background-color: rgba(255, 255, 255, 0.96);
        padding: 1rem;
    }

    h1, h2, h3, h4, .stMarkdown, label {
        color: #0c6e72 !important;
        font-weight: bold !important;
    }

    .stSelectbox label, .stMultiselect label, .stSlider label {
        font-size: 1.1rem !important;
        font-weight: bold !important;
        color: #0c6e72 !important;
    }

    .stSelectbox div[data-baseweb="select"],
    .stMultiselect div[data-baseweb="select"],
    .stSlider {
        font-size: 1.05rem !important;
    }

    .stTextInput input, .stNumberInput input {
        font-size: 1rem !important;
        padding: 0.4rem !important;
    }

    .stButton > button {
        background-color: #cc4b00 !important;
        color: white !important;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #e76f00 !important;
    }

    .stDataFrameContainer tbody tr {
        background-color: #fef9f3 !important;
    }

    hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)
# --- SIDEBAR PARAMETER SELECTION IN BOX ---
st.sidebar.markdown("""
<div style='
    background-color: #f2f7f9;
    border-left: 5px solid #0c6e72;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 8px;
'>
    <h4 style='color:#0c6e72;'>📌 Parameter Selection</h4>
</div>
""", unsafe_allow_html=True)

available_params = sorted(df_long["CharacteristicName"].dropna().unique())
selected_param = st.sidebar.selectbox("Select a parameter to visualize on the map", available_params)

# --- SIDEBAR TIME FILTER IN BOX ---
st.sidebar.markdown("""
<div style='
    background-color: #fff9f3;
    border-left: 5px solid #cc4b00;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-radius: 8px;
'>
    <h4 style='color:#cc4b00;'>📅 Date Filter</h4>
</div>
""", unsafe_allow_html=True)

df_param = df_long[df_long["CharacteristicName"] == selected_param].copy()
df_param["YearMonth"] = df_param["ActivityStartDate"].dt.to_period("M").astype(str)
unique_periods = sorted(df_param["YearMonth"].dropna().unique())
selected_period = st.sidebar.selectbox("Select a Month-Year", ["All"] + unique_periods)

if selected_period != "All":
    df_param = df_param[df_param["YearMonth"] == selected_period]

filtered_df = df_param.copy()

# --- EXTRACT LATEST VALUES FOR MAP ---
latest_values = (
    filtered_df.sort_values("ActivityStartDate")
    .groupby("StationKey")
    .tail(1)
    .set_index("StationKey")
)
# --- MAP VIEW ---
if st.session_state.view == "map":
    m = folium.Map(
        location=[(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2],
        tiles=basemap_tiles[basemap_option]["tiles"],
        attr=basemap_tiles[basemap_option]["attr"],
        zoom_start=9,
        control_scale=True
    )
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Add county boundary layer
    folium.GeoJson(
        gdf_safe,
        name="Counties",
        style_function=lambda x: {
            "fillColor": "#b2dfdb",
            "color": "#00695c",
            "weight": 2,
            "fillOpacity": 0.2,
        },
        tooltip=folium.GeoJsonTooltip(fields=["COUNTY", "NAME"], aliases=["County:", "Name:"], sticky=False)
    ).add_to(m)

    # Add colored circle markers
    for key, row in latest_values.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row["ResultMeasureValue"]
        color = colormap(val)

        popup_content = f"""
        <div style='font-family:Georgia,serif; font-size:14px; line-height:1.5;'>
            <b style='color:#0c6e72;'>Station:</b> {row.get('Name', 'N/A')}<br>
            <b>Description:</b> {row.get('Description', 'N/A')}<br>
            <b>Basin:</b> {row.get('Basin', 'N/A')}<br>
            <b>County:</b> {row.get('County', 'N/A')}<br>
            <b>Latitude:</b> {lat:.5f}<br>
            <b>Longitude:</b> {lon:.5f}<br>
            <b>{selected_param}:</b> {val:.2f}<br>
            <b>Date:</b> {row['ActivityStartDate'].strftime('%Y-%m-%d')}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup_content, max_width=300),
        ).add_to(m)

    colormap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    st.markdown("### 🗺️ Interactive Map", unsafe_allow_html=True)
    st_data = st_folium(m, width=None, height=600)

    # Trigger detail view on click
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

    st.markdown(f"""
    <div style='
        background-color: #f2f7f9;
        padding: 1rem 2rem;
        border-left: 5px solid #0c6e72;
        margin-bottom: 1.5rem;
        border-radius: 8px;
    '>
        <h3 style='color:#0c6e72;'>📊 Station Analysis</h3>
        <p><strong>📍 Coordinates:</strong> {lat:.5f}, {lon:.5f}</p>
    </div>
    """, unsafe_allow_html=True)

    # دکمه بازگشت
    with st.form("back_form"):
        submitted = st.form_submit_button("🔙 Back to Map")
        if submitted:
            st.session_state.view = "map"
            st.rerun()

    # فیلتر داده برای ایستگاه انتخابی
    ts_df = df_long[df_long["StationKey"] == coords].sort_values("ActivityStartDate")
    subparams = sorted(ts_df["CharacteristicName"].dropna().unique())

    # باکس انتخاب پارامترها
    st.markdown("""
    <div style='
        background-color: #fff4ec;
        border-left: 5px solid #cc4b00;
        padding: 1rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        border-radius: 8px;
    '>
        <h4 style='color:#cc4b00;'>📉 Select Parameters for Analysis</h4>
    </div>
    """, unsafe_allow_html=True)

    selected = st.multiselect("Select one or more parameters", subparams, default=subparams[:1])

    # تعریف تب‌ها
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Time Series", "📉 Scatter Plot", "📊 Summary Statistics", "🧮 Correlation Heatmap",
        "📦 Boxplot", "📐 Trend Analysis", "💧 WQI", "🗺️ Spatio-Temporal Heatmap",
        "🚨 Anomaly Detection", "📍 Clustering"
    ])

    # هشدار اگر هیچ پارامتری انتخاب نشده
    if not selected:
        for tab in [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10]:
            with tab:
                st.warning("⚠️ Please select at least one parameter to display results.")
with tab1:
    st.markdown("### 📈 Time Series of Selected Parameters")

    # راهنما در یک باکس نرم
    with st.expander("❔ Help – How to use this chart"):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.7;'>
        🔹 Use this chart to explore how each parameter changes over time.<br>
        🔹 Spot seasonal variations, sudden spikes, or long-term changes.<br>
        🔹 Hover over points to view exact values and dates.
        </div>
        """, unsafe_allow_html=True)

    # پردازش و رسم نمودار
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
                ax.plot(plot_df.index, plot_df[col], marker='o', label=col)

            ax.set_ylabel("Value", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_title("📈 Time Series of Selected Parameters", fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)

            st.pyplot(fig)

            # دکمه دانلود تصویر نمودار
            buf_ts = BytesIO()
            fig.savefig(buf_ts, format="png", bbox_inches="tight")
            st.download_button(
                "💾 Download Time Series Chart",
                data=buf_ts.getvalue(),
                file_name="time_series.png"
            )
    except Exception as e:
        st.error(f"❌ Failed to generate time series plot: {e}")
with tab2:
    st.markdown("### 📉 Scatter Plot: Compare Two Parameters")

    # باکس راهنما
    with st.expander("❔ Help – Understanding scatter plots"):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.7;'>
        🔸 Scatter plots show the relationship between two variables.<br>
        🔸 Use them to detect correlation (positive, negative, or none).<br>
        🔸 Outliers may indicate unusual events or errors.
        </div>
        """, unsafe_allow_html=True)

    try:
        all_params = sorted(ts_df["CharacteristicName"].dropna().unique())
        x_var = st.selectbox("📌 X-axis Variable", all_params, key="scatter_x")
        y_options = [p for p in all_params if p != x_var]
        y_var = st.selectbox("📌 Y-axis Variable", y_options, key="scatter_y")

        scatter_df = (
            ts_df[ts_df["CharacteristicName"].isin([x_var, y_var])]
            .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
            .dropna()
        )

        if scatter_df.empty:
            st.info("⚠️ Not enough data to generate scatter plot.")
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.scatter(scatter_df[x_var], scatter_df[y_var], c='darkorange', edgecolor='black', alpha=0.7)
            ax2.set_xlabel(x_var, fontsize=12)
            ax2.set_ylabel(y_var, fontsize=12)
            ax2.set_title(f"{y_var} vs. {x_var}", fontsize=14, fontweight="bold")
            ax2.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig2)

            # دکمه دانلود
            buf_scatter = BytesIO()
            fig2.savefig(buf_scatter, format="png", bbox_inches="tight")
            st.download_button(
                "💾 Download Scatter Plot Chart",
                data=buf_scatter.getvalue(),
                file_name="scatter_plot.png"
            )
    except Exception as e:
        st.error(f"❌ Failed to generate scatter plot: {e}")
with tab3:
    st.markdown("### 📊 Descriptive Summary Statistics")

    # راهنمای تحلیلی
    with st.expander("❔ Help – How to interpret summary statistics"):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.7;'>
        🔹 **Mean / Median:** Show central tendency of parameter values.<br>
        🔹 **Std:** High value = high variability.<br>
        🔹 **Min / Max:** Check for outliers or data errors.<br>
        🔹 **Quartiles (25%, 75%):** Useful for understanding spread.
        </div>
        """, unsafe_allow_html=True)

    # پردازش و جدول آماری
    try:
        stats_df = (
            ts_df[ts_df["CharacteristicName"].isin(selected)]
            .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
            .describe()
            .T
            .round(2)
        )

        if stats_df.empty:
            st.info("⚠️ No valid data to summarize.")
        else:
            st.markdown("#### 📋 Summary Table")
            st.dataframe(stats_df.style.set_properties(**{
                'text-align': 'center',
                'background-color': '#f9f9f9',
                'border-color': 'lightgray',
            }))

            csv_stats = stats_df.to_csv().encode("utf-8")
            st.download_button(
                "💾 Download Summary as CSV",
                data=csv_stats,
                file_name="summary_statistics.csv"
            )
    except Exception as e:
        st.error(f"❌ Failed to compute summary statistics: {e}")
with tab4:
    st.markdown("### 🧮 Correlation Heatmap of Selected Parameters")

    # راهنما در باکس نرم
    with st.expander("❔ Help – Interpreting correlations"):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.7;'>
        🔸 Values close to +1 → strong positive correlation<br>
        🔸 Values close to -1 → strong negative correlation<br>
        🔸 0 means no correlation<br>
        ✅ Use this to detect potential relationships between parameters.
        </div>
        """, unsafe_allow_html=True)

    try:
        corr_df = (
            ts_df[ts_df["CharacteristicName"].isin(selected)]
            .pivot(index="ActivityStartDate", columns="CharacteristicName", values="ResultMeasureValue")
        )

        corr_matrix = corr_df.corr()

        if corr_matrix.empty or corr_matrix.isna().all().all():
            st.info("⚠️ Not enough data to generate correlation heatmap.")
        else:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, linecolor="white", cbar_kws={'label': 'Correlation'},
                square=True, ax=ax_corr
            )
            ax_corr.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
            st.pyplot(fig_corr)

            # دکمه دانلود
            buf_corr = BytesIO()
            fig_corr.savefig(buf_corr, format="png", bbox_inches="tight")
            st.download_button(
                label="💾 Download Correlation Heatmap",
                data=buf_corr.getvalue(),
                file_name="correlation_heatmap.png"
            )
    except Exception as e:
        st.error(f"❌ Failed to generate correlation heatmap: {e}")
with tab5:
    st.markdown("### 📦 Temporal Boxplots")

    # راهنما در expander
    with st.expander("❔ Help – Seasonal/Monthly/Yearly boxplots"):
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.7;'>
        📊 Boxplots show the spread and outliers of parameter values over time.<br>
        🔸 Use them to detect seasonal or annual shifts in water quality.<br>
        🔹 Wider boxes = more variability<br>
        🔹 Dots = outliers
        </div>
        """, unsafe_allow_html=True)

    # تابع تعیین فصل
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    try:
        seasonal_df = ts_df[ts_df["CharacteristicName"].isin(selected)].copy()
        seasonal_df["Month"] = seasonal_df["ActivityStartDate"].dt.strftime("%b")
        seasonal_df["Year"] = seasonal_df["ActivityStartDate"].dt.year
        seasonal_df["Season"] = seasonal_df["ActivityStartDate"].dt.month.apply(get_season)

        box_type = st.radio("🕒 Group by:", ["Season", "Month", "Year"], horizontal=True, index=0)

        if seasonal_df.empty:
            st.info("⚠️ Not enough data to generate temporal boxplots.")
        else:
            fig_box, ax_box = plt.subplots(figsize=(12, 5))

            palette = "Set2" if box_type == "Season" else "Set3" if box_type == "Month" else "Set1"

            if box_type == "Season":
                sns.boxplot(
                    x="Season", y="ResultMeasureValue", hue="CharacteristicName",
                    data=seasonal_df, palette=palette, ax=ax_box
                )
            elif box_type == "Month":
                sns.boxplot(
                    x="Month", y="ResultMeasureValue", hue="CharacteristicName",
                    data=seasonal_df, palette=palette,
                    order=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                    ax=ax_box
                )
            else:  # Year
                sns.boxplot(
                    x="Year", y="ResultMeasureValue", hue="CharacteristicName",
                    data=seasonal_df, palette=palette, ax=ax_box
                )

            ax_box.set_ylabel("Value", fontsize=12)
            ax_box.set_xlabel(box_type, fontsize=12)
            ax_box.set_title(f"{box_type}ly Distribution of Parameters", fontsize=14, fontweight="bold")
            ax_box.legend(loc="upper right")
            st.pyplot(fig_box)

            # دکمه دانلود
            buf_box = BytesIO()
            fig_box.savefig(buf_box, format="png", bbox_inches="tight")
            st.download

