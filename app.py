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

# ... [unchanged code above remains here]

        st.subheader("📈 Time Series")
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in plot_df.columns:
            ax.plot(plot_df.index, plot_df[col], 'o-', label=col)
        ax.set_ylabel("Value")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        img_bytes = BytesIO()
        fig.savefig(img_bytes, format="png")
        st.download_button("📥 Download Time Series", data=img_bytes.getvalue(), file_name="time_series.png", mime="image/png")

        st.markdown("---")

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

            img_bytes2 = BytesIO()
            fig3.savefig(img_bytes2, format="png")
            st.download_button("📥 Download Scatter Plot", data=img_bytes2.getvalue(), file_name="scatter_plot.png", mime="image/png")
        elif len(selected) > 2:
            st.info("⚠️ Scatter plot only available when exactly two parameters are selected.")

        st.markdown("---")

        st.subheader("📊 Summary Statistics")
        st.dataframe(plot_df.describe().T.style.format("{:.2f}"))

        st.markdown("---")

        st.subheader("🧮 Correlation Heatmap")
        corr = plot_df.corr()
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax2)
        st.pyplot(fig2)

        img_bytes3 = BytesIO()
        fig2.savefig(img_bytes3, format="png")
        st.download_button("📥 Download Correlation Heatmap", data=img_bytes3.getvalue(), file_name="correlation_heatmap.png", mime="image/png")
