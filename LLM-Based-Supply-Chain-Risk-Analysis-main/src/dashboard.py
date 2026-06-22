import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

from hybrid_predict import predict


# -------------------------
# LOAD DATA (optional if you want analytics)
# -------------------------
try:
    df = pd.read_json("outputs/labeled_data.json")
except:
    df = None


st.set_page_config(page_title="Supply Chain Risk Dashboard", layout="wide")

st.title("🚢 Supply Chain Risk Intelligence System")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Input Section")

user_text = st.sidebar.text_area("Enter supply chain news text")

run_btn = st.sidebar.button("Analyze")


# -------------------------
# MAIN PREDICTION
# -------------------------
if run_btn and user_text:

    result = predict(user_text)

    st.subheader("🔍 Prediction Result")

    st.json(result)


# -------------------------
# DATA ANALYSIS SECTION
# -------------------------
if df is not None:

    st.markdown("## 📊 Dataset Insights")

    col1, col2 = st.columns(2)

    # Risk distribution
    with col1:

        fig1 = px.histogram(
            df,
            x="risk_category",
            title="Risk Category Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)


    # Source of labeling (ML vs LLM)
    with col2:

        if "mode" in df.columns:

            fig2 = px.pie(
                df,
                names="mode",
                title="ML vs LLM Usage"
            )
            st.plotly_chart(fig2, use_container_width=True)


    # Severity distribution (if exists)
    if "severity_score" in df.columns:

        st.markdown("## ⚠️ Severity Analysis")

        fig3 = px.histogram(
            df,
            x="severity_score",
            nbins=10,
            title="Severity Score Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)


# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Built with ML + LLM Hybrid System 🚀")