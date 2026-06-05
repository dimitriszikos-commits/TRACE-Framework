import streamlit as st
import pandas as pd
import os
import streamlit.components.v1 as components

# ---> THIS IS THE ENGINE CONNECTION <---
from Attempt6 import run_master_pipeline

st.set_page_config(page_title="Clinical Sequence Extractor", layout="wide")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; background-color: #1E3A8A; color: white; height: 50px; font-size: 18px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONFIGURATION 
# ==========================================
st.sidebar.title("⚙️ System Parameters")

st.sidebar.header("1. Cohort & Targets")
TARGET_COL = st.sidebar.text_input("Target Outcome Column", value="READMIT_30")
MIN_TRAIN_PREVALENCE = st.sidebar.number_input("Support Floor (%)", value=0.5, step=0.1)
MIN_TRAIN_PREVALENCE_PCT = MIN_TRAIN_PREVALENCE / 100.0

st.sidebar.header("2. Engine Thresholds")
ENABLE_JACCARD = st.sidebar.toggle("Enable Jaccard Bundling", value=True)
JACCARD_THRESHOLD = st.sidebar.slider("Jaccard Threshold", 0.50, 1.00, 0.90, disabled=not ENABLE_JACCARD)
MIN_CO_OCCURRENCE = st.sidebar.number_input("Min Co-occurrence (Count)", value=5, step=1)
ASYM_FLOOR = st.sidebar.slider("Asymmetry Floor", 0.0, 0.50, 0.10)

st.sidebar.header("3. Causal Gates (pCMI)")
COLLIDER_INDEP_LOWER = st.sidebar.number_input("Independence Lower Bound", value=0.75, step=0.05)
COLLIDER_INDEP_UPPER = st.sidebar.number_input("Independence Upper Bound", value=1.33, step=0.05)
COLLIDER_DEP_LOWER = st.sidebar.number_input("Dependence Lower Bound", value=0.66, step=0.05)
COLLIDER_DEP_UPPER = st.sidebar.number_input("Dependence Upper Bound", value=1.50, step=0.05)

st.sidebar.header("4. Negative Nodes")
ENABLE_NEGATIVES = st.sidebar.toggle("Extract Informative Absences", value=True)
MIN_PREVALENCE_FOR_NEGATIVE = st.sidebar.slider("Negative Anchor Support Floor", 0.01, 0.20, 0.05)

st.sidebar.header("5. Advanced Settings")
SAMPLE_SIZE = st.sidebar.number_input("Data Sample Size (0 = All Data)", value=0)
AUTO_OPTIMIZE = st.sidebar.toggle("Enable Auto-Optimization", value=True)
MAX_SEQ_LEN = st.sidebar.number_input("Maximum Pathway Length", value=5, min_value=2, max_value=15)
USE_NORMALIZED = st.sidebar.toggle("Use Normalized Asymmetry", value=False)
MAX_PRUNE = st.sidebar.number_input("Max Sequences for ML Model", value=500, step=50)

# ==========================================
# MAIN EXECUTION AREA
# ==========================================
st.title("Clinical Sequence Extractor: Master Pipeline")
st.write("Configure your mathematical thresholds in the sidebar, then execute the pipeline.")
st.divider()

if st.button("🚀 Execute Pipeline"):
    with st.spinner("Engine is running. Extracting clinical sequences from cohort..."):
        try:
            # ---> THIS ACTUALLY RUNS THE MATH IN ATTEMPT6.PY <---
            run_master_pipeline(
                target_col=TARGET_COL, 
                min_train_prev_pct=MIN_TRAIN_PREVALENCE_PCT, 
                enable_jaccard=ENABLE_JACCARD, 
                jaccard_threshold=JACCARD_THRESHOLD, 
                min_co_occurrence=MIN_CO_OCCURRENCE, 
                asym_floor=ASYM_FLOOR,
                collider_indep_lower=COLLIDER_INDEP_LOWER,
                collider_indep_upper=COLLIDER_INDEP_UPPER,
                collider_dep_lower=COLLIDER_DEP_LOWER,
                collider_dep_upper=COLLIDER_DEP_UPPER,
                enable_negatives=ENABLE_NEGATIVES,
                min_prev_for_negative=MIN_PREVALENCE_FOR_NEGATIVE,
                sample_size=int(SAMPLE_SIZE),
                auto_optimize=AUTO_OPTIMIZE,
                max_seq_len=int(MAX_SEQ_LEN),
                use_normalized_diff=USE_NORMALIZED,
                max_prune_search=int(MAX_PRUNE)
            )
            
            st.success("✅ Pipeline Execution Complete! Visualizations rendered below.")
            
            st.divider()
            st.subheader("Validated Clinical Pathways")
            
            dashboard_path = "DASHBOARD_2_All_Validated_Pathways.html"
            if os.path.exists(dashboard_path):
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    html_data = f.read()
                components.html(html_data, height=800, scrolling=True)
            else:
                st.warning("Dashboard file not found. Ensure the pipeline saved it correctly.")

        except Exception as e:
            st.error(f"Engine Failed: {str(e)}")
