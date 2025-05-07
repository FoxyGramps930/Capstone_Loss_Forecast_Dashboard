import streamlit as st
import pandas as pd
import numpy as np
import json
import gzip
import joblib
import plotly.express as px

st.set_page_config(layout="wide")

# Load model and features
with gzip.open("rf_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

with open("feature_columns.json") as f:
    feature_cols = json.load(f)

# Load dataset
df = pd.read_csv("county_hazard_dataset.csv")
df[feature_cols] = df[feature_cols].fillna(0)
df["FIPS"] = df["NRI_ID"].str[1:]
df["STATE"] = df["STATE"].str.upper().str.strip()

# Hazard multipliers
group_definitions = {
    "Geophysical Hazards": {
        "Avalanche": "AVLN_EALT",
        "Earthquake": "ERQK_EALT",
        "Landslide": "LNDS_EALT",
        "Volcanic Activity": "VLCN_EALT",
        "Tsunami": "TSUN_EALT"
    },
    "Hydro-Meteorological Hazards": {
        "Riverine Flooding": "RFLD_EALT",
        "Coastal Flooding": "CFLD_EALT",
        "Hurricane": "HRCN_EALT",
        "Tornado": "TRND_EALT",
        "Strong Wind": "SWND_EALT",
        "Hail": "HAIL_EALT",
        "Lightning": "LTNG_EALT"
    },
    "Climatological Hazards": {
        "Cold Wave": "CWAV_EALT",
        "Drought": "DRGT_EALT",
        "Heat Wave": "HWAV_EALT",
        "Ice Storm": "ISTM_EALT",
        "Winter Weather": "WNTW_EALT",
        "Wildfire": "WFIR_EALT"
    }
}

if "multipliers" not in st.session_state:
    st.session_state.multipliers = {col: 1.0 for col in feature_cols}

# Sidebar controls
with st.sidebar:
    st.header("Hazard Multipliers")
    st.caption("Use the sliders to simulate changes in severity for different types of hazards.")

    if st.button("Reset All Multipliers"):
        for col in feature_cols:
            st.session_state.multipliers[col] = 1.0

    for group, hazards in group_definitions.items():
        with st.expander(group, expanded=False):
            for label, col in hazards.items():
                if col in feature_cols:
                    st.session_state.multipliers[col] = st.slider(
                        label,
                        min_value=0.0,
                        max_value=3.0,
                        value=st.session_state.multipliers[col],
                        step=0.1
                    )

# Apply multipliers
X = df[feature_cols].copy()
for col in feature_cols:
    X[col] *= st.session_state.multipliers.get(col, 1.0)

df["Predicted_EAL_VALT"] = model.predict(X)
df["ColorScaleEAL"] = np.sqrt(df["Predicted_EAL_VALT"])  # more sensitivity to large values

# Dashboard layout
st.title("Forecasted Disaster Losses by County")
st.markdown("""
This AI model predicts estimated disaster-related insurance claim risk over the <strong>next 5 years</strong> for each U.S. county.
**How to Use:**
- Adjust the sliders to simulate different hazard conditions.
- Hover over counties on the map for detailed estimates.
- View the top 15 counties by predicted loss below the map.
""")

# High-level metrics
total_predicted = df["Predicted_EAL_VALT"].sum()
average_predicted = df["Predicted_EAL_VALT"].mean()

col1, col2 = st.columns(2)
col1.metric("Total Predicted Annual Loss (Nationwide)", f"${total_predicted/1e9:,.2f}B")
col2.metric("Average Loss Per County", f"${average_predicted/1e6:,.2f}M")

# Choropleth map
st.subheader("Forecast Map")
fig = px.choropleth(
    df,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="ColorScaleEAL",
    color_continuous_scale=["#001f3f", "#0074D9", "#7FDBFF", "#FFDC00", "#FF4136"],
    range_color=(df["ColorScaleEAL"].min(), df["ColorScaleEAL"].quantile(0.97)),
    labels={"ColorScaleEAL": "Predicted Loss (scaled)"},
    scope="usa"
)
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="#0e1117",
    geo=dict(bgcolor="#0e1117")
)
st.plotly_chart(fig, use_container_width=True)

# Top 15 counties table
st.subheader("Top 15 Counties by Predicted Loss")
top_15 = df[["FIPS", "STATE", "COUNTY", "Predicted_EAL_VALT"]].sort_values(
    by="Predicted_EAL_VALT", ascending=False).head(15)
st.dataframe(top_15.style.format({"Predicted_EAL_VALT": "${:,.0f}"}))

# Model explanation
with st.expander("About This Forecast"):
    st.markdown("""
### Project Background

This dashboard was developed as part of a capstone project for the **University of Hartford’s Master of Science in Business Analytics** program. The goal is to help users explore how different natural hazard scenarios might impact estimated disaster-related losses across U.S. counties.

### Model Details

- **Model Type:** XGBoost Regressor (Gradient Boosted Decision Trees)
- **Target Variable:** Estimated Annual Loss (EAL_VALT)
- **Features Used:** Exposure levels to 18 natural hazards (e.g., hurricane, wildfire, flooding), along with resilience and vulnerability indicators from FEMA’s [National Risk Index](https://hazards.fema.gov/nri)
- **Preprocessing:** Included missing value imputation, feature normalization, and hazard reweighting logic
- **Training Data:** FEMA NRI dataset (2023 snapshot), ~3,000 U.S. counties

### Important Notes

- The hazard sliders simulate increases in risk levels by applying multipliers to each hazard’s baseline exposure score.
- Predictions shown are hypothetical and meant for **educational and exploratory purposes only**.
- Do not use this tool for real-world financial or policy decisions without consulting domain experts.
    """)
