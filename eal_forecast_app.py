
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

# Hazard groupings for sliders
hazard_groups = {
    "Geophysical Hazards": {
        "Avalanche": "AVLN_EALT",
        "Earthquake": "ERQK_EALT",
        "Landslide": "LNDS_EALT",
        "Volcanic Activity": "VLCN_EALT",
        "Tsunami": "TSUN_EALT"
    },
    "Hydrological & Meteorological Hazards": {
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

# Create a label lookup
hazard_label_lookup = {v: k for group in hazard_groups.values() for k, v in group.items()}

# Load data
df = pd.read_csv("county_hazard_dataset.csv")
df[feature_cols] = df[feature_cols].fillna(0)
df["FIPS"] = df["NRI_ID"].str[1:]
df["BASE_EAL_VALT"] = model.predict(df[feature_cols])

# UI Title + Instructions
st.title("Forecasted Disaster Losses by County")

st.markdown("""
Use the sliders below to simulate changes in hazard severity.

### How to Use:
- Adjust hazard multipliers to simulate increasing or decreasing risk
- Forecast updates automatically
- Hover over counties to view estimated losses
""")

# Sidebar UI
st.sidebar.header("Hazard Multipliers")

# Initialize session state
if "multipliers" not in st.session_state:
    st.session_state.multipliers = {col: 1.0 for col in feature_cols}

# Reset button
if st.sidebar.button("Reset All Multipliers"):
    st.session_state.multipliers = {col: 1.0 for col in feature_cols}

# Grouped sliders
for group, hazards in hazard_groups.items():
    with st.sidebar.expander(group, expanded=False):
        for label, col in hazards.items():
            if col in feature_cols:
                st.session_state.multipliers[col] = st.slider(
                    label, 0.0, 5.0, st.session_state.multipliers[col], 0.1
                )

# Apply multipliers and predict
X = df[feature_cols].copy()
for col in feature_cols:
    X[col] *= st.session_state.multipliers.get(col, 1.0)

df["Predicted_EAL_VALT"] = model.predict(X)
df["ColorScaleEAL"] = np.sqrt(df["Predicted_EAL_VALT"])

# Summary stats
st.markdown("### National Forecast Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Loss", f"${df['Predicted_EAL_VALT'].sum():,.0f}")
col2.metric("Avg per County", f"${df['Predicted_EAL_VALT'].mean():,.0f}")
col3.metric("Counties", f"{len(df):,}")

# Choropleth map
st.subheader("Forecast Map by County")
fig = px.choropleth(
    df,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="ColorScaleEAL",
    color_continuous_scale="Viridis",
    range_color=(df["ColorScaleEAL"].min(), df["ColorScaleEAL"].quantile(0.95)),
    labels={"ColorScaleEAL": "Predicted Loss"},
    scope="usa"
)
fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
st.plotly_chart(fig, use_container_width=True)
