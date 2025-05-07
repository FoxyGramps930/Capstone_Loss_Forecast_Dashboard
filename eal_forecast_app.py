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
    st.header("Adjust Hazard Multipliers")
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
                        max_value=5.0,
                        value=st.session_state.multipliers[col],
                        step=0.1
                    )

# Apply multipliers
X = df[feature_cols].copy()
for col in feature_cols:
    X[col] *= st.session_state.multipliers.get(col, 1.0)

df["Predicted_EAL_VALT"] = model.predict(X)
df["ColorScaleEAL"] = np.log1p(df["Predicted_EAL_VALT"])

# Dashboard layout
st.title("Forecasted Disaster Losses by County")
st.markdown("""
This interactive dashboard predicts expected annual losses (EAL) by U.S. county due to natural hazards.

**How to Use:**
- Adjust hazard multipliers using the sidebar.
- Hover over counties on the map for details.
- View the top 15 highest-loss counties under the map.
""")

# Choropleth map
st.subheader("Forecast Map")
fig = px.choropleth(
    df,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="ColorScaleEAL",
    color_continuous_scale="Viridis",
    range_color=(df["ColorScaleEAL"].min(), df["ColorScaleEAL"].quantile(0.95)),
    labels={"ColorScaleEAL": "Predicted Loss (log scale)"},
    scope="usa"
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

# Top 15 counties table
st.subheader("Top 15 Counties by Predicted Loss")
top_15 = df[["FIPS", "STATE", "COUNTY", "Predicted_EAL_VALT"]].sort_values(
    by="Predicted_EAL_VALT", ascending=False).head(15)
st.dataframe(top_15.style.format({"Predicted_EAL_VALT": "${:,.0f}"}))
