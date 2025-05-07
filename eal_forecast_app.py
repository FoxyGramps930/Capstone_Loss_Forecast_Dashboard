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

# Ensure all state codes are uppercase and clean
df["STATE"] = df["STATE"].str.upper().str.strip()

# Hazard multipliers
hazard_groups = {
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

with st.sidebar:
    st.header("Hazard Multipliers")
    if st.button("Reset All Multipliers"):
        for col in feature_cols:
            st.session_state.multipliers[col] = 1.0

    for group, hazards in hazard_groups.items():
        with st.expander(group, expanded=False):
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
df["ColorScaleEAL"] = np.log1p(df["Predicted_EAL_VALT"])

# No regional filtering
df_filtered = df.copy()

# Main dashboard content
st.title("Forecasted Disaster Losses by County")
st.markdown("""
This dashboard forecasts expected annual losses (EAL) by county across the United States based on user-defined hazard multipliers.
Use the filters to explore how different hazard scenarios impact forecasted losses.
""")

# Choropleth map
st.subheader("Forecast Map")
fig = px.choropleth(
    df_filtered,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="ColorScaleEAL",
    color_continuous_scale="Viridis",
    range_color=(df_filtered["ColorScaleEAL"].min(), df_filtered["ColorScaleEAL"].quantile(0.95)),
    labels={"ColorScaleEAL": "Predicted Loss (log scale)"},
    scope="usa"
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

# Top 15 counties table
st.subheader("Top 15 Counties by Predicted Loss")
top_15 = df_filtered[["FIPS", "STATE", "COUNTY", "Predicted_EAL_VALT"]].sort_values(
    by="Predicted_EAL_VALT", ascending=False).head(15)
st.dataframe(top_15.style.format({"Predicted_EAL_VALT": "${:,.0f}"}))
