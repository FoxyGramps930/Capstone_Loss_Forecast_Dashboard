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

# Region Mapping
region_map = {
    "All Regions": [],
    "Northeast": ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"],
    "Midwest": ["OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    "South": ["DE", "MD", "DC", "VA", "WV", "NC", "SC", "GA", "FL", "KY", "TN", "MS", "AL", "OK", "TX", "AR", "LA"],
    "West": ["ID", "MT", "WY", "NV", "UT", "CO", "AZ", "NM", "AK", "WA", "OR", "CA", "HI"]
}

all_regions = list(region_map.keys())

# Sidebar filters
with st.sidebar:
    selected_region = st.selectbox("Select Region", all_regions)
    selected_states = region_map[selected_region] if selected_region != "All Regions" else df["STATE"].unique().tolist()

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

# Predictions
df["BASE_EAL_VALT"] = model.predict(df[feature_cols])
df["Predicted_EAL_VALT"] = model.predict(X)
df["Delta_EAL"] = df["Predicted_EAL_VALT"] - df["BASE_EAL_VALT"]
df["ColorScaleEAL"] = df["Delta_EAL"]

# Filter by selected region
df_filtered = df[df["STATE"].isin(selected_states)]
if df_filtered.empty:
    st.warning("No counties match the selected filters. Showing all counties instead.")
    df_filtered = df.copy()

# Main dashboard content
st.title("Forecasted Disaster Loss Differences by County")
st.markdown("""
This dashboard forecasts expected annual losses (EAL) by county across the United States based on user-defined hazard multipliers.
The map shows the difference between the base forecast and your adjusted scenario. Positive values indicate increased losses.
""")

# Choropleth map showing difference
st.subheader("Forecast Difference Map")
fig = px.choropleth(
    df_filtered,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="ColorScaleEAL",
    color_continuous_scale="RdBu",
    range_color=(df_filtered["ColorScaleEAL"].min(), df_filtered["ColorScaleEAL"].max()),
    labels={"ColorScaleEAL": "Change in Predicted Loss"},
    scope="usa"
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

# Top 15 counties by increase
st.subheader("Top 15 Counties by Increase in Predicted Loss")
top_15 = df_filtered[["FIPS", "STATE", "COUNTY", "Delta_EAL"]].sort_values(
    by="Delta_EAL", ascending=False).head(15)
st.dataframe(top_15.style.format({"Delta_EAL": "${:,.0f}"}))
