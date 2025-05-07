import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import gzip
import joblib
import json

# Load compressed model
with gzip.open("rf_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Load feature column names
with open("feature_columns.json") as f:
    feature_cols = json.load(f)

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stSlider > div[data-baseweb="slider"] > div {
        color: #3f51b5 !important;
    }
    input[type=range]::-webkit-slider-thumb {
        background: #3f51b5 !important;
    }
    input[type=range]::-webkit-slider-runnable-track {
        background: #c5cae9 !important;
    }
    </style>
""", unsafe_allow_html=True)


# Load model and features
import gzip
with gzip.open("rf_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Region definitions
region_map = {
    "Northeast": ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont',
                  'New Jersey', 'New York', 'Pennsylvania'],
    "Midwest": ['Illinois', 'Indiana', 'Iowa', 'Kansas', 'Michigan', 'Minnesota', 'Missouri',
                'Nebraska', 'North Dakota', 'Ohio', 'South Dakota', 'Wisconsin'],
    "South": ['Alabama', 'Arkansas', 'Delaware', 'Florida', 'Georgia', 'Kentucky', 'Louisiana',
              'Maryland', 'Mississippi', 'North Carolina', 'Oklahoma', 'South Carolina', 'Tennessee',
              'Texas', 'Virginia', 'West Virginia', 'District of Columbia'],
    "West": ['Arizona', 'California', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico',
             'Oregon', 'Utah', 'Washington', 'Wyoming']
}
region_map["All Regions"] = sum(region_map.values(), [])

region_centers = {
    "Northeast": {"lat": 43.0, "lon": -71.5},
    "Midwest": {"lat": 42.0, "lon": -93.0},
    "South": {"lat": 33.0, "lon": -84.0},
    "West": {"lat": 39.5, "lon": -111.5},
    "All Regions": {"lat": 39.5, "lon": -98.35}
}

# Load dataset
df = pd.read_csv("county_hazard_dataset.csv")
df[feature_cols] = df[feature_cols].fillna(0)
df = df[~df["STATE"].isin(["Alaska", "Hawaii"])].copy()
df["FIPS"] = df["NRI_ID"].str[1:]
df["BASE_EAL_VALT"] = model.predict(df[feature_cols])

# Hazard label map
hazard_name_map = {
    'CWAV_EALT': 'Cold Wave', 'DRGT_EALT': 'Drought', 'ERQK_EALT': 'Earthquake',
    'HAIL_EALT': 'Hail', 'HWAV_EALT': 'Heat Wave', 'HRCN_EALT': 'Hurricane',
    'ISTM_EALT': 'Ice Storm', 'LNDS_EALT': 'Landslide', 'LTNG_EALT': 'Lightning',
    'RFLD_EALT': 'River Flooding', 'SWND_EALT': 'Strong Wind', 'TRND_EALT': 'Tornado',
    'WFIR_EALT': 'Wildfire', 'WNTW_EALT': 'Winter Weather'
}

# Page Title and Intro
st.title("Forecasted Disaster Losses by County")
st.markdown("""
This interactive dashboard allows users to explore estimated annual losses from natural hazards 
in U.S. counties based on FEMA risk scores. Use the sliders to simulate different hazard conditions 
(e.g., more frequent hurricanes or colder winters) and see how predicted losses change by region and county.

**How to Use:**
- Use the region and state filters in the sidebar to zoom into specific areas.
- Select a hazard scenario or adjust sliders manually to model changing risk.
- Hover over counties for detailed info and scroll down for top 10 impacted areas.
""")

# Sidebar Filters
with st.sidebar.expander("Region & State Filter", expanded=False):
    selected_region = st.selectbox("Region", list(region_map.keys()), index=4)
    default_states = region_map[selected_region]
    selected_states = st.multiselect("States", sorted(df["STATE"].unique()), default=default_states)
    if st.button("Clear All States"):
        selected_states = []
    if st.button("Select All States"):
        selected_states = sorted(df["STATE"].unique())

# Presets
st.sidebar.subheader("Hazard Scenario Presets")
preset = st.sidebar.radio("Choose a Preset", ["Default", "Hurricane Season", "Wildfire Surge", "Freeze Event"], index=0)

if "multipliers" not in st.session_state:
    st.session_state.multipliers = {col: 1.0 for col in feature_cols}

preset_values = {
    "Default": {col: 1.0 for col in feature_cols},
    "Hurricane Season": {"HRCN_EALT": 2.0, "RFLD_EALT": 1.5, "SWND_EALT": 1.5},
    "Wildfire Surge": {"WFIR_EALT": 2.0, "HWAV_EALT": 1.5},
    "Freeze Event": {"CWAV_EALT": 2.0, "ISTM_EALT": 1.5, "WNTW_EALT": 1.5}
}
for col in feature_cols:
    st.session_state.multipliers[col] = preset_values[preset].get(col, 1.0)

st.sidebar.subheader("Hazard Multipliers")
if st.sidebar.button("Reset All Multipliers"):
    for col in feature_cols:
        st.session_state.multipliers[col] = 1.0

for col in feature_cols:
    label = hazard_name_map.get(col, col)
    st.session_state.multipliers[col] = st.sidebar.slider(label, 0.0, 5.0, st.session_state.multipliers[col], 0.1)

# Recalculate
df_filtered = df[df["STATE"].isin(selected_states)].copy()
X = df_filtered[feature_cols].copy()
for col in feature_cols:
    X[col] *= st.session_state.multipliers[col]

df_filtered["Predicted_EAL_VALT"] = model.predict(X)
df_filtered["ColorScaleEAL"] = np.sqrt(df_filtered["Predicted_EAL_VALT"])
df_filtered["Delta"] = df_filtered["Predicted_EAL_VALT"] - df_filtered["BASE_EAL_VALT"]

# Summary
st.subheader("National Forecast Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Loss", f"${df_filtered['Predicted_EAL_VALT'].sum():,.0f}")
col2.metric("Average per County", f"${df_filtered['Predicted_EAL_VALT'].mean():,.0f}")
col3.metric("Counties Displayed", len(df_filtered))

# Map
st.subheader("Forecast Map by County")
center_coords = region_centers[selected_region]
fig = px.choropleth(
    df_filtered,
    geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
    locations="FIPS",
    color="ColorScaleEAL",
    color_continuous_scale="Viridis",
    range_color=(df_filtered["ColorScaleEAL"].min(), df_filtered["ColorScaleEAL"].quantile(0.95)),
    labels={"ColorScaleEAL": "Predicted Loss"},
    hover_name="COUNTY",
    hover_data={"STATE": True, "Predicted_EAL_VALT": ":,.0f", "Delta": ":+,.0f"},
    scope="usa",
    center=center_coords
)
fig.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)"), height=800, margin={"r":0, "t":0, "l":0, "b":0})
st.plotly_chart(fig, use_container_width=True)

# Top 10
st.subheader("Top 10 Counties by Estimated Loss")
top10 = df_filtered.sort_values(by="Predicted_EAL_VALT", ascending=False).head(10)
st.dataframe(top10[["STATE", "COUNTY", "Predicted_EAL_VALT", "Delta"]].style.format({
    "Predicted_EAL_VALT": "${:,.0f}", "Delta": "{:+,.0f}"
}))

# Feature Importance Chart
st.subheader("Hazard Importance in Model Predictions")
importance_df = pd.read_csv("feature_importance_chart.csv")
fig2 = px.bar(importance_df, x="Importance", y="Hazard", orientation="h",
              title="Relative Importance of Each Hazard",
              labels={"Importance": "Model Weight", "Hazard": "Hazard Type"},
              color="Importance", color_continuous_scale="Blues")
st.plotly_chart(fig2, use_container_width=True)
