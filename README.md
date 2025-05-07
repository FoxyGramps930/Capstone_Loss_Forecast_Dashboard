# Forecasted Disaster Loss Dashboard

This interactive Streamlit app forecasts disaster-related economic losses at the U.S. county level. It uses FEMA's National Risk Index and a machine learning model to estimate potential impacts under different hazard scenarios.

## Features
- Interactive sliders for 18 natural hazards
- Region and state filters
- Built-in scenario presets (Hurricane Season, Wildfire Surge, etc.)
- Feature importance chart
- Top 10 impacted counties table

## How to Run Locally
1. Install Python 3.9+
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run eal_forecast_app.py
   ```

## Data Source
Based on FEMA’s National Risk Index hazard data.

## Deployment
To deploy publicly, upload all files to GitHub and deploy with [Streamlit Cloud](https://streamlit.io/cloud).
