import streamlit as st
import pandas as pd
import glob
import os
import altair as alt
import numpy as np
import plotly.express as px
from datetime import datetime

# --------------------------------------------------------------------------------
# 1) Page Setup
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Flight Delay and Departure Overview Dashboard</h4>",
            unsafe_allow_html=True)

# CSS for uniform cards
st.markdown(
    """
    <style>
    .card {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
        text-align: center;
        box-sizing: border-box;
        height: 100px; /* Adjust as needed */
    }
    .card h4 {
        font-size: 14px;
        margin: 0;
        line-height: 1.2;
    }
    .card p {
        font-size: 26px; /* Larger number text */
        margin: 0;
        font-weight: bold;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------------
# 2) Load Data
# --------------------------------------------------------------------------------


@st.cache_data
def load_data():
    path = r'C:\Users\tempo\Desktop\Final\flight_app\pages'
    parquet_files = glob.glob(os.path.join(path, '*.parquet'))
    if not parquet_files:
        st.error("No parquet files found in the specified directory!")
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(file)
                   for file in parquet_files], ignore_index=True)
    date_format = "%d-%m-%Y %H:%M:%S"
    time_cols = ["FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME",
                 "FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME"]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    # Compute delays in minutes
    df['DEPTDEL'] = (df["ACTUAL OFF BLOCK TIME"] -
                     df["FILED OFF BLOCK TIME"]).dt.total_seconds() / 60
    df['ARVLDEL'] = (df["ACTUAL ARRIVAL TIME"] -
                     df["FILED ARRIVAL TIME"]).dt.total_seconds() / 60
    return df


df = load_data()

# --------------------------------------------------------------------------------
# 3) Sidebar Controls
# --------------------------------------------------------------------------------
st.sidebar.header("Filters")

# Default to December 1 - December 2, 2018
start_date = st.sidebar.date_input(
    "Start Date", value=datetime(2018, 12, 1).date())
end_date = st.sidebar.date_input(
    "End Date", value=datetime(2018, 12, 2).date())
if start_date > end_date:
    st.sidebar.error("Error: Start date must be before End date.")

delay_threshold = st.sidebar.slider(
    "Delay Threshold (minutes)", min_value=15, max_value=60, value=15, step=15)
airport_options = sorted(df['ADEP'].dropna().unique()) if not df.empty else []
selected_airports = st.sidebar.multiselect(
    "Select Airport(s) (leave empty for all)", options=airport_options)

# Number of airports to display in scatter plot
n_display = st.sidebar.slider(
    "Number of Airports to Display (Scatter Plot)", min_value=5, max_value=25, value=10, step=5)

# --------------------------------------------------------------------------------
# 4) Data Filtering
# --------------------------------------------------------------------------------
df_filtered = df[(df["FILED OFF BLOCK TIME"].dt.date >= start_date) &
                 (df["FILED OFF BLOCK TIME"].dt.date <= end_date)]
if selected_airports:
    df_filtered = df_filtered[df_filtered['ADEP'].isin(selected_airports)]

# --------------------------------------------------------------------------------
# 5) Key Metrics (Cards)
# --------------------------------------------------------------------------------
num_departures = len(df_filtered)
num_delayed = df_filtered[df_filtered['DEPTDEL'] > delay_threshold].shape[0]
delayed_percentage = (num_delayed / num_departures *
                      100) if num_departures > 0 else 0
avg_dep_delay = df_filtered['DEPTDEL'].mean() if num_departures > 0 else 0
avg_arr_delay = df_filtered['ARVLDEL'].mean() if num_departures > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"<div class='card'><h4>Total Departures</h4><p>{num_departures}</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(
        f"<div class='card'><h4>Delayed Flights</h4><p>{num_delayed}</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown(
        f"<div class='card'><h4>Delay Percentage</h4><p>{delayed_percentage:.1f}%</p></div>", unsafe_allow_html=True)
with col4:
    st.markdown(
        f"<div class='card'><h4>Avg Dep Delay (min)</h4><p>{avg_dep_delay:.1f}</p></div>", unsafe_allow_html=True)
with col5:
    st.markdown(
        f"<div class='card'><h4>Avg Arr Delay (min)</h4><p>{avg_arr_delay:.1f}</p></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 6) Top 5 Delayed Airports and Airlines (Horizontal Bars)
# --------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Top 5 Delayed Airports and Top 5 Delayed Airlines")

# Filter by date only (ignore selected airport filter) for these charts
df_date_filtered = df[(df["FILED OFF BLOCK TIME"].dt.date >= start_date) &
                      (df["FILED OFF BLOCK TIME"].dt.date <= end_date)].copy()

# --- Airports ---
# Now include City and Country from the first record per airport.
airport_delays = df_date_filtered.groupby("ADEP").agg(
    total_departures=("ADEP", "count"),
    delayed_flights=("DEPTDEL", lambda x: (x > delay_threshold).sum()),
    City=("City", "first"),
    Country=("Country", "first")
).reset_index()
airport_delays['Delay %'] = airport_delays['delayed_flights'] / \
    airport_delays['total_departures'] * 100
# Sort descending so that the most delayed is on top
top5_airports = airport_delays.sort_values(
    "delayed_flights", ascending=False).head(5)
top5_airports = top5_airports.rename(columns={"ADEP": "Airport"})

fig_airports = px.bar(
    top5_airports,
    x="delayed_flights",
    y="Airport",
    orientation='h',
    text="delayed_flights",
    labels={"delayed_flights": "Delayed Flights"},
    title="Top 5 Delayed Airports",
    color_discrete_sequence=["#8B0000"],  # darker red
    category_orders={"Airport": list(top5_airports["Airport"])}
)
# Bring text inside the bars, make it bold/white, and include City and Country in hover data.
fig_airports.update_traces(
    textposition='inside',
    textfont=dict(color='white', size=12, family='Arial-Bold'),
    width=0.5
)
fig_airports.update_layout(
    bargap=0.15, xaxis_title="Number of Delayed Flights", yaxis_title="Airport")
fig_airports.update_traces(
    hovertemplate="<b>%{y}</b><br>Delayed Flights: %{x}<br>City: %{customdata[0]}<br>Country: %{customdata[1]}")
fig_airports.update_traces(
    customdata=top5_airports[['City', 'Country']].values)

# --- Airlines ---
airline_delays = df_date_filtered.groupby("AC Operator").agg(
    total_departures=("AC Operator", "count"),
    delayed_flights=("DEPTDEL", lambda x: (x > delay_threshold).sum())
).reset_index()
airline_delays['Delay %'] = airline_delays['delayed_flights'] / \
    airline_delays['total_departures'] * 100
top5_airlines = airline_delays.sort_values(
    "delayed_flights", ascending=False).head(5)
top5_airlines = top5_airlines.rename(columns={"AC Operator": "Airline"})

fig_airlines = px.bar(
    top5_airlines,
    x="delayed_flights",
    y="Airline",
    orientation='h',
    text="delayed_flights",
    labels={"delayed_flights": "Delayed Flights"},
    title="Top 5 Delayed Airlines",
    color_discrete_sequence=["#FFA500"],  # orange
    category_orders={"Airline": list(top5_airlines["Airline"])}
)
fig_airlines.update_traces(
    textposition='inside',
    textfont=dict(color='white', size=12, family='Arial-Bold'),
    width=0.5
)
fig_airlines.update_layout(
    bargap=0.15, xaxis_title="Number of Delayed Flights", yaxis_title="Airline")

col_bar1, col_bar2 = st.columns(2)
with col_bar1:
    st.plotly_chart(fig_airports, use_container_width=True)
with col_bar2:
    st.plotly_chart(fig_airlines, use_container_width=True)

# --------------------------------------------------------------------------------
# 7) Departure Delay Over Time Plot by Brackets (Altair)
# --------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Departure Delay Over Time by Delay Brackets")


def assign_delay_bracket(delay):
    if delay < 15:
        return "0-15"
    elif delay < 30:
        return "15-30"
    elif delay < 90:
        return "30-90"
    else:
        return "90+"


df_plot = df_filtered.copy()
df_plot['Delay Bracket'] = df_plot['DEPTDEL'].apply(assign_delay_bracket)
df_plot['TimeBin'] = df_plot["FILED OFF BLOCK TIME"].dt.floor("H")
time_bracket = df_plot.groupby(
    ["TimeBin", "Delay Bracket"]).size().reset_index(name="Count")

# Use 'basis' interpolation to soften the curves
chart = alt.Chart(time_bracket).mark_line(point=True, interpolate='basis').encode(
    x=alt.X("TimeBin:T", title="Time"),
    y=alt.Y("Count:Q", title="Number of Flights"),
    color=alt.Color("Delay Bracket:N", title="Delay Bracket"),
    tooltip=["TimeBin:T", "Delay Bracket:N", "Count:Q"]
).properties(
    width=800,
    height=400
).interactive()

st.altair_chart(chart, use_container_width=True)

# --------------------------------------------------------------------------------
# 8) Scatter Plot: Total Departures vs. Delayed Flights by Airport
# --------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Scatter Plot: Departures vs. Delayed Flights by Airport")

scatter_data = airport_delays.copy().rename(columns={"ADEP": "Airport"})
scatter_data = scatter_data.sort_values(
    "total_departures", ascending=False).head(n_display)

fig_scatter = px.scatter(
    scatter_data,
    x="total_departures",
    y="delayed_flights",
    text="Airport",
    labels={"total_departures": "Total Departures",
            "delayed_flights": "Delayed Flights"},
    title="Airport Performance: Departures vs. Delayed Flights",
    template="plotly_white",
    hover_data=["City", "Country"]
)
fig_scatter.update_traces(textposition='top center')
st.plotly_chart(fig_scatter, use_container_width=True)
