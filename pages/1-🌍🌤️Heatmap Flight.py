import streamlit as st
import pandas as pd
import folium
from matplotlib.colors import to_hex
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import plotly.graph_objects as go
import os
import glob


# --------------------------------------------------------------------------------
# 1) Page Setup
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide")  # Use full screen width

# Optional: Inject custom CSS to adjust container padding.
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
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
    for col in ["FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME", "FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME"]:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    df['DEPTDEL'] = (df["ACTUAL OFF BLOCK TIME"] -
                     df["FILED OFF BLOCK TIME"]).dt.total_seconds() / 60
    df['ARVLDEL'] = (df["ACTUAL ARRIVAL TIME"] -
                     df["FILED ARRIVAL TIME"]).dt.total_seconds() / 60
    # NOTE: The cleaning section has been removed since the data is already pre-cleaned.
    return df


df = load_data()

# --------------------------------------------------------------------------------
# 3) Sidebar Controls
# --------------------------------------------------------------------------------
st.sidebar.title("Filters")
selected_date = st.sidebar.date_input(
    "Select Date", value=pd.Timestamp("2018-12-30"))
threshold_minutes = st.sidebar.slider(
    "Delay Threshold (minutes)", min_value=0, max_value=120, value=15, step=5)
# Removed slider for Minimum Flights per Day; using constant value instead.
min_flights_threshold = 20  # constant value
top_airports_option = st.sidebar.checkbox("Top Delayed Airports", value=True)
if top_airports_option:
    top_airports_count = st.sidebar.slider(
        "Number of Top Airports to Display", min_value=10, max_value=100, value=30, step=10)
else:
    top_airports_count = None
# New slider for number of rows in the top tables
top_table_rows = st.sidebar.slider(
    "Number Of Top Airports/Airlines", min_value=5, max_value=10, value=5, step=5)
# Optional airport filter (affects only the map)
selected_airports = st.sidebar.multiselect(
    "Select Airport(s)", options=sorted(df['ADEP'].unique()))

# --------------------------------------------------------------------------------
# 4) Map Generation Function
# --------------------------------------------------------------------------------


def plot_flight_delays(date, threshold, top_airports_count, min_flights_threshold, selected_airports=None):
    date = pd.to_datetime(date)
    filtered_df = df[(df["FILED OFF BLOCK TIME"] >= date) &
                     (df["FILED OFF BLOCK TIME"] < date + pd.Timedelta(days=1))].copy()
    if selected_airports:
        filtered_df = filtered_df[filtered_df['ADEP'].isin(selected_airports)]
    filtered_df['DelayedCalc'] = filtered_df['DEPTDEL'] > threshold

    # Include weather columns in the aggregation (using 'first' as aggregation method)
    airport_stats = (
        filtered_df.groupby('ADEP')
        .agg(
            delayed_flights=('DelayedCalc', 'sum'),
            total_departures=('ADEP', 'count'),
            latitude=('ADEP Latitude', 'first'),
            longitude=('ADEP Longitude', 'first'),
            city=('City', 'first'),
            prcp=('prcp', 'first'),
            tavg=('tavg', 'first'),
            snow=('snow', 'first')
        )
        .dropna(subset=['latitude', 'longitude'])
    )
    airport_stats = airport_stats[airport_stats['total_departures']
                                  >= min_flights_threshold]
    airport_stats = airport_stats[airport_stats['delayed_flights'] > 0]
    airport_stats = airport_stats.sort_values(
        'delayed_flights', ascending=False)
    if top_airports_count is not None:
        top_airports = airport_stats.head(top_airports_count).reset_index()
    else:
        top_airports = airport_stats.reset_index()

    flight_map = folium.Map(
        location=[38.8566, 0.3522], zoom_start=2.3, tiles=None)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=False,
        control=True,
    ).add_to(flight_map)

    colormap = mcolors.LinearSegmentedColormap.from_list(
        "custom_colormap", ["yellow", "red", "black"])
    norm = mcolors.Normalize(vmin=0, vmax=80)
    for _, airport in top_airports.iterrows():
        delay_rate = (airport['delayed_flights'] /
                      airport['total_departures']) * 100
        circle_color = to_hex(colormap(norm(delay_rate)))

        # Function to replace NaN with "-" and format values with units
        def safe_value(val, unit=""):
            if pd.isna(val):
                return "-"
            if unit:
                return f"{val:.1f} {unit}"
            return val

        # Determine weather icon based on weather conditions:
        weather_icon_html = ""
        if safe_value(airport['snow'], "") != "-" and airport['snow'] > 0:
            if safe_value(airport['prcp'], "") != "-" and airport['prcp'] > 0:
                # **Cloud with Bigger Snowflake & Rain**
                weather_icon_html = ("<img src='https://img.icons8.com/emoji/96/000000/cloud-with-snow-emoji.png' "
                                     "width='32' height='32'>")  # Set size to 32px
            else:
                # **Cloud with a Bigger Snowflake**
                weather_icon_html = ("<img src='https://img.icons8.com/emoji/96/000000/cloud-with-snow-emoji.png' "
                                     "width='32' height='32'>")  # Set size to 32px
        elif safe_value(airport['prcp'], "") != "-" and airport['prcp'] > 0:
            # **Rain Icon**
            weather_icon_html = ("<img src='https://img.icons8.com/emoji/96/000000/cloud-with-rain-emoji.png' "
                                 "width='32' height='32'>")  # Set size to 32px
        else:
            # **Sunny Icon (if no snow or rain)**
            weather_icon_html = ("<img src='https://img.icons8.com/emoji/96/000000/sun-emoji.png' "
                                 "width='32' height='32'>")  # Set size to 32px

        # Build the popup HTML with a separator line between flight info & weather data
        popup_html = (
            f"<table style='border: 1px solid black; border-collapse: collapse; font-size:12px;'>"
            f"<tr><th>ADEP</th><td>{safe_value(airport['ADEP'])}</td></tr>"
            f"<tr><th>City</th><td>{safe_value(airport['city'])}</td></tr>"
            f"<tr><th>Departures</th><td>{safe_value(airport['total_departures'])}</td></tr>"
            f"<tr><th>Delayed</th><td>{safe_value(airport['delayed_flights'])}</td></tr>"
            f"<tr><th>Delayed Rate</th><td>{safe_value(delay_rate, '%')}</td></tr>"

            # **Separator Line (No Extra Spacing)**
            f"<tr><td colspan='2' style='border-top: 1px solid black;'></td></tr>"

            f"<tr><th>PRCP</th><td>{safe_value(airport['prcp'], 'mm')}</td></tr>"
            f"<tr><th>TAVG</th><td>{safe_value(airport['tavg'], '°C')}</td></tr>"
            f"<tr><th>SNOW</th><td>{safe_value(airport['snow'], 'cm')}</td></tr>"
        )

        if weather_icon_html:
            popup_html += f"<tr><th>Weather</th><td>{weather_icon_html}</td></tr>"

        popup_html += "</table>"

        raw_radius = airport['delayed_flights'] / 2
        radius = max(9, min(raw_radius, 30))
        folium.CircleMarker(
            location=[airport['latitude'], airport['longitude']],
            radius=radius,
            color=circle_color,
            fill=True,
            fill_color=circle_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(flight_map)
    return flight_map


# --------------------------------------------------------------------------------
# 5) Filter Data for the Map (using selected airports)
# --------------------------------------------------------------------------------
date_filter = pd.to_datetime(selected_date)
subset_df = df[(df["FILED OFF BLOCK TIME"] >= date_filter) &
               (df["FILED OFF BLOCK TIME"] < date_filter + pd.Timedelta(days=1))].copy()
if selected_airports:
    subset_df = subset_df[subset_df['ADEP'].isin(selected_airports)]
subset_df['DelayedCalc'] = subset_df['DEPTDEL'] > threshold_minutes

# --------------------------------------------------------------------------------
# 6) Donut Chart (for Flight Status)
# --------------------------------------------------------------------------------
# Create a compact donut chart of constant size 150x150 pixels (figsize=(1.5,1.5))
delayed_count = subset_df['DelayedCalc'].sum()
total_count = len(subset_df)
on_time_count = total_count - delayed_count
fig_donut, ax_donut = plt.subplots(figsize=(.8, .8))

# Set transparent background
fig_donut.patch.set_alpha(0)  # Transparent figure background
ax_donut.set_facecolor("none")  # Transparent axes background

if total_count > 0:
    sizes = [on_time_count, delayed_count]
    # Draw donut without external labels; only center text will be displayed.
    wedges, _ = ax_donut.pie(sizes, startangle=90, wedgeprops=dict(width=0.3))
    ax_donut.set(aspect="equal")
    overall_rate = (delayed_count / total_count * 100)
    # Ensure text remains visible in dark mode
    text_color = "white" if st.get_option("theme.base") == "dark" else "black"

    ax_donut.text(0, 0, f"{overall_rate:.1f}%",
                  ha='center', va='center', fontsize=7, fontweight='bold', color=text_color)
    ax_donut.text(0, -0.30, "Delayed",
                  ha='center', va='center', fontsize=4, color=text_color)

else:
    ax_donut.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=10)

# --------------------------------------------------------------------------------
# 7) Prepare Top Airports & Top Airlines Tables (for the entire day, ignoring selected airports)
# --------------------------------------------------------------------------------
table_subset = df[(df["FILED OFF BLOCK TIME"] >= date_filter) &
                  (df["FILED OFF BLOCK TIME"] < date_filter + pd.Timedelta(days=1))].copy()
table_subset['DelayedCalc'] = table_subset['DEPTDEL'] > threshold_minutes

# Top Airports Table: Separate columns for Delayed Flights and Total Departures
airport_stats_chart = (
    table_subset.groupby('ADEP')
    .agg(
        delayed_flights=('DelayedCalc', 'sum'),
        total_departures=('ADEP', 'count')
    )
    .reset_index()
)
airport_stats_chart = airport_stats_chart.sort_values(
    'delayed_flights', ascending=False).head(top_table_rows)
airport_stats_chart['delayed_rate'] = (
    airport_stats_chart['delayed_flights'] / airport_stats_chart['total_departures'] * 100)
airport_stats_chart = airport_stats_chart[[
    'ADEP', 'delayed_flights', 'total_departures', 'delayed_rate']]

# Top Airlines Table: Separate columns for Delayed Flights and Total Departures
airline_stats_chart = (
    table_subset.groupby('AC Operator')
    .agg(
        delayed_flights=('DelayedCalc', 'sum'),
        total_departures=('AC Operator', 'count')
    )
    .reset_index()
)
airline_stats_chart = airline_stats_chart.sort_values(
    'delayed_flights', ascending=False).head(top_table_rows)
airline_stats_chart['delayed_rate'] = (
    airline_stats_chart['delayed_flights'] / airline_stats_chart['total_departures'] * 100)
airline_stats_chart = airline_stats_chart[[
    'AC Operator', 'delayed_flights', 'total_departures', 'delayed_rate']]

# Additional Table: Top Country & City with most delays
country_city_stats = (
    table_subset.groupby(['Country', 'City'])
    .agg(
        delayed_flights=('DelayedCalc', 'sum'),
        total_departures=('City', 'count')
    )
    .reset_index()
)
country_city_stats = country_city_stats.sort_values(
    'delayed_flights', ascending=False).head(top_table_rows)
country_city_stats['delayed_rate'] = (
    country_city_stats['delayed_flights'] / country_city_stats['total_departures'] * 100)
country_city_stats = country_city_stats[[
    'Country', 'City', 'delayed_flights', 'total_departures', 'delayed_rate']]

# --------------------------------------------------------------------------------
# 8) Create the Map
# --------------------------------------------------------------------------------
map_result = plot_flight_delays(str(selected_date), threshold_minutes,
                                top_airports_count, min_flights_threshold, selected_airports)

# --------------------------------------------------------------------------------
# 9) Layout
# --------------------------------------------------------------------------------
st.markdown("<h3 style='font-size:22px;'>Flight Delay Analysis Dashboard</h3>",
            unsafe_allow_html=True)

# Row 1: Donut Chart above the map (centered)
st.markdown("<h4 style='font-size:16px;'>Flight Status</h4>",
            unsafe_allow_html=True)
st.pyplot(fig_donut, use_container_width=False)

# Row 2: Two Columns for Map and Tables (Map on the left; Top Airports & Top Airlines on the right)
col_map, col_table = st.columns([0.56, 0.44])
with col_map:
    st.markdown("<h4 style='font-size:16px;'>Flight Delay Map</h4>",
                unsafe_allow_html=True)
    folium_static(map_result, width=550, height=450)

with col_table:
    st.markdown("<h4 style='font-size:16px;'>Top Airports and Top Airlines</h4>",
                unsafe_allow_html=True)

    # Top Airports Table
    st.markdown("**Top Airports**", unsafe_allow_html=True)
    st.dataframe(
        airport_stats_chart,
        hide_index=True,
        use_container_width=True,

        column_order=["ADEP", "delayed_flights",
                      "total_departures", "delayed_rate"],
        column_config={
            "ADEP": st.column_config.TextColumn("Airport"),
            "delayed_flights": st.column_config.ProgressColumn(
                "Delayed",
                help="Number of delayed flights",
                format="%d",
                min_value=0,
                max_value=int(airport_stats_chart["delayed_flights"].max()) if len(
                    airport_stats_chart) else 1,
            ),
            "total_departures": st.column_config.ProgressColumn(
                "Departures",
                help="Total flights",
                format="%d",
                min_value=0,
                max_value=int(airport_stats_chart["total_departures"].max()) if len(
                    airport_stats_chart) else 1,
            ),
            "delayed_rate": st.column_config.ProgressColumn(
                "Rate(%)",
                help="Delayed Rate (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
            ),
        },
        # width=200,
    )

    # Top Airlines Table
    st.markdown("**Top Airlines**", unsafe_allow_html=True)
    st.dataframe(
        airline_stats_chart,
        hide_index=True,
        use_container_width=True,
        column_order=["AC Operator", "delayed_flights",
                      "total_departures", "delayed_rate"],
        column_config={
            "AC Operator": st.column_config.TextColumn("Airline"),
            "delayed_flights": st.column_config.ProgressColumn(
                "Delayed",
                help="Number of delayed flights",
                format="%d",
                min_value=0,
                max_value=int(airline_stats_chart["delayed_flights"].max()) if len(
                    airline_stats_chart) else 1,
            ),
            "total_departures": st.column_config.ProgressColumn(
                "Departures",
                help="Total flights",
                format="%d",
                min_value=0,
                max_value=int(airline_stats_chart["total_departures"].max()) if len(
                    airline_stats_chart) else 1,
            ),
            "delayed_rate": st.column_config.ProgressColumn(
                "Rate(%)",
                help="Delayed Rate (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
            ),
        },
        width=300,
    )

# New Row: Full-width container for Top Country & City Table (placed underneath the two-column layout)
with st.container():
    st.markdown("<h4 style='font-size:16px;'>Top Country & City</h4>",
                unsafe_allow_html=True)
    st.dataframe(
        country_city_stats,
        hide_index=True,
        use_container_width=True,
        column_order=["Country", "City", "delayed_flights",
                      "total_departures", "delayed_rate"],
        column_config={
            "Country": st.column_config.TextColumn("Country"),
            "City": st.column_config.TextColumn("City"),
            "delayed_flights": st.column_config.ProgressColumn(
                "Delayed",
                help="Number of delayed flights",
                format="%d",
                min_value=0,
                max_value=int(country_city_stats["delayed_flights"].max()) if len(
                    country_city_stats) else 1,
            ),
            "total_departures": st.column_config.ProgressColumn(
                "Departures",
                help="Total flights",
                format="%d",
                min_value=0,
                max_value=int(country_city_stats["total_departures"].max()) if len(
                    country_city_stats) else 1,
            ),
            "delayed_rate": st.column_config.ProgressColumn(
                "Rate(%)",
                help="Delayed Rate (%)",
                format="%.1f",
                min_value=0.0,
                max_value=100.0,
            ),
        },
        width=300,
    )

# --------------------------------------------------------------------------------
# 10) Instructions
# --------------------------------------------------------------------------------
st.markdown("""
**Instructions:**
- Use the **sidebar** to select the date, delay threshold, and (via the constant) minimum flights per day (20-100), and optionally specific airport(s) (this filter affects only the map).
- Use the **Number Of Top Airports/Airlines** slider to set how many rows (5 to 10) appear in the Top Airports, Top Airlines, and Top Country & City tables (these tables are generated from the entire day's data).
- The **donut chart** (in the top row) shows the overall delayed percentage (centered only).
- The **map** (in the left column of the second row) shows airports with at least the specified number of departures and at least one delayed flight. The popup includes the city name.
- The **tables** (in the right column of the second row) display the Top Airports, Top Airlines, and Top Country & City with separate columns for "Delayed", "Departures", and "Rate(%)", sorted by the number of delayed flights.
""")
