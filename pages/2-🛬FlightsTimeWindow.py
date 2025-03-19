import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import altair as alt
import folium
from folium.plugins import TimestampedGeoJson
from streamlit_folium import st_folium
import numpy as np

# --------------------------------------------------------------------------------
# 1) Page Setup
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stDataFrame {
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
        df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")

    df["DEPTDEL"] = (df["ACTUAL OFF BLOCK TIME"] -
                     df["FILED OFF BLOCK TIME"]).dt.total_seconds() / 60
    df["ARVLDEL"] = (df["ACTUAL ARRIVAL TIME"] -
                     df["FILED ARRIVAL TIME"]).dt.total_seconds() / 60

    # **Round Delay Values to 1 Decimal Place**
    df["DEPTDEL"] = df["DEPTDEL"].round(1)
    df["ARVLDEL"] = df["ARVLDEL"].round(1)

    return df


df = load_data()

# --------------------------------------------------------------------------------
# 3) Sidebar Controls with Updated Airport Selection
# --------------------------------------------------------------------------------
st.sidebar.title("Filters")

# Create a DataFrame of unique airports with their corresponding cities.
airport_options = df[["ADEP", "City"]].drop_duplicates().sort_values("ADEP")
airport_options["option"] = airport_options["ADEP"] + \
    " (" + airport_options["City"] + ")"
airport_display_options = airport_options["option"].tolist()

default_airport = "LFPG"
if default_airport in airport_options["ADEP"].values:
    default_option = airport_options[airport_options["ADEP"]
                                     == default_airport]["option"].values[0]
else:
    default_option = airport_display_options[0]

selected_option = st.sidebar.selectbox(
    "Select Airport (ADEP)", airport_display_options, index=airport_display_options.index(default_option))
selected_airport = airport_options[airport_options["option"]
                                   == selected_option]["ADEP"].iloc[0]

selected_date = st.sidebar.date_input(
    "Select Date", value=pd.Timestamp("2018-12-30"))
# Set default time to 19:00 (7:00 PM) instead of auto-updating to now.
selected_time = st.sidebar.time_input("Select Time", value=datetime.strptime(
    "19:00", "%H:%M").time(), key="selected_time_input")

dep_interval_hours = st.sidebar.selectbox(
    "Select Departure Interval (hours)", options=[2, 4], index=0)
prev_interval_hours = st.sidebar.selectbox(
    "Select Previous Flights Interval (hours)", options=[12, 24, 48], index=1)
delay_threshold = st.sidebar.slider(
    "Delay Threshold (minutes)", min_value=0, max_value=120, value=15, step=5)

# --------------------------------------------------------------------------------
# 4) Define Time Windows & Cache Filtered Data
# --------------------------------------------------------------------------------
# Initialize session state keys if they do not exist.
if "selected_airport" not in st.session_state:
    st.session_state.selected_airport = selected_airport
    st.session_state.selected_date = selected_date
    st.session_state.selected_time = selected_time
    st.session_state.dep_interval_hours = dep_interval_hours
    st.session_state.prev_interval_hours = prev_interval_hours
    st.session_state.delay_threshold = delay_threshold
    st.session_state.filtered_df = None
    st.session_state.delayed_flights = None

# Check if any filter has changed or the filtered data is not initialized.
if (
    selected_airport != st.session_state.selected_airport
    or selected_date != st.session_state.selected_date
    or selected_time != st.session_state.selected_time
    or dep_interval_hours != st.session_state.dep_interval_hours
    or prev_interval_hours != st.session_state.prev_interval_hours
    or delay_threshold != st.session_state.delay_threshold
    or st.session_state.filtered_df is None
    or st.session_state.delayed_flights is None
):
    st.session_state.selected_airport = selected_airport
    st.session_state.selected_date = selected_date
    st.session_state.selected_time = selected_time
    st.session_state.dep_interval_hours = dep_interval_hours
    st.session_state.prev_interval_hours = prev_interval_hours
    st.session_state.delay_threshold = delay_threshold

    end_time = datetime.combine(selected_date, selected_time)
    start_time_dep = end_time - timedelta(hours=dep_interval_hours)
    start_time_prev = end_time - timedelta(hours=prev_interval_hours)

    filtered_df = df[df["ADEP"] == selected_airport]
    filtered_df = filtered_df[
        (filtered_df["ACTUAL OFF BLOCK TIME"] >= start_time_dep)
        & (filtered_df["ACTUAL OFF BLOCK TIME"] <= end_time)
    ]
    delayed_flights = filtered_df[filtered_df["DEPTDEL"] > delay_threshold]

    st.session_state.filtered_df = filtered_df
    st.session_state.delayed_flights = delayed_flights
else:
    end_time = datetime.combine(
        st.session_state.selected_date, st.session_state.selected_time)
    start_time_dep = end_time - \
        timedelta(hours=st.session_state.dep_interval_hours)
    start_time_prev = end_time - \
        timedelta(hours=st.session_state.prev_interval_hours)
    filtered_df = st.session_state.filtered_df
    delayed_flights = st.session_state.delayed_flights

# Display the Departure Delay Time Window header (with time on the next line)
st.markdown(
    f"<h4 style='font-size:20px;'>Departure Delay Time Window:<br>{start_time_dep} to {end_time}</h4>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# 5) Find Delayed Departures from Selected Airport
# --------------------------------------------------------------------------------
st.markdown(
    f"<h4 style='font-size:20px;'>Flights with Departure Delay > {delay_threshold} min at {selected_airport}</h4>",
    unsafe_allow_html=True,
)
if delayed_flights.empty:
    st.warning("No delayed flights found in the selected time window.")
else:
    st.dataframe(
        delayed_flights[
            [
                "ECTRL ID",
                "ADEP",
                "ADES",
                "FILED OFF BLOCK TIME",
                "ACTUAL OFF BLOCK TIME",
                "DEPTDEL",
                "AC Registration",
            ]
        ],
        hide_index=True,
        column_config={
            "DEPTDEL": st.column_config.ProgressColumn(
                "Departure Delay (min)",
                help="Departure delay in minutes",
                format="%d",
                min_value=0,
                max_value=int(delayed_flights["DEPTDEL"].max()) if len(
                    delayed_flights) else 1,
            )
        },
    )

# --------------------------------------------------------------------------------
# 6) Show Previous Flights of Delayed Aircraft to Selected Airport with Arrival Delay
# --------------------------------------------------------------------------------
if not delayed_flights.empty:
    tail_numbers = delayed_flights["AC Registration"].unique()
    last_departure_time = delayed_flights["ACTUAL OFF BLOCK TIME"].max()

    previous_flights = df[
        (df["AC Registration"].isin(tail_numbers))
        & (df["ADES"] == selected_airport)
        & (df["ACTUAL ARRIVAL TIME"] >= start_time_prev)
        & (df["ACTUAL ARRIVAL TIME"] <= last_departure_time)
    ]

    delayed_previous_flights = previous_flights[previous_flights["ARVLDEL"] > 15]
    delayed_previous_flights = delayed_previous_flights.sort_values(
        by="ACTUAL ARRIVAL TIME", ascending=False)

    st.markdown(
        f"<h4 style='font-size:20px;'>Previous Flights of Delayed Aircraft to {selected_airport} (Arrival Delay > 15 min)</h4>",
        unsafe_allow_html=True,
    )
    if delayed_previous_flights.empty:
        st.warning(
            "No previous delayed flights found for these aircraft to the selected airport.")
    else:
        st.dataframe(
            delayed_previous_flights[
                [
                    "ECTRL ID",
                    "ADEP",
                    "ADES",
                    "FILED ARRIVAL TIME",
                    "ACTUAL ARRIVAL TIME",
                    "ARVLDEL",
                    "AC Registration",
                ]
            ],
            hide_index=True,
            column_config={
                "ARVLDEL": st.column_config.ProgressColumn(
                    "Arrival Delay (min)",
                    help="Arrival delay in minutes",
                    format="%d",
                    min_value=0,
                    max_value=int(delayed_previous_flights["ARVLDEL"].max()) if len(
                        delayed_previous_flights) else 1,
                )
            },
        )
        selected_ac = st.selectbox(
            "Select Aircraft Registration", delayed_previous_flights["AC Registration"].unique())

        # --------------------------------------------------------------------------------
        # 7) Show Route History of Selected Aircraft
        # --------------------------------------------------------------------------------
        if selected_ac:
            route_history = df[
                (df["AC Registration"] == selected_ac)
                & (df["ACTUAL OFF BLOCK TIME"] >= start_time_prev)
                & (df["ACTUAL OFF BLOCK TIME"] <= last_departure_time)
            ]
            route_history = route_history.sort_values(
                by="ACTUAL OFF BLOCK TIME", ascending=False)

            st.markdown(
                f"<h4 style='font-size:20px;'>Route History of {selected_ac} (From {start_time_prev} to {last_departure_time})</h4>",
                unsafe_allow_html=True,
            )
            if route_history.empty:
                st.warning(
                    "No previous route history found for this aircraft within the selected interval.")
            else:
                st.dataframe(
                    route_history[
                        [
                            "ECTRL ID",
                            "ADEP",
                            "ADES",
                            "ACTUAL OFF BLOCK TIME",
                            "FILED ARRIVAL TIME",
                            "ARVLDEL",
                            "DEPTDEL",
                            "AC Registration",
                        ]
                    ],
                    hide_index=True,
                    column_config={
                        "DEPTDEL": st.column_config.ProgressColumn(
                            "Departure Delay (min)",
                            help="Departure delay in minutes",
                            format="%d",
                            min_value=0,
                            max_value=int(route_history["DEPTDEL"].max()) if len(
                                route_history) else 1,
                        ),
                        "ARVLDEL": st.column_config.ProgressColumn(
                            "Arrival Delay (min)",
                            help="Arrival delay in minutes",
                            format="%d",
                            min_value=0,
                            max_value=int(route_history["ARVLDEL"].max()) if len(
                                route_history) else 1,
                        ),
                    },
                )

                # --------------------------------------------------------------------------------
                # 8) Gantt Chart for Route History with Enhanced Visualization
                # --------------------------------------------------------------------------------
                def plot_route_history_chart(route_history):
                    route_history["Flight_Label"] = route_history["ADEP"] + \
                        " → " + route_history["ADES"]
                    route_history["Departure_Status"] = route_history["DEPTDEL"].apply(
                        lambda x: "Delayed" if x > delay_threshold else "On Time"
                    )
                    # Generate gridlines over the entire route history time range.
                    grid_start = route_history["ACTUAL OFF BLOCK TIME"].min()
                    grid_end = route_history["ACTUAL ARRIVAL TIME"].max()
                    grid_df = pd.DataFrame(
                        {"x": pd.date_range(grid_start, grid_end, freq="30T")})
                    gridlines = alt.Chart(grid_df).mark_rule(
                        color="lightgray", strokeWidth=1).encode(x="x:T")

                    actual_bar = alt.Chart(route_history).mark_bar(size=15).encode(
                        x="ACTUAL OFF BLOCK TIME:T",
                        x2="ACTUAL ARRIVAL TIME:T",
                        y=alt.Y("Flight_Label:N", title="Flights"),
                        color=alt.condition(
                            alt.datum.Departure_Status == "Delayed", alt.value(
                                "orange"), alt.value("green")
                        ),
                        tooltip=[
                            "ADEP",
                            "ADES",
                            alt.Tooltip("ACTUAL OFF BLOCK TIME:T",
                                        title="ADEPT", format="%Y-%m-%d %H:%M"),
                            alt.Tooltip("ACTUAL ARRIVAL TIME:T",
                                        title="AARVLT", format="%Y-%m-%d %H:%M"),
                            "DEPTDEL",
                            "ARVLDEL",
                        ],
                    )

                    filed_line = alt.Chart(route_history).mark_rule(color="blue", strokeDash=[4, 2], strokeWidth=2).encode(
                        x="FILED OFF BLOCK TIME:T",
                        x2="FILED ARRIVAL TIME:T",
                        y=alt.Y("Flight_Label:N"),
                    )

                    dep_arrow = alt.Chart(route_history).mark_text(align="left", dx=3, fontSize=14, text="➔").encode(
                        x="ACTUAL OFF BLOCK TIME:T", y=alt.Y("Flight_Label:N")
                    )

                    arr_arrow = alt.Chart(route_history).mark_text(align="right", dx=-3, fontSize=14, text="➔").encode(
                        x="ACTUAL ARRIVAL TIME:T", y=alt.Y("Flight_Label:N")
                    )

                    dep_dot = alt.Chart(route_history).mark_point(filled=True, size=100).encode(
                        x="ACTUAL OFF BLOCK TIME:T", y=alt.Y("Flight_Label:N")
                    )

                    arr_dot = alt.Chart(route_history).mark_point(filled=True, size=100).encode(
                        x="ACTUAL ARRIVAL TIME:T", y=alt.Y("Flight_Label:N")
                    )

                    chart = (
                        gridlines
                        + actual_bar
                        + filed_line
                        + dep_arrow
                        + arr_arrow
                        + dep_dot
                        + arr_dot
                    ).properties(
                        title="Route History (Filed vs. Actual Departures & Arrivals)",
                        width=800,
                        height=400,
                    ).interactive()

                    return chart

                st.markdown(
                    "<h4 style='font-size:20px;'>Route Time Window</h4>", unsafe_allow_html=True)
                st.altair_chart(plot_route_history_chart(
                    route_history), use_container_width=True)

# --------------------------------------------------------------------------------
# 9) Map Visualization of Route History with Arcs (Static Version, Color-Coded Routes)
# --------------------------------------------------------------------------------
st.markdown("<h4 style='font-size:20px;'>Route History Map</h4>",
            unsafe_allow_html=True)


def create_arc(start, end, num_points=50, curvature=0.2):
    # start and end are tuples: (lat, lon)
    lat1, lon1 = start
    lat2, lon2 = end
    # Calculate midpoint
    mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    # Offset for control point (adjust curvature)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    control_lat = mid_lat + curvature * (-delta_lon)
    control_lon = mid_lon + curvature * delta_lat
    # Create points along the Bézier curve
    t_values = np.linspace(0, 1, num_points)
    arc_points = []
    for t in t_values:
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * \
            t * control_lat + t ** 2 * lat2
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * \
            t * control_lon + t ** 2 * lon2
        arc_points.append([lat, lon])
    return arc_points


if "route_history" in locals() and not route_history.empty:
    first_flight = route_history.iloc[0]
    m = folium.Map(location=[first_flight["ADEP Latitude"],
                   first_flight["ADEP Longitude"]], zoom_start=6)

    folium.TileLayer(
        tiles="http://mt0.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google Maps",
        name="Google Maps",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.LayerControl().add_to(m)

    for _, row in route_history.iterrows():
        start_coord = [row["ADEP Latitude"], row["ADEP Longitude"]]
        end_coord = [row["ADES Latitude"], row["ADES Longitude"]]
        arc_coords = create_arc(start_coord, end_coord, curvature=0.2)
        filed_departure = row["FILED OFF BLOCK TIME"]
        actual_departure = row["ACTUAL OFF BLOCK TIME"]
        filed_arrival = row["FILED ARRIVAL TIME"]
        actual_arrival = row["FILED ARRIVAL TIME"]
        tooltip_text = (
            f"AC Operator: {row['AC Operator']}<br>"
            f"AC Type: {row['AC Type']}<br>"
            f"Departure: {row['ADEP']}<br>"
            f"FDEPT: {filed_departure.strftime('%Y-%m-%d %H:%M') if pd.notnull(filed_departure) else 'N/A'}<br>"
            f"ADEPT: {actual_departure.strftime('%Y-%m-%d %H:%M') if pd.notnull(actual_departure) else 'N/A'}<br>"
            f"Arrival: {row['ADES']}<br>"
            f"FARVT: {filed_arrival.strftime('%Y-%m-%d %H:%M') if pd.notnull(filed_arrival) else 'N/A'}<br>"
            f"AARVT: {actual_arrival.strftime('%Y-%m-%d %H:%M') if pd.notnull(actual_arrival) else 'N/A'}"
        )
        route_color = "orange" if row["DEPTDEL"] > delay_threshold else "green"
        folium.PolyLine(arc_coords, color="transparent", weight=10,
                        opacity=0, tooltip=tooltip_text).add_to(m)
        folium.PolyLine(arc_coords, color=route_color, weight=2,
                        opacity=0.7, tooltip=tooltip_text).add_to(m)

    nodes = pd.concat(
        [
            route_history[["ADEP", "ADEP Latitude", "ADEP Longitude"]].rename(
                columns={"ADEP": "Airport", "ADEP Latitude": "Latitude",
                         "ADEP Longitude": "Longitude"}
            ),
            route_history[["ADES", "ADES Latitude", "ADES Longitude"]].rename(
                columns={"ADES": "Airport", "ADES Latitude": "Latitude",
                         "ADES Longitude": "Longitude"}
            ),
        ]
    ).drop_duplicates(subset=["Airport"])

    for idx, row in nodes.iterrows():
        popup_text = f"{row['Airport']}"
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=popup_text,
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        ).add_to(m)

    st_folium(m, width=800, height=500)
else:
    st.warning("No route history available for map visualization.")
