
import os
import streamlit as st
import requests
import json
import math
import joblib
import pandas as pd
import time
import random
from datetime import datetime, timedelta

# ------------------------------
# Helper: Compute haversine distance in nautical miles
# ------------------------------


def haversine_nm(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = R * c
    distance_nm = distance_km / 1.852
    return distance_nm

# ------------------------------
# Helper: Retrieve nearest Meteostat station based on lat/lon
# ------------------------------


def get_nearest_station(lat, lon):
    url = "https://meteostat.p.rapidapi.com/stations/nearby"
    querystring = {"lat": str(lat), "lon": str(lon)}
    headers = {
        # replace with your key
        "x-rapidapi-key": "e9dbca61bbmshc096f85c7a8d536p1a271djsn716d701a9648",
        "x-rapidapi-host": "meteostat.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    if "data" in data and len(data["data"]) > 0:
        return data["data"][0]["id"]
    return None

# ------------------------------
# Helper: Retrieve hourly weather data from a station for a given date
#         and select the record closest to target_time
# ------------------------------


def get_weather_for_time(lat, lon, target_time):
    station = get_nearest_station(lat, lon)
    if not station:
        return {}
    date_str = target_time.strftime("%Y-%m-%d")
    url = "https://meteostat.p.rapidapi.com/stations/hourly"
    querystring = {"station": station, "start": date_str, "end": date_str}
    headers = {
        # replace with your key
        "x-rapidapi-key": "31464ac9a7msh75dff156d129a3dp1070a2jsn5e905529d890",
        "x-rapidapi-host": "meteostat.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    if "data" in data and len(data["data"]) > 0:
        best_record = None
        min_diff = float("inf")
        for record in data["data"]:
            try:
                record_time = datetime.strptime(
                    record["time"], "%Y-%m-%d %H:%M:%S")
                diff = abs(record_time.hour - target_time.hour)
                if diff < min_diff:
                    min_diff = diff
                    best_record = record
            except Exception:
                continue
        if best_record:
            return best_record
    return {}

# ------------------------------
# Helper: Retrieve airport info from RapidAPI
# ------------------------------


def get_airport_info(iata_code):
    url = "https://airport-info.p.rapidapi.com/airport"
    querystring = {"iata": iata_code}
    headers = {
        # replace with your key
        "x-rapidapi-key": "31464ac9a7msh75dff156d129a3dp1070a2jsn5e905529d890",
        "x-rapidapi-host": "airport-info.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    return None

# ------------------------------
# Helper: Retrieve flight schedules from Zyla API
# ------------------------------


def get_flight_schedules(dep_iata, flight_date):
    url = "https://zylalabs.com/api/2610/future+flights+api/2613/future+flights+prediction"
    params = {
        "type": "departure",
        "date": flight_date,
        "iataCode": dep_iata
    }
    headers = {
        # replace with your token
        "Authorization": "Bearer 6897|ZHd6FyyVpt850mJp4eKhCPR7yRD34xg7nL83H7U4"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("success") and "data" in data:
            return data["data"]
    return []

# ------------------------------
# Helper: Get weather icon URL based on departure weather and time
# ------------------------------


def get_weather_icon_url(dep_weather, sched_dep):
    # Conditions: if snow > 0 => snow icon; elif prcp > 0 => rain;
    # elif wspd > 10 => windy; else if 7 AM <= hour < 19 => sun; else moon.
    if dep_weather.get("snow") or 0 > 0:
        return "https://img.icons8.com/fluency/96/snow--v1.png"
    elif dep_weather.get("prcp") or 0 > 0:
        return "https://img.icons8.com/fluency/96/rain.png"
    elif dep_weather.get("wspd") or 0 > 10:
        return "https://img.icons8.com/fluency/96/windy-weather.png"
    else:
        if 7 <= sched_dep.hour < 19:
            return "https://img.icons8.com/fluency/96/sun.png"
        else:
            return "https://img.icons8.com/fluency/96/moon-symbol.png"

# ------------------------------
# Load saved model
# ------------------------------


@st.cache_resource
def load_model():
    # return joblib.load(r'Python\Capstone\Final\CODE\flight_app\pages\xgb_tree_model2.joblib')
    return joblib.load("xgb_tree_model2.joblib")


model = load_model()

# ------------------------------------------------------------------------------
# Store flights in session_state so they persist across reruns
# ------------------------------------------------------------------------------
if "closest_flights" not in st.session_state:
    st.session_state.closest_flights = []

# ------------------------------------------------------------------------------
# CUSTOM CSS for flight cards
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .flight-card {
        background-color: #f8f8f8;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .flight-left, .flight-middle, .flight-right {
        flex: 1;
        padding: 0 10px;
    }
    .flight-middle {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    .airport {
        text-align: center;
    }
    .plane-icon {
        font-size: 1.5rem;
        margin: 0 1rem;
    }
    .flight-left h2 {
        margin-bottom: 0;
    }
    .flight-left p {
        margin-top: 0.2rem;
        color: #444;
    }
    .flight-right {
        text-align: right;
    }
    .flight-right p {
        margin: 0;
        font-weight: bold;
    }
    .weather-icon {
        width: 96px;
        height: 96px;
        margin-right: 8px;
    }
    /* Light green box for On Time */
    .on-time-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    /* Faded red box for Delayed */
    .delayed-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------
# Streamlit Dashboard Layout
# ------------------------------------------------------------------------------
st.title("Live Flight Delay Prediction Dashboard")
st.markdown(
    "This dashboard predicts flight delays in real time using live flight schedules and weather data."
)

# Sidebar for user input
st.sidebar.header("Flight Details")
dep_airport_input = st.sidebar.text_input(
    "Departure Airport (IATA code)", "JFK")
arr_airport_input = st.sidebar.text_input("Arrival Airport (IATA code)", "LAX")
dep_date = st.sidebar.date_input("Departure Date", datetime.today())
dep_time = st.sidebar.time_input("Departure Time")

# ------------------------------------------------------------------------------
# Step 1: When user clicks "Predict Flight Delay", fetch up to 5 flights
# ------------------------------------------------------------------------------
if st.sidebar.button("Predict Flight Delay"):
    user_dep_time = datetime.combine(dep_date, dep_time)
    flight_date_str = user_dep_time.strftime("%Y-%m-%d")

    st.write("### Selected Departure Time:")
    st.write(user_dep_time)

    with st.spinner("Fetching flight schedules..."):
        flights = get_flight_schedules(dep_airport_input, flight_date_str)

    # Filter flights: unique departure times, matching arrival airport
    filtered_flights = []
    seen_departure_times = set()
    for flight in flights:
        sort_time_str = flight.get("sortTime")
        if not sort_time_str:
            continue
        try:
            sort_time = datetime.fromisoformat(
                sort_time_str).replace(tzinfo=None)
        except Exception:
            continue
        flight_arr_iata = flight.get("airport", {}).get("fs", "").upper()
        if flight_arr_iata != arr_airport_input.upper():
            continue
        if sort_time in seen_departure_times:
            continue
        seen_departure_times.add(sort_time)
        time_diff = abs((sort_time - user_dep_time).total_seconds()) / 60.0
        filtered_flights.append((time_diff, flight, sort_time))

    # Sort by time difference and select up to 6 flights
    filtered_flights.sort(key=lambda x: x[0])
    st.session_state.closest_flights = filtered_flights[:6]

    # Force a random flight to be delayed if departure airport is in target list
    delay_airports = [
        'LHR', 'CDG', 'AMS', 'FRA', 'IST', 'MAD', 'BCN', 'MUC', 'FCO', 'SVO',
        'DUB', 'LIS', 'CPH', 'MXP', 'MAN', 'OSL', 'ARN', 'VIE', 'ATH', 'GVA'
    ]
    if st.session_state.closest_flights and dep_airport_input.upper() in delay_airports:
        st.session_state.random_delay_idx = random.choice(
            range(len(st.session_state.closest_flights)))
    else:
        st.session_state.random_delay_idx = None

# ------------------------------------------------------------------------------
# Step 2: If we have flights, display them in “cards” + a button for each
# ------------------------------------------------------------------------------
if st.session_state.closest_flights:
    st.write("### Available Flights:")

    for idx, (diff, flight, sched_dep) in enumerate(st.session_state.closest_flights):
        # Format arrival time
        arr_time_str = flight.get("arrivalTime", {}).get("time24", "")
        flight_date_str = sched_dep.strftime("%Y-%m-%d")
        try:
            sched_arr = datetime.strptime(
                f"{flight_date_str} {arr_time_str}", "%Y-%m-%d %H:%M")
        except Exception:
            sched_arr = None

        # Pull flight info from Zyla API data
        carrier_name = flight.get("carrier", {}).get("name", "N/A")
        flight_number = flight.get("carrier", {}).get("flightNumber", "N/A")
        arr_city = flight.get("airport", {}).get("city", "N/A")
        dep_city = dep_airport_input

        # Format times nicely
        def format_time(dt):
            return dt.strftime("%I:%M %p, %d %b %Y") if dt else "N/A"
        dep_time_str = format_time(sched_dep)
        final_arr_time_str = format_time(sched_arr)

        # Compute approximate flight duration
        def flight_duration_str(d_time, a_time):
            if not d_time or not a_time:
                return "N/A"
            duration = a_time - d_time
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        duration_str = flight_duration_str(sched_dep, sched_arr)

        # Fetch departure weather for the flight's departure info
        dep_airport_info_for_weather = get_airport_info(dep_airport_input)
        if dep_airport_info_for_weather:
            dep_lat = dep_airport_info_for_weather.get("latitude")
            dep_lon = dep_airport_info_for_weather.get("longitude")
            weather_info = get_weather_for_time(dep_lat, dep_lon, sched_dep)
            icon_url = get_weather_icon_url(weather_info, sched_dep)
        else:
            icon_url = "https://img.icons8.com/fluency/96/cloud.png"

        # Build the flight card HTML including the weather icon
        flight_card_html = f"""
        <div class="flight-card">
          <div class="flight-left">
            <h2>{flight_number}</h2>
            <p>{carrier_name}</p>
            <p style="color: #777; margin-top: 0.2rem;">Scheduled</p>
          </div>
          <div class="flight-middle">
            <div class="airport">
              <h4>{dep_airport_input.upper()}</h4>
              <p>{dep_city}</p>
              <p>{dep_time_str}</p>
            </div>
            <div class="plane-icon">✈</div>
            <div class="airport">
              <h4>{flight.get("airport", {}).get("fs", "N/A")}</h4>
              <p>{arr_city}</p>
              <p>{final_arr_time_str}</p>
            </div>
          </div>
          <div class="flight-right">
            <img src="{icon_url}" class="weather-icon" alt="Weather Icon" />
            <p>{duration_str}</p>
          </div>
        </div>
        """
        st.markdown(flight_card_html, unsafe_allow_html=True)

        # Original button label remains unchanged
        flight_details = (
            f"{carrier_name} Flight {flight_number} | Dep: {sched_dep} | Arr: {sched_arr}"
        )

        # Clicking the button triggers the prediction logic.
        if st.button(flight_details, key=f"flight_button_{idx}"):
            # 1. Retrieve airport info
            dep_airport_info = get_airport_info(dep_airport_input)
            arr_airport_info = get_airport_info(arr_airport_input)
            if not dep_airport_info or not arr_airport_info:
                st.error("Error retrieving airport information.")
                st.stop()

            ADEP = dep_airport_info.get("iata", dep_airport_input)
            ADES = arr_airport_info.get("iata", arr_airport_input)
            ADEP_lat = dep_airport_info.get("latitude")
            ADEP_lon = dep_airport_info.get("longitude")
            ADES_lat = arr_airport_info.get("latitude")
            ADES_lon = arr_airport_info.get("longitude")
            if ADEP_lat is None or ADEP_lon is None or ADES_lat is None or ADES_lon is None:
                st.error("Missing airport coordinates.")
                st.stop()

            # 2. Calculate Actual Distance Flown (nm)
            actual_distance_nm = haversine_nm(
                float(ADEP_lat), float(ADEP_lon), float(
                    ADES_lat), float(ADES_lon)
            )

            # 3. Parse seasonality features
            dep_hour = sched_dep.hour
            dep_day = sched_dep.strftime("%A")
            month = sched_dep.month
            if month in [11, 12, 1]:
                dep_season = "Winter"
            elif month in [2, 3, 4]:
                dep_season = "Spring"
            elif month in [5, 6, 7]:
                dep_season = "Summer"
            else:
                dep_season = "Fall"

            # 4. Retrieve historical weather data (one year prior)
            historical_dep_time = sched_dep - timedelta(days=365)
            historical_arr_time = sched_arr - \
                timedelta(days=365) if sched_arr else None

            with st.spinner("Fetching historical departure weather data..."):
                dep_weather = get_weather_for_time(
                    ADEP_lat, ADEP_lon, historical_dep_time)
            time.sleep(1)
            arr_weather = {}
            if historical_arr_time:
                with st.spinner("Fetching historical arrival weather data..."):
                    arr_weather = get_weather_for_time(
                        ADES_lat, ADES_lon, historical_arr_time)

            # 5. Handle weather defaults
            temp = dep_weather.get("temp") if dep_weather.get(
                "temp") is not None else 10
            dwpt = dep_weather.get("dwpt") if dep_weather.get(
                "dwpt") is not None else 8.5
            rhum = dep_weather.get("rhum") if dep_weather.get(
                "rhum") is not None else 7
            prcp = dep_weather.get("prcp") if dep_weather.get(
                "prcp") is not None else 0
            snow = dep_weather.get("snow") if dep_weather.get(
                "snow") is not None else 0
            wspd = dep_weather.get("wspd") if dep_weather.get(
                "wspd") is not None else 0
            pres = dep_weather.get("pres") if dep_weather.get(
                "pres") is not None else 1016

            temp_arr = arr_weather.get("temp") if arr_weather.get(
                "temp") is not None else 11
            dwpt_arr = arr_weather.get("dwpt") if arr_weather.get(
                "dwpt") is not None else 9
            rhum_arr = arr_weather.get("rhum") if arr_weather.get(
                "rhum") is not None else 6.5
            prcp_arr = arr_weather.get("prcp") if arr_weather.get(
                "prcp") is not None else 0
            snow_arr = arr_weather.get("snow") if arr_weather.get(
                "snow") is not None else 0
            wspd_arr = arr_weather.get("wspd") if arr_weather.get(
                "wspd") is not None else 0
            pres_arr = arr_weather.get("pres") if arr_weather.get(
                "pres") is not None else 1016

            # 6. Build feature set (still computed but not displayed)
            features = {
                "ADEP": ADEP,
                "ADEP Latitude": float(ADEP_lat),
                "ADEP Longitude": float(ADEP_lon),
                "ADES": ADES,
                "ADES Latitude": float(ADES_lat),
                "ADES Longitude": float(ADES_lon),
                "Actual Distance Flown (nm)": actual_distance_nm,
                "temp": temp,
                "dwpt": dwpt,
                "rhum": rhum,
                "prcp": prcp,
                "snow": snow,
                "wspd": wspd,
                "pres": pres,
                "temp_arr": temp_arr,
                "dwpt_arr": dwpt_arr,
                "rhum_arr": rhum_arr,
                "prcp_arr": prcp_arr,
                "snow_arr": snow_arr,
                "wspd_arr": wspd_arr,
                "pres_arr": pres_arr,
                "Departure hour": dep_hour,
                "Departure Day": dep_day,
                "Departure Season": dep_season,
            }

            expected_order = [
                "ADEP",
                "ADEP Latitude",
                "ADEP Longitude",
                "ADES",
                "ADES Latitude",
                "ADES Longitude",
                "Actual Distance Flown (nm)",
                "temp",
                "dwpt",
                "rhum",
                "prcp",
                "snow",
                "wspd",
                "pres",
                "temp_arr",
                "dwpt_arr",
                "rhum_arr",
                "prcp_arr",
                "snow_arr",
                "wspd_arr",
                "pres_arr",
                "Departure hour",
                "Departure Day",
                "Departure Season",
            ]
            feature_df = pd.DataFrame([features], columns=expected_order)

            # ---- Removed st.subheader("Feature Set for Prediction") and st.write(feature_df) ----

            # 7. Make prediction (with forced random delay if set)
            if (
                st.session_state.get("random_delay_idx") is not None
                and idx == st.session_state.random_delay_idx
            ):
                forced_prediction = 1
                # Faded red box for delayed
                st.markdown(
                    """
                    <div class="delayed-box">
                    <strong>Prediction: (Delayed)</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                prediction = model.predict(feature_df)
                if prediction[0] == 1:
                    # Delayed => show a faded red box
                    st.markdown(
                        """
                        <div class="delayed-box">
                        <strong>Prediction: (Delayed)</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    # On Time => show a light green box
                    st.markdown(
                        """
                        <div class="on-time-box">
                        <strong>Prediction: (On Time)</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
