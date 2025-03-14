#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import requests
import json
import math
import joblib
import pandas as pd
import time
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
        "x-rapidapi-key": "e9dbca61bbmshc096f85c7a8d536p1a271djsn716d701a9648",  # replace with your key
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
        "x-rapidapi-key": "e9dbca61bbmshc096f85c7a8d536p1a271djsn716d701a9648",  # replace with your key
        "x-rapidapi-host": "meteostat.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    if "data" in data and len(data["data"]) > 0:
        best_record = None
        min_diff = float("inf")
        for record in data["data"]:
            try:
                record_time = datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S")
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
        "x-rapidapi-key": "e9dbca61bbmshc096f85c7a8d536p1a271djsn716d701a9648",  # replace with your key
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
        "Authorization": "Bearer 6897|ZHd6FyyVpt850mJp4eKhCPR7yRD34xg7nL83H7U4"  # replace with your token
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("success") and "data" in data:
            return data["data"]
    return []

# ------------------------------
# Load saved model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_tree_model2.joblib")

model = load_model()

# ------------------------------------------------------------------------------
# Ensure session_state has a place to store flights so we don't lose them on reruns
# ------------------------------------------------------------------------------
if "closest_flights" not in st.session_state:
    st.session_state.closest_flights = []

# ------------------------------
# Streamlit Dashboard Layout
# ------------------------------
st.title("Live Flight Delay Prediction Dashboard")
st.markdown(
    "This dashboard predicts flight delays in real time using live flight schedules and historical weather data."
)

# Sidebar for user input
st.sidebar.header("Flight Details")
dep_airport_input = st.sidebar.text_input("Departure Airport (IATA code)", "JFK")
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
            sort_time = datetime.fromisoformat(sort_time_str).replace(tzinfo=None)
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

    # Sort by time difference and select up to 5 flights
    filtered_flights.sort(key=lambda x: x[0])
    st.session_state.closest_flights = filtered_flights[:5]  # store in session_state

# ------------------------------------------------------------------------------
# Step 2: If we have flights, display them as buttons. Clicking a button
#         immediately performs the prediction logic (no rerun needed).
# ------------------------------------------------------------------------------
if st.session_state.closest_flights:
    st.write("### Available Flights:")
    # Show each flight as a separate button
    for idx, (diff, flight, sched_dep) in enumerate(st.session_state.closest_flights):
        # Format flight details
        arr_time_str = flight.get("arrivalTime", {}).get("time24", "")
        flight_date_str = sched_dep.strftime("%Y-%m-%d")
        try:
            sched_arr = datetime.strptime(f"{flight_date_str} {arr_time_str}", "%Y-%m-%d %H:%M")
        except Exception:
            sched_arr = None
        carrier_name = flight.get("carrier", {}).get("name", "N/A")
        flight_number = flight.get("carrier", {}).get("flightNumber", "N/A")
        flight_details = (
            f"{carrier_name} Flight {flight_number} | Dep: {sched_dep} | Arr: {sched_arr}"
        )

        # ----------------------------------------------------------------------------
        # When you click the button, we do the entire prediction for that flight
        # ----------------------------------------------------------------------------
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
                float(ADEP_lat), float(ADEP_lon), float(ADES_lat), float(ADES_lon)
            )

            # 3. Parse seasonality features from the selected departure time
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
            historical_arr_time = sched_arr - timedelta(days=365) if sched_arr else None

            with st.spinner("Fetching historical departure weather data..."):
                dep_weather = get_weather_for_time(ADEP_lat, ADEP_lon, historical_dep_time)
            time.sleep(1)
            arr_weather = {}
            if historical_arr_time:
                with st.spinner("Fetching historical arrival weather data..."):
                    arr_weather = get_weather_for_time(ADES_lat, ADES_lon, historical_arr_time)

            # 5. Handle weather defaults
            temp = dep_weather.get("temp") if dep_weather.get("temp") is not None else 10
            dwpt = dep_weather.get("dwpt") if dep_weather.get("dwpt") is not None else 8.5
            rhum = dep_weather.get("rhum") if dep_weather.get("rhum") is not None else 7
            prcp = dep_weather.get("prcp") if dep_weather.get("prcp") is not None else 0
            snow = dep_weather.get("snow") if dep_weather.get("snow") is not None else 0
            wspd = dep_weather.get("wspd") if dep_weather.get("wspd") is not None else 0
            pres = dep_weather.get("pres") if dep_weather.get("pres") is not None else 1016

            temp_arr = arr_weather.get("temp") if arr_weather.get("temp") is not None else 11
            dwpt_arr = arr_weather.get("dwpt") if arr_weather.get("dwpt") is not None else 9
            rhum_arr = arr_weather.get("rhum") if arr_weather.get("rhum") is not None else 6.5
            prcp_arr = arr_weather.get("prcp") if arr_weather.get("prcp") is not None else 0
            snow_arr = arr_weather.get("snow") if arr_weather.get("snow") is not None else 0
            wspd_arr = arr_weather.get("wspd") if arr_weather.get("wspd") is not None else 0
            pres_arr = arr_weather.get("pres") if arr_weather.get("pres") is not None else 1016

            # 6. Build feature set
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
            st.subheader("Feature Set for Prediction")
            st.write(feature_df)

            # 7. Make prediction
            prediction = model.predict(feature_df)
            st.success(f"Prediction (0: On Time, 1: Delayed): {prediction[0]}")

            # We won't break here, so you can click multiple flights in one run


# In[ ]:




