#!/usr/bin/env python
# coding: utf-8

# In[29]:


import streamlit as st
import requests
import math
import pandas as pd
import random
from datetime import datetime, timedelta

# -----------------------------------------------------------
# Mapping Dictionaries for Carriers and Airports
# -----------------------------------------------------------
carrier_names = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "BA": "British Airways",
    "LH": "Lufthansa",
    "AF": "Air France",
    "EK": "Emirates",
    "QR": "Qatar Airways",
    "SQ": "Singapore Airlines",
    "KL": "KLM Royal Dutch Airlines",
    "AZ": "Alitalia",
    "JL": "Japan Airlines",
    "NH": "ANA",
    "VX": "Virgin America",
    "QF": "Qantas",
    "CX": "Cathay Pacific",
    "TK": "Turkish Airlines",
    "SA": "South African Airways",
    "MU": "China Eastern Airlines",
    "CA": "Air China",
    "AI": "Air India",
    "BR": "EVA Air",
    "NZ": "Air New Zealand",
    "GS": "China Southern Airlines",
    "SU": "Aeroflot Russian Airlines",
    "AM": "Aeromexico",
    "RY": "Ryanair",
    "FR": "Flybe",
    "LO": "LOT Polish Airlines",
    "SK": "SAS Scandinavian Airlines",
    "FI": "Icelandair",
    "TG": "Thai Airways",
    "LM": "Wizz Air",
    "9W": "Jet Airways",
    "9M": "Central Mountain Air",
    "IG": "Meridiana",
    "ST": "Smart Wings",
    "AC": "Air Canada",
    "X3": "Condor Airlines",
    "HV": "Transavia",
    "SC": "Shandong Airlines",
    "BD": "British Midland",
    "FY": "Firefly",
    "VS": "Virgin Atlantic Airways",
    "WN": "Southwest Airlines",
    "HL": "Hainan Airlines",
    "IR": "Iran Air",
    "HY": "Uzbekistan Airways",
    "OS": "Austrian Airlines",
    "PK": "Pakistan International Airlines",
    "VY": "Vueling",
    "TP": "TAP Air Portugal",
    "DE": "Condor",
    "ZB": "Monarch Airlines",
    "HG": "Travel Service",
    "RG": "Varig",
    "XJ": "Thai AirAsia",
    "AK": "AirAsia",
    "FD": "Thai AirAsia",
    "QZ": "Indonesia AirAsia",
    "ZZ": "AtlasGlobal",
    "TZ": "Scoot",
    "NO": "Neos",
    "SE": "Smartwings",
    "F9": "Frontier Airlines",
    "OK": "Czech Airlines",
    "PR": "Philippine Airlines",
    "SN": "Brussels Airlines",
    "CU": "Cuba Airlines",
    "KU": "Kuwait Airways",
    "MK": "Air Mauritius",
    "BG": "BG Airlines",
    "OY": "Yellowstone",
    "LF": "Lufthansa Regional",
    "XK": "Air Kosovo",
    "PE": "Peninsula Airways",
    "VZ": "VietJet Air",
    "VP": "Aerovias de Mexico",
    "WG": "Sunwing Airlines",
    "ZV": "Severstal Aircompany",
    "CM": "Copa Airlines",
    "BL": "Jetstar Pacific",
    "YQ": "Air Arabia",
    "JT": "Lion Air",
    "WS": "WestJet",
    "RU": "AirBridgeCargo Airlines"
}

airport_info_dict = {
    "ATL": "Hartsfield-Jackson Atlanta International Airport, Atlanta, USA",
"PEK": "Beijing Capital International Airport, Beijing, China",
"LAX": "Los Angeles International Airport, Los Angeles, USA",
"HND": "Tokyo Haneda Airport, Tokyo, Japan",
"ORD": "O'Hare International Airport, Chicago, USA",
"LHR": "London Heathrow Airport, London, UK",
"CDG": "Charles de Gaulle Airport, Paris, France",
"DFW": "Dallas/Fort Worth International Airport, Dallas/Fort Worth, USA",
"DXB": "Dubai International Airport, Dubai, UAE",
"FRA": "Frankfurt Airport, Frankfurt, Germany",
"JFK": "John F. Kennedy International Airport, New York City, USA",
"IST": "Istanbul Airport, Istanbul, Turkey",
"ICN": "Incheon International Airport, Seoul, South Korea",
"SGN": "Tan Son Nhat International Airport, Ho Chi Minh City, Vietnam",
"AMS": "Amsterdam Schiphol Airport, Amsterdam, Netherlands",
"DEL": "Indira Gandhi International Airport, Delhi, India",
"BKK": "Suvarnabhumi Airport, Bangkok, Thailand",
"SYD": "Sydney Kingsford Smith Airport, Sydney, Australia",
"CGK": "Soekarno-Hatta International Airport, Jakarta, Indonesia",
"SEA": "Seattle-Tacoma International Airport, Seattle, USA",
"BCN": "Barcelona El Prat Airport, Barcelona, Spain",
"LAS": "McCarran International Airport, Las Vegas, USA",
"KLIA": "Kuala Lumpur International Airport, Kuala Lumpur, Malaysia",
"YVR": "Vancouver International Airport, Vancouver, Canada",
"DOH": "Hamad International Airport, Doha, Qatar",
"ZRH": "Zurich Airport, Zurich, Switzerland",
"LGW": "London Gatwick Airport, London, UK",
"SHJ": "Sharjah International Airport, Sharjah, UAE",
"SAN": "San Diego International Airport, San Diego, USA",
"CTU": "Chengdu Shuangliu International Airport, Chengdu, China",
"AUH": "Abu Dhabi International Airport, Abu Dhabi, UAE",
"PEM": "Punta Cana International Airport, Punta Cana, Dominican Republic",
"SFO": "San Francisco International Airport, San Francisco, USA",
"CAN": "Guangzhou Baiyun International Airport, Guangzhou, China",
"TLV": "Ben Gurion International Airport, Tel Aviv, Israel",
"GRU": "São Paulo–Guarulhos International Airport, São Paulo, Brazil",
"CLT": "Charlotte Douglas International Airport, Charlotte, USA",
"HEL": "Helsinki-Vantaa Airport, Helsinki, Finland",
"SVO": "Sheremetyevo International Airport, Moscow, Russia",
"DTW": "Detroit Metropolitan Airport, Detroit, USA",
"MEL": "Melbourne Tullamarine Airport, Melbourne, Australia",
"MUC": "Munich Airport, Munich, Germany",
"JNB": "O.R. Tambo International Airport, Johannesburg, South Africa",
"JED": "King Abdulaziz International Airport, Jeddah, Saudi Arabia",
"NGO": "Chubu Centrair International Airport, Nagoya, Japan",
"BNE": "Brisbane Airport, Brisbane, Australia",
"YYZ": "Toronto Pearson International Airport, Toronto, Canada",
"MEX": "Mexico City International Airport, Mexico City, Mexico",
"KTW": "Katowice International Airport, Katowice, Poland",
"KIX": "Kansai International Airport, Osaka, Japan",
"TPE": "Taipei Taoyuan International Airport, Taipei, Taiwan",
"STN": "London Stansted Airport, London, UK",
"CMN": "Mohammed V International Airport, Casablanca, Morocco",
"VIE": "Vienna International Airport, Vienna, Austria",
"YYC": "Calgary International Airport, Calgary, Canada",
"YUL": "Montréal-Pierre Elliott Trudeau International Airport, Montreal, Canada",
"RIX": "Riga International Airport, Riga, Latvia",
"BOM": "Chhatrapati Shivaji International Airport, Mumbai, India",
"IKA": "Imam Khomeini International Airport, Tehran, Iran",
"KUL": "Kuala Lumpur International Airport, Kuala Lumpur, Malaysia",
"FCO": "Leonardo da Vinci International Airport, Rome, Italy",
"STO": "Stockholm Arlanda Airport, Stockholm, Sweden",
"YEG": "Edmonton International Airport, Edmonton, Canada",
"SCL": "Comodoro Arturo Merino Benítez International Airport, Santiago, Chile",
"MXP": "Milan Malpensa Airport, Milan, Italy",
"SHA": "Shanghai Hongqiao International Airport, Shanghai, China",
"MAN": "Manchester Airport, Manchester, UK",
"RUH": "King Khalid International Airport, Riyadh, Saudi Arabia",
"BLR": "Kempegowda International Airport, Bangalore, India",
"PEK": "Beijing Capital International Airport, Beijing, China",
"HKG": "Hong Kong International Airport, Hong Kong, China",
"VCP": "Viracopos International Airport, Campinas, Brazil",
"BCN": "Barcelona El Prat Airport, Barcelona, Spain",
"ORD": "O'Hare International Airport, Chicago, USA",
"GRU": "São Paulo–Guarulhos International Airport, São Paulo, Brazil",
"FLL": "Fort Lauderdale-Hollywood International Airport, Fort Lauderdale, USA",
"STL": "Lambert-St. Louis International Airport, St. Louis, USA",
"SJC": "Norman Y. Mineta San José International Airport, San José, USA",
"DFW": "Dallas/Fort Worth International Airport, Dallas/Fort Worth, USA",
"SCL": "Comodoro Arturo Merino Benítez International Airport, Santiago, Chile",
"PMI": "Palma de Mallorca Airport, Palma, Spain",
"TPA": "Tampa International Airport, Tampa, USA",
"IAH": "George Bush Intercontinental Airport, Houston, USA",
"MAD": "Adolfo Suárez Madrid–Barajas Airport, Madrid, Spain",
"YVR": "Vancouver International Airport, Vancouver, Canada",
"PDX": "Portland International Airport, Portland, USA",
"LIS": "Humberto Delgado Airport, Lisbon, Portugal",
"LTN": "London Luton Airport, London, UK",
"AMS": "Amsterdam Schiphol Airport, Amsterdam, Netherlands"

}

# -----------------------------------------------------------
# Helper Function: Flight Duration as a String
# -----------------------------------------------------------
def flight_duration_str(departure, arrival):
    if not departure or not arrival:
         return "N/A"
    duration = arrival - departure
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    return f"{hours}h {minutes}m"

# -----------------------------------------------------------
# Meteostat API Helpers (using provided API key)
# -----------------------------------------------------------
def get_nearest_station(lat, lon):
    url = "https://meteostat.p.rapidapi.com/stations/nearby"
    querystring = {"lat": str(lat), "lon": str(lon)}
    headers = {
        "x-rapidapi-key": "31464ac9a7msh75dff156d129a3dp1070a2jsn5e905529d890",
        "x-rapidapi-host": "meteostat.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    if "data" in data and len(data["data"]) > 0:
        return data["data"][0]["id"]
    return None

def get_weather_for_time(lat, lon, target_time):
    station = get_nearest_station(lat, lon)
    if not station:
        return {}
    date_str = target_time.strftime("%Y-%m-%d")
    url = "https://meteostat.p.rapidapi.com/stations/hourly"
    querystring = {"station": station, "start": date_str, "end": date_str}
    headers = {
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

# -----------------------------------------------------------
# Helper: Retrieve Airport Info from RapidAPI (if needed)
# -----------------------------------------------------------
def get_airport_info(iata_code):
    url = "https://airport-info.p.rapidapi.com/airport"
    querystring = {"iata": iata_code}
    headers = {
        "x-rapidapi-key": "31464ac9a7msh75dff156d129a3dp1070a2jsn5e905529d890",  # Replace with your key if needed
        "x-rapidapi-host": "airport-info.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    return None

# -----------------------------------------------------------
# Helper: Retrieve Flight Delays using the Flight Delays API
# -----------------------------------------------------------
def get_flight_delays(dep_iata):
    url = "https://zylalabs.com/api/2581/flight+delays+api/2581/get+delays"
    params = {
        "delay": "30",         # fixed delay parameter
        "type": "departures",   # fixed type parameter
        "dep_iata": dep_iata    # user-provided departure airport
    }
    headers = {
        "Authorization": "Bearer 7242|LR7VxWh2ZH6i8XPom8EY4pCkPsWh6cOyM2c3skuW"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("success") and "data" in data:
            return data["data"]
    return []

# -----------------------------------------------------------
# Session State for Flight Options
# -----------------------------------------------------------
if "flight_options" not in st.session_state:
    st.session_state.flight_options = []

# -----------------------------------------------------------
# Custom CSS for Flight Cards: Increased width, reduced vertical padding, and aligned text
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    .flight-card {
        background-color: #f8f8f8;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    .flight-left, .flight-middle, .flight-right {
        flex: 1;
        padding: 0 10px;
    }
    .flight-middle {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
    }
    .flight-left h2, .flight-middle h4, .flight-right p {
        margin: 0;
    }
    .flight-left p, .flight-middle p {
        margin: 0.2rem 0;
    }
    .weather-icon {
        width: 96px;
        height: 96px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# Streamlit Dashboard Layout
# -----------------------------------------------------------
st.title("Delayed Flight Visualization Dashboard")
st.markdown("This dashboard displays delayed flight options using live flight delay data from the API.")

# Sidebar: User input for departure airport and departure date
st.sidebar.header("Flight Details")
dep_airport_input = st.sidebar.text_input("Departure Airport (IATA code)", "JFK")
dep_date = st.sidebar.date_input("Departure Date", datetime.today())

# -----------------------------------------------------------
# Step 1: Fetch Flight Delays using the API when the button is clicked
# -----------------------------------------------------------
if st.sidebar.button("Get Flight Delays"):
    flights_api = get_flight_delays(dep_airport_input.upper())
    if flights_api:
        # Randomly select 5 flights (if more than 5 are returned)
        flight_options = random.sample(flights_api, 5) if len(flights_api) > 5 else flights_api
        updated_flights = []
        for flight in flight_options:
            try:
                # Use UTC times from the API:
                # Parse departure time from 'dep_time_utc'
                api_dep_time = datetime.strptime(flight["dep_time_utc"], "%Y-%m-%d %H:%M")
                new_dep_time = datetime.combine(dep_date, api_dep_time.time())
                flight["dep_time_utc"] = new_dep_time.strftime("%Y-%m-%d %H:%M")
                
                # Parse arrival time from 'arr_time_utc'
                api_arr_time = datetime.strptime(flight["arr_time_utc"], "%Y-%m-%d %H:%M")
                new_arr_time = datetime.combine(dep_date, api_arr_time.time())
                # If arrival time is earlier than departure, assume flight arrives next day
                if new_arr_time < new_dep_time:
                    new_arr_time += timedelta(days=1)
                flight["arr_time_utc"] = new_arr_time.strftime("%Y-%m-%d %H:%M")
                updated_flights.append(flight)
            except Exception:
                continue
        st.session_state.flight_options = updated_flights
    else:
        st.error("No flight delay data found.")

# -----------------------------------------------------------
# Step 2: Display Flight Cards for Each Delayed Flight Option
# -----------------------------------------------------------
# ... (the rest of your imports, mappings, and code remain the same)

if st.session_state.flight_options:
    st.write("### Delayed Flight Options:")
    for flight in st.session_state.flight_options:
        flight_number = flight.get("flight_number", "N/A")
        airline_code = flight.get("airline_iata", "N/A")
        # Lookup full carrier name
        carrier_full = carrier_names.get(airline_code, airline_code)

        dep_time_str = flight.get("dep_time_utc", "N/A")
        arr_time_str = flight.get("arr_time_utc", "N/A")

        try:
            sched_dep = datetime.strptime(dep_time_str, "%Y-%m-%d %H:%M")
            dep_display = sched_dep.strftime("%I:%M %p, %d %b %Y")
        except Exception:
            dep_display = dep_time_str

        try:
            sched_arr = datetime.strptime(arr_time_str, "%Y-%m-%d %H:%M")
            arr_display = sched_arr.strftime("%I:%M %p, %d %b %Y")
        except Exception:
            arr_display = arr_time_str

        # Calculate flight duration
        duration_display = flight_duration_str(sched_dep, sched_arr)

        # Get airport info from dictionary
        dep_airport_code = dep_airport_input.upper()
        dep_airport_full = airport_info_dict.get(dep_airport_code, dep_airport_code)

        arr_airport_code = flight.get("arr_iata", "N/A").upper()
        arr_airport_full = airport_info_dict.get(arr_airport_code, arr_airport_code)

        # Retrieve weather info for departure airport/time
        dep_info_for_weather = get_airport_info(dep_airport_code)
        if dep_info_for_weather:
            dep_lat = dep_info_for_weather.get("latitude")
            dep_lon = dep_info_for_weather.get("longitude")
            weather_info = get_weather_for_time(dep_lat, dep_lon, sched_dep)

            def get_weather_icon_url(dep_weather, sched_dep):
                if dep_weather.get("snow") and dep_weather.get("snow") > 0:
                    return "https://img.icons8.com/fluency/96/snow--v1.png"
                elif dep_weather.get("prcp") and dep_weather.get("prcp") > 0:
                    return "https://img.icons8.com/fluency/96/rain.png"
                elif dep_weather.get("wspd") and dep_weather.get("wspd") > 10:
                    return "https://img.icons8.com/fluency/96/windy-weather.png"
                else:
                    if 7 <= sched_dep.hour < 19:
                        return "https://img.icons8.com/fluency/96/sun.png"
                    else:
                        return "https://img.icons8.com/fluency/96/moon-symbol.png"
            icon_url = get_weather_icon_url(weather_info, sched_dep)
        else:
            icon_url = "https://img.icons8.com/fluency/96/cloud.png"

        # -------------------------
        # Flight Card HTML
        # -------------------------
        flight_card_html = f"""
        <div class="flight-card">
          <!-- Left column: Flight number & carrier -->
          <div class="flight-left">
            <h2 style="margin: 0; font-size: 1.2rem;">{flight_number}</h2>
            <p style="font-size: 0.9rem; margin: 0;">
              Carrier Name: {carrier_full} ({airline_code})
            </p>
          </div>

          <!-- Middle column: Airport info & times -->
          <div class="flight-middle">
            <p style="font-size: 0.9rem; margin: 0;">
              {dep_airport_code}: {dep_airport_full}
            </p>
            <p style="font-size: 0.9rem; margin: 0;">
              Dep: {dep_display}
            </p>
            <br/>
            <p style="font-size: 0.9rem; margin: 0;">
              {arr_airport_code}: {arr_airport_full}
            </p>
            <p style="font-size: 0.9rem; margin: 0;">
              Arr: {arr_display}
            </p>
          </div>

          <!-- Right column: Weather icon & flight duration -->
          <div class="flight-right" style="text-align: center;">
            <img src="{icon_url}" class="weather-icon" alt="Weather Icon" />
            <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0;">
              Duration: {duration_display}
            </p>
          </div>
        </div>
        """

        st.markdown(flight_card_html, unsafe_allow_html=True)


# In[ ]:




