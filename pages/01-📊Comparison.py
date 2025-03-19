import streamlit as st
import pandas as pd
import glob
import os
import calendar
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------------
# 1) Page Setup
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide")

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
    # Convert departure times
    for col in ["FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME"]:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    # Convert arrival times
    for col in ["FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME"]:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')

    df = df.dropna(subset=["FILED OFF BLOCK TIME", "FILED ARRIVAL TIME"])

    # Compute delays (in minutes)
    df['DEPTDEL'] = (df["ACTUAL OFF BLOCK TIME"] -
                     df["FILED OFF BLOCK TIME"]).dt.total_seconds() / 60
    df['ARVLDEL'] = (df["ACTUAL ARRIVAL TIME"] -
                     df["FILED ARRIVAL TIME"]).dt.total_seconds() / 60
    return df


df = load_data()

# --------------------------------------------------------------------------------
# 3) Sidebar Controls & Comparison Mode Selection
# --------------------------------------------------------------------------------
st.sidebar.title("Filters")

# Choose comparison mode: Hourly (Daily) or Monthly
comparison_mode = st.sidebar.radio(
    "Comparison Mode", options=["Daily", "Monthly"])

# Airport selection by city: show as "CODE (City)"
airport_df = df[['ADEP', 'City']].dropna(
    subset=['ADEP', 'City']).drop_duplicates()
airport_df["option"] = airport_df["ADEP"].astype(
    str) + " (" + airport_df["City"].astype(str) + ")"
airport_options = sorted(airport_df["option"].unique(), key=lambda x: str(x))
selected_airports = st.sidebar.multiselect(
    "Select Airport(s)", options=airport_options)
if not selected_airports:
    st.warning("Please select at least one airport.")
    st.stop()
# Extract airport codes (first token)
selected_airport_codes = [option.split()[0] for option in selected_airports]

# --------------------------------------------------------------------------------
# 4) Hourly (Daily) Mode Setup
# --------------------------------------------------------------------------------
if comparison_mode == "Daily":
    # Rename sidebar label to reflect daily selection
    selected_dates = st.sidebar.date_input("Select Daily Date(s)", value=[
                                           df["FILED OFF BLOCK TIME"].max().date() - pd.Timedelta(days=1)])
    if not isinstance(selected_dates, (list, tuple)):
        selected_dates = [selected_dates]
    elif len(selected_dates) == 1 and isinstance(selected_dates[0], tuple):
        selected_dates = list(selected_dates[0])
    else:
        selected_dates = list(selected_dates)
    selected_dates = sorted(selected_dates)

    # Custom hour ordering: x-axis spans from 04 to 03
    custom_hours = list(range(4, 24)) + list(range(24, 28))
    custom_labels = [f"{h:02d}" if h <
                     24 else f"{h-24:02d}" for h in custom_hours]

    # Functions to compute hourly averages
    def get_hourly_avg_dep_delay(df, date, airport):
        date = pd.Timestamp(date)
        subset = df[(df["FILED OFF BLOCK TIME"].dt.date ==
                     date.date()) & (df["ADEP"] == airport)].copy()
        subset = subset[subset["DEPTDEL"] > 0]
        subset['Hour'] = subset["FILED OFF BLOCK TIME"].dt.hour.astype(int)
        subset['Hour_order'] = subset['Hour'].apply(
            lambda x: x if x >= 4 else x + 24)
        hourly = subset.groupby("Hour_order")["DEPTDEL"].mean().reset_index()
        hourly['Time'] = hourly['Hour_order'].apply(
            lambda x: f"{x if x < 24 else x-24:02d}")
        return hourly.sort_values("Hour_order")

    def get_hourly_avg_arr_delay(df, date, airport):
        date = pd.Timestamp(date)
        subset = df[(df["FILED ARRIVAL TIME"].dt.date == date.date())
                    & (df["ADEP"] == airport)].copy()
        subset = subset[subset["ARVLDEL"] > 0]
        subset['Hour'] = subset["FILED ARRIVAL TIME"].dt.hour.astype(int)
        subset['Hour_order'] = subset['Hour'].apply(
            lambda x: x if x >= 4 else x + 24)
        hourly = subset.groupby("Hour_order")["ARVLDEL"].mean().reset_index()
        hourly['Time'] = hourly['Hour_order'].apply(
            lambda x: f"{x if x < 24 else x-24:02d}")
        return hourly.sort_values("Hour_order")

    # Determine dash styles and marker symbols based on date order
    dash_styles = {}
    marker_symbols = {}
    if len(selected_dates) == 1:
        date_str = pd.Timestamp(selected_dates[0]).strftime("%Y-%m-%d")
        dash_styles[date_str] = "solid"
        marker_symbols[date_str] = "circle"
    elif len(selected_dates) == 2:
        date_str1 = pd.Timestamp(selected_dates[0]).strftime("%Y-%m-%d")
        date_str2 = pd.Timestamp(selected_dates[1]).strftime("%Y-%m-%d")
        dash_styles[date_str1] = "dash"
        dash_styles[date_str2] = "solid"
        marker_symbols[date_str1] = "circle"
        marker_symbols[date_str2] = "square"

    palette = px.colors.qualitative.Plotly
    color_map = {airport: palette[i % len(
        palette)] for i, airport in enumerate(selected_airport_codes)}

    # ----- Create Hourly Departure Delay Plot -----
    fig_dep = go.Figure()
    for airport in selected_airport_codes:
        for d in selected_dates:
            date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            hourly_data = get_hourly_avg_dep_delay(df, d, airport)
            # Merge with full set of hours to ensure all hours (04 to 03) appear
            full_hours_df = pd.DataFrame(
                {"Hour_order": custom_hours, "Time": custom_labels})
            hourly_data = pd.merge(full_hours_df, hourly_data, on=[
                                   "Hour_order", "Time"], how="left")
            hourly_data = hourly_data.sort_values("Hour_order")
            fig_dep.add_trace(
                go.Scatter(
                    x=hourly_data["Hour_order"],
                    y=hourly_data["DEPTDEL"],
                    mode="lines+markers",
                    name=f"{airport} - {date_str}",
                    line=dict(
                        color=color_map[airport],
                        width=1.5,
                        dash=dash_styles.get(date_str, "solid"),
                        shape="spline",
                        smoothing=1.3
                    ),
                    marker=dict(
                        symbol=marker_symbols.get(date_str, "circle"),
                        size=8,
                        color=color_map[airport]
                    ),
                    connectgaps=True,
                    hovertemplate="Dep Del: %{y:.2f} min<extra></extra>"
                )
            )
    fig_dep.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=custom_hours,
            ticktext=custom_labels,
            title="Hour (4:00 Am to 3:00 Am next Day)"
        ),
        yaxis=dict(title="Average Departure Delay (min)"),
        title="Daily Departure Delay Comparison",
        hovermode="x unified"
    )

    # ----- Create Hourly Arrival Delay Plot -----
    fig_arr = go.Figure()
    for airport in selected_airport_codes:
        for d in selected_dates:
            date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            hourly_data = get_hourly_avg_arr_delay(df, d, airport)
            full_hours_df = pd.DataFrame(
                {"Hour_order": custom_hours, "Time": custom_labels})
            hourly_data = pd.merge(full_hours_df, hourly_data, on=[
                                   "Hour_order", "Time"], how="left")
            hourly_data = hourly_data.sort_values("Hour_order")
            fig_arr.add_trace(
                go.Scatter(
                    x=hourly_data["Hour_order"],
                    y=hourly_data["ARVLDEL"],
                    mode="lines+markers",
                    name=f"{airport} - {date_str}",
                    line=dict(
                        color=color_map[airport],
                        width=1.5,
                        dash=dash_styles.get(date_str, "solid"),
                        shape="spline",
                        smoothing=1.3
                    ),
                    marker=dict(
                        symbol=marker_symbols.get(date_str, "circle"),
                        size=8,
                        color=color_map[airport]
                    ),
                    connectgaps=True,
                    hovertemplate="Arr Del: %{y:.2f} min<extra></extra>"
                )
            )
    fig_arr.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=custom_hours,
            ticktext=custom_labels,
            title="Hour (4:00 Am to 3:00 Am next Day)"
        ),
        yaxis=dict(title="Average Arrival Delay (min)"),
        title="Daily Arrival Delay Comparison",
        hovermode="x unified"
    )

    st.plotly_chart(fig_dep, use_container_width=True)
    st.plotly_chart(fig_arr, use_container_width=True)

# --------------------------------------------------------------------------------
# 5) Monthly Mode Setup
# --------------------------------------------------------------------------------
elif comparison_mode == "Monthly":
    # For monthly mode, let the user select one or two months (with year shown as "2018 - MM")
    available_months = [
        f"2018 - {month:02d}" for month in sorted(df["FILED OFF BLOCK TIME"].dt.month.unique())]
    selected_months = st.sidebar.multiselect(
        "Select Month(s)", options=available_months, default=[available_months[-1]])
    if not selected_months:
        st.warning("Please select at least one month for monthly comparison.")
        st.stop()
    selected_months = sorted(selected_months)
    # Parse selected months to get month numbers
    parsed_months = [int(option.split("-")[1].strip())
                     for option in selected_months]
    # Determine common x-axis range: days 1 to min(max_day for each selected month)
    max_days = {m: calendar.monthrange(2018, m)[1] for m in parsed_months}
    overall_max_day = min(max_days.values())
    full_days = pd.DataFrame({"Day": list(range(1, overall_max_day + 1))})

    # --- UPDATED FUNCTIONS: now filter by airport ---
    def get_daily_avg_dep_delay(df, month, airport):
        subset = df[(df["FILED OFF BLOCK TIME"].dt.month == month)
                    & (df["ADEP"] == airport)].copy()
        subset = subset[subset["DEPTDEL"] > 0]
        subset["Day"] = subset["FILED OFF BLOCK TIME"].dt.day.astype(int)
        daily = subset.groupby("Day")["DEPTDEL"].mean().reset_index()
        return daily

    def get_daily_avg_arr_delay(df, month, airport):
        subset = df[(df["FILED ARRIVAL TIME"].dt.month == month)
                    & (df["ADEP"] == airport)].copy()
        subset = subset[subset["ARVLDEL"] > 0]
        subset["Day"] = subset["FILED ARRIVAL TIME"].dt.day.astype(int)
        daily = subset.groupby("Day")["ARVLDEL"].mean().reset_index()
        return daily

    # Determine dash styles and marker symbols for months (earlier month dashed)
    dash_styles_month = {}
    marker_symbols_month = {}
    if len(parsed_months) == 1:
        m_str = str(parsed_months[0])
        dash_styles_month[m_str] = "solid"
        marker_symbols_month[m_str] = "circle"
    elif len(parsed_months) == 2:
        m_str1 = str(parsed_months[0])
        m_str2 = str(parsed_months[1])
        dash_styles_month[m_str1] = "dash"
        dash_styles_month[m_str2] = "solid"
        marker_symbols_month[m_str1] = "circle"
        marker_symbols_month[m_str2] = "square"

    palette = px.colors.qualitative.Plotly
    color_map_month = {airport: palette[i % len(
        palette)] for i, airport in enumerate(selected_airport_codes)}

    # ----- Create Monthly Departure Delay Plot -----
    fig_dep_month = go.Figure()
    # Outer loop: for each selected month, then for each airport.
    for option in selected_months:
        m = int(option.split("-")[1].strip())
        for airport in selected_airport_codes:
            m_str = option  # e.g. "2018 - 03"
            daily_data = get_daily_avg_dep_delay(df, m, airport)
            # Create full days for this month
            full_days_month = pd.DataFrame(
                {"Day": list(range(1, calendar.monthrange(2018, m)[1] + 1))})
            daily_data = pd.merge(
                full_days_month, daily_data, on="Day", how="left")
            # Restrict to common days among selected months (overall_max_day)
            daily_data = daily_data[daily_data["Day"] <= overall_max_day]
            fig_dep_month.add_trace(
                go.Scatter(
                    x=daily_data["Day"],
                    y=daily_data["DEPTDEL"],
                    mode="lines+markers",
                    name=f"{airport} - {m_str}",
                    line=dict(
                        color=color_map_month[airport],
                        width=1.5,
                        dash=dash_styles_month.get(str(m), "solid"),
                        shape="spline",
                        smoothing=1.3
                    ),
                    marker=dict(
                        symbol=marker_symbols_month.get(str(m), "circle"),
                        size=8,
                        color=color_map_month[airport]
                    ),
                    connectgaps=True,
                    hovertemplate="Dep Del: %{y:.2f} min<extra></extra>"
                )
            )
    fig_dep_month.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1,
            range=[1, overall_max_day],
            title="Day of Month"
        ),
        yaxis=dict(title="Average Departure Delay (min)"),
        title="Monthly Departure Delay Comparison",
        hovermode="x unified"
    )

    # ----- Create Monthly Arrival Delay Plot -----
    fig_arr_month = go.Figure()
    for option in selected_months:
        m = int(option.split("-")[1].strip())
        for airport in selected_airport_codes:
            m_str = option
            daily_data = get_daily_avg_arr_delay(df, m, airport)
            full_days_month = pd.DataFrame(
                {"Day": list(range(1, calendar.monthrange(2018, m)[1] + 1))})
            daily_data = pd.merge(
                full_days_month, daily_data, on="Day", how="left")
            daily_data = daily_data[daily_data["Day"] <= overall_max_day]
            fig_arr_month.add_trace(
                go.Scatter(
                    x=daily_data["Day"],
                    y=daily_data["ARVLDEL"],
                    mode="lines+markers",
                    name=f"{airport} - {m_str}",
                    line=dict(
                        color=color_map_month[airport],
                        width=1.5,
                        dash=dash_styles_month.get(str(m), "solid"),
                        shape="spline",
                        smoothing=1.3
                    ),
                    marker=dict(
                        symbol=marker_symbols_month.get(str(m), "circle"),
                        size=8,
                        color=color_map_month[airport]
                    ),
                    connectgaps=True,
                    hovertemplate="Arr Del: %{y:.2f} min<extra></extra>"
                )
            )
    fig_arr_month.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1,
            range=[1, overall_max_day],
            title="Day of Month"
        ),
        yaxis=dict(title="Average Arrival Delay (min)"),
        title="Monthly Arrival Delay Comparison",
        hovermode="x unified"
    )

    st.plotly_chart(fig_dep_month, use_container_width=True)
    st.plotly_chart(fig_arr_month, use_container_width=True)
