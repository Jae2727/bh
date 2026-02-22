# Top3 Propose Date with Streamlit_Final (input date within 3 months limitation version)
# Add White Night Exception

import streamlit as st
import datetime
import math
import pandas as pd
import pytz

from astral import LocationInfo, moon
from astral.sun import sun, elevation

from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

from skyfield.api import load, utc



# Function to get location and timezone
def get_location(city_name):
    geolocator = Nominatim(user_agent="twilight_app")
    location = geolocator.geocode(city_name)

    if location is None:
        raise ValueError("City not found. Please enter a valid city name.")

    else:
        # Extraction of latitude and longitude
        lat = location.latitude
        lon = location.longitude

        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=lat, lng=lon)
        tz = pytz.timezone(timezone_str)
        
        # Create a LocationInfo object
        city = LocationInfo(city_name, "", timezone_str, lat, lon)
        
        return city, tz


# Solar information by date and location
def get_sun_times(city, date_input, tz):
    try:
        s = sun(city.observer, date=date_input, tzinfo=tz)
        return s["sunset"], s["dusk"]
    except Exception:
        # White Night Exception Handling
        return None, None


# Feature 1 - Duration of Civil Twilight
# Longer duration means more time for pink time, which is better for photography
def civil_twilight_duration(sunset, dusk):
    duration = (dusk - sunset).total_seconds() / 60  # in minutes
    return duration


# Feature 2 - Moon Illuminated Fraction
# A lower illuminated fraction means less moonlight, which can enhance the visibility of pink time colors
def moon_illuminated_fraction(date_input):
    date = datetime.date(date_input.year, date_input.month, date_input.day)
    moon_age = moon.phase(date)
    moon_cycle = 29.530588
    moon_phase_angle = (moon_age / moon_cycle) * 2 * math.pi
    illuminated_fraction = (1 - math.cos(moon_phase_angle)) / 2
    return illuminated_fraction


# Feature 3 - Distance between Sun and Earth
# A greater distance can lead to a more intense pink time due to less atmospheric scattering of sunlight
def sun_earth_distance(date_input):
    # load astronomical data (ephemeris))
    planets = load('de421.bsp')
    earth, sun = planets['earth'], planets['sun']
    # datetime including timezone information
    date = datetime.datetime(date_input.year, date_input.month, date_input.day, tzinfo=utc)
    ts = load.timescale()
    t = ts.from_datetime(date)
    # calculation of distance
    astrometric = earth.at(t).observe(sun)
    distance = astrometric.distance()
    return distance.km #in kilometers


# 30 days calculation
def thirty_days_values(city, date_input, tz):
    results = []
    skipped_days = 0
    for i in range(30):
        m_date = date_input + datetime.timedelta(days=i)
        
        # get_sun_times returns sunset and dusk times for the given date and location
        sunset, dusk = get_sun_times(city, m_date, tz)
        
        if sunset is None or dusk is None:
            # Handle White Night Exception
            skipped_days += 1
            continue # Skip this date as it does not have valid sunset and dusk times

        f1 = civil_twilight_duration(sunset, dusk)
        f2 = moon_illuminated_fraction(m_date)
        f3 = sun_earth_distance(m_date)
        
        results.append({
            "date": m_date,
            "f1": f1,
            "f2": f2,
            "f3": f3,
        })
    return pd.DataFrame(results), skipped_days


# Nomralization and Final Score Calcuation
def calculate_final_score(df):
    for col in ["f1", "f2", "f3"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df["final_score"] = df["f1"] * 0.55 + (1 - df["f2"]) * 0.35 + df["f3"] * 0.1
    return df




#Streamlit App
# App title
st.title("Big Ring Theory")

today = datetime.date.today()
three_months_later = today + datetime.timedelta(days=90)

# User input for city and date
city_name = st.text_input("Enter a city name:")
date_input = st.date_input("Select a date:", min_value=today, max_value=three_months_later, value=today)


if st.button("Calculate"):
    city, tz = get_location(city_name)
    df_raw, skipped_days = thirty_days_values(city, date_input, tz)
    
    if skipped_days > 0:
        st.warning(f"{skipped_days} day(s) were skipped due to White Night conditions where the sun does not set.")
    if len(df_raw) < 5:
        st.error("Too few valid days to generate recommendations.")
        st.stop()

    df = calculate_final_score(df_raw)
    top3 = df.sort_values("final_score", ascending = False).head(3)

    for _, row in top3.iterrows():
        top3_date = row["date"]
        sunset, dusk = get_sun_times(city, top3_date, tz)
    
        #Pink time calculation for the top3 dates
        pink_start = None
        pink_end = None
        current = sunset

        while current <= dusk:
            h = elevation(city.observer, current)

            if -4 <= h <= -1:
                if pink_start is None:
                    pink_start = current
                pink_end = current

            current += datetime.timedelta(minutes=1)
        
        # Display results
        st.write("Top3 dates for Proposal: ", top3_date)
        st.write("Civil Twilight: ", sunset.strftime("%H:%M"),
                "~", dusk.strftime("%H:%M"))
        st.write("Pink Time: ", pink_start.strftime("%H:%M"),
                "~", pink_end.strftime("%H:%M"))

