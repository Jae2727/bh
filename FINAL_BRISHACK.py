#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 08:09:36 2026

@author: zehra
"""

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
import requests
import calendar
import numpy as np


def get_location(city_name):
    geolocator = Nominatim(user_agent="twilight_app")
    location = geolocator.geocode(city_name)
    if location is None:
        raise ValueError("City not found. Please enter a valid city name.")
    lat = location.latitude
    lon = location.longitude
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    tz = pytz.timezone(timezone_str)
    city = LocationInfo(city_name, "", timezone_str, lat, lon)
    return city, tz


def get_sun_times(city, date_input, tz):
    try:
        s = sun(city.observer, date=date_input, tzinfo=tz)
        return s["sunset"], s["dusk"]
    except Exception:
        return None, None


def civil_twilight_duration(sunset, dusk):
    return (dusk - sunset).total_seconds() / 60


def moon_illuminated_fraction(date_input):
    date = datetime.date(date_input.year, date_input.month, date_input.day)
    moon_age = moon.phase(date)
    moon_cycle = 29.530588
    moon_phase_angle = (moon_age / moon_cycle) * 2 * math.pi
    return (1 - math.cos(moon_phase_angle)) / 2


def sun_earth_distance(date_input):
    planets = load('de421.bsp')
    earth, sun = planets['earth'], planets['sun']
    date = datetime.datetime(date_input.year, date_input.month, date_input.day, tzinfo=utc)
    ts = load.timescale()
    t = ts.from_datetime(date)
    astrometric = earth.at(t).observe(sun)
    return astrometric.distance().km


def thirty_days_values(city, date_input, tz):
    results = []
    skipped_days = 0
    for i in range(30):
        m_date = date_input + datetime.timedelta(days=i)
        sunset, dusk = get_sun_times(city, m_date, tz)
        if sunset is None or dusk is None:
            skipped_days += 1
            continue
        results.append({
            "date": m_date,
            "f1": civil_twilight_duration(sunset, dusk),
            "f2": moon_illuminated_fraction(m_date),
            "f3": sun_earth_distance(m_date),
        })
    return pd.DataFrame(results), skipped_days


def calculate_final_score(df):
    for col in ["f1", "f2", "f3"]:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df["twilight_score"] = df["f1"] * 0.55 + (1 - df["f2"]) * 0.35 + df["f3"] * 0.1
    return df


def get_romantic_weather_prediction(city_name, target_month, target_year=2026):
    geolocator = Nominatim(user_agent="big_ring_theory")
    location = geolocator.geocode(city_name)
    if not location:
        return pd.DataFrame()

    lat, lon = location.latitude, location.longitude
    is_northern = lat > 0

    if is_northern:
        if target_month in [6, 7, 8]:    ideal_t = 20
        elif target_month in [12, 1, 2]: ideal_t = 5
        elif target_month in [3, 4, 5]:  ideal_t = 15
        else:                             ideal_t = 12
    else:
        if target_month in [12, 1, 2]:   ideal_t = 20
        elif target_month in [6, 7, 8]:  ideal_t = 5
        elif target_month in [3, 4, 5]:  ideal_t = 15
        else:                             ideal_t = 12

    all_years = []
    for year in range(target_year - 3, target_year):
        last_day   = calendar.monthrange(year, target_month)[1]
        start_date = f"{year}-{target_month:02d}-01"
        end_date   = f"{year}-{target_month:02d}-{last_day}"

        w_params  = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
                     "daily": ["temperature_2m_max", "precipitation_sum", "cloud_cover_mean", "wind_speed_10m_max"],
                     "timezone": "auto"}
        aq_params = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
                     "hourly": ["pm2_5", "ozone"], "timezone": "auto"}
        try:
            w_res  = requests.get("https://archive-api.open-meteo.com/v1/archive",        params=w_params)
            aq_res = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=aq_params)

            if w_res.status_code == 200:
                df_w    = pd.DataFrame(w_res.json()['daily'])
                aq_data = aq_res.json()

                if 'hourly' in aq_data:
                    df_aq = pd.DataFrame(aq_data['hourly'])
                    df_aq['date'] = pd.to_datetime(df_aq['time']).dt.date
                    df_w['pm2_5'] = df_aq.groupby('date')['pm2_5'].mean().values if 'pm2_5' in df_aq.columns else 10.0
                    df_w['ozone'] = df_aq.groupby('date')['ozone'].mean().values if 'ozone' in df_aq.columns else 300.0
                else:
                    df_w['pm2_5'] = 10.0
                    df_w['ozone'] = 300.0

                all_years.append(df_w)
        except Exception as e:
            print(f"Error fetching data for {year}: {e}")

    if not all_years:
        return pd.DataFrame()

    hist = pd.concat(all_years)
    hist['day'] = pd.to_datetime(hist['time']).dt.day
    daily_avg   = hist.groupby('day').mean(numeric_only=True).reset_index()

    def calc_score(row):
        c_score = (100 - row['cloud_cover_mean']) / 100        # 0–1
        t_score = np.exp(-((row['temperature_2m_max'] - ideal_t)**2) / (2 * 5**2))
        color_bonus = min(0.1, (row['ozone'] / 350) * 0.1)    # was max 10, now max 0.1


        r_mult = 1.0 if row['precipitation_sum']  < 0.1 else max(0,   1 - row['precipitation_sum']  / 3)
        w_mult = 1.0 if row['wind_speed_10m_max'] < 12  else max(0.1, 1 - row['wind_speed_10m_max'] / 18)
        p_mult = 1.0 if row['pm2_5']              < 10  else max(0,   1 - (row['pm2_5'] - 10) / 40)

        finalw_score = (c_score * 0.35) + (t_score * 0.20) + (r_mult * 0.25) + (w_mult * 0.15) + (p_mult * 0.05) + color_bonus

        if r_mult < 0.9:        vibe = "💧 WATER | Trust the flow; clarity comes after the soak."
        elif w_mult < 0.9:      vibe = "🌬️ AIR | Fresh perspectives are heading your way."
        elif color_bonus > 8.5: vibe = "🩷 Cotton-candy sky | The ozone is electric. Expect majestic sky hues."
        elif finalw_score > 80: vibe = "✨ ETHER | THE Perfect Match. The sky is cosmically aligned."
        elif t_score > 80:      vibe = "🔥 FIRE | Fortune favors the bold—make that big move today."
        else:                   vibe = "🌿 EARTH | Grounded and stable. A day for steady progress."

        # Normalize to 0-1 here so display is clean percentage everywhere
        return pd.Series({'weather_score': round(finalw_score, 4), 'vibe': vibe})

    new_cols     = daily_avg.apply(calc_score, axis=1)
    full_results = pd.concat([daily_avg, new_cols], axis=1)

    def safe_date(d):
        try:
            return datetime.date(target_year, target_month, int(d))
        except ValueError:
            return None

    full_results["date"] = full_results["day"].apply(safe_date)
    full_results = full_results.dropna(subset=["date"])

    return full_results[["date", "weather_score", "vibe"]]


# ── STREAMLIT APP ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Big Ring Theory", page_icon="💍", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(to bottom, #f12711, #f5af19, #654ea3, #24243e); color: white; }
h1, h2, h3, p, span, label { color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
.vibe-card { background-color: rgba(255,255,255,0.1); padding:20px; border-radius:20px; border:1px solid rgba(255,255,255,0.2); backdrop-filter: blur(15px); text-align:center; min-height:120px; }
[data-testid="stMetricValue"] { color:#f5af19 !important; font-weight:bold; }
.stButton>button { background: linear-gradient(45deg, #f12711, #f5af19); color:white !important; border-radius:30px; border:none; font-weight:bold; width:100%; }
.stButton>button:hover { transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

st.title("💍 Big Ring Theory")
st.write("Find your perfect proposal moment — where the sky, stars, and weather align.")

today              = datetime.date.today()
three_months_later = today + datetime.timedelta(days=90)

col1, col2 = st.columns(2)
with col1:
    city_name  = st.text_input("Where will you propose? (City)", placeholder="e.g. Bristol", key="city_input")
with col2:
    date_input = st.date_input("Starting date:", min_value=today, max_value=three_months_later, value=today)

if st.button("✨ Calculate Romantic Potential"):

    if not city_name:
        st.warning("Please enter a city name!")
    else:
        with st.spinner("Analyzing the stars and the atmosphere..."):

            try:
                city, tz = get_location(city_name)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            df_raw, skipped_days = thirty_days_values(city, date_input, tz)

            if skipped_days > 0:
                st.warning(f"{skipped_days} day(s) skipped due to White Night conditions.")
            if len(df_raw) < 5:
                st.error("Too few valid days to generate recommendations.")
                st.stop()

            df = calculate_final_score(df_raw)

            target_month = date_input.month
            target_year  = date_input.year
            df_weather   = get_romantic_weather_prediction(city_name, target_month, target_year)
            df_weather2  = get_romantic_weather_prediction(city_name, (target_month % 12) + 1, target_year)
            df_weather   = pd.concat([df_weather, df_weather2], ignore_index=True)

            if df_weather.empty:
                st.error("Could not fetch weather data. Please try again.")
                st.stop()

            df = pd.merge(df, df_weather[["date", "weather_score", "vibe"]], on="date", how="left")
            df["weather_score"] = df["weather_score"].fillna(0)
            df["vibe"]          = df["vibe"].fillna("🌿 EARTH | Grounded and stable. A day for steady progress.")

            df["romance_score"] = (df["twilight_score"] * 0.6 + df["weather_score"] * 0.4).round(4)

            top3 = df.sort_values("romance_score", ascending=False).head(3)

            st.balloons()
            st.subheader("🏆 Top 3 Proposal Dates")

            for rank, (_, row) in enumerate(top3.iterrows(), start=1):
                top3_date    = row["date"]
                sunset, dusk = get_sun_times(city, top3_date, tz)

                pink_start, pink_end = None, None
                current = sunset
                while current <= dusk:
                    h = elevation(city.observer, current)
                    if -4 <= h <= -1:
                        if pink_start is None:
                            pink_start = current
                        pink_end = current
                    current += datetime.timedelta(minutes=1)

                with st.container():
                    st.markdown(f"### #{rank} — {top3_date.strftime('%A, %d %B %Y')}")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("💍 Romance Score", f"{row['romance_score']:.0%}")
                    c2.metric("🌅 Twilight Score", f"{row['twilight_score']:.0%}")
                    c3.metric("🌤️ Weather Score",  f"{row['weather_score']:.0%}")

                    st.write(f"🌇 **Civil Twilight:** {sunset.strftime('%H:%M')} ~ {dusk.strftime('%H:%M')}")

                    if pink_start and pink_end:
                        st.write(f"🩷 **Pink Time:** {pink_start.strftime('%H:%M')} ~ {pink_end.strftime('%H:%M')}")
                    else:
                        st.write("🩷 **Pink Time:** Not available for this date.")

                    st.markdown(f'<div class="vibe-card"><p>{row["vibe"]}</p></div>', unsafe_allow_html=True)
                    st.divider()

