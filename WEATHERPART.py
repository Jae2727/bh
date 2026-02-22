#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 06:18:34 2026

@author: zehra
"""


import requests
import calendar
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim 
from datetime import datetime

def get_romantic_weather_prediction(city_name, target_month, target_year = 2026):
    print(f"--- PREDICTING TOP PROPOSAL DAYS FOR {city_name} on MONTH {target_month} ---")
    
    # Location Logic
    geolocator = Nominatim(user_agent="big_ring_theory") #nominatim turns cities into GPS coordinates
    location = geolocator.geocode(city_name) #Find the location of the city
    if not location: return "Location not found."
    
    lat, lon = location.latitude, location.longitude
    is_northern = lat > 0  

    #Ideal Temp for Northern H.
    if is_northern:
        if target_month in [6, 7, 8]: ideal_t = 20    # Summer
        elif target_month in [12, 1, 2]: ideal_t = 5  # Winter
        elif target_month in [3, 4, 5]: ideal_t = 15  # Spring
        else: ideal_t = 12                            # Autumn
        
    #Ideal Temp for Southern H.
    else:
        if target_month in [12, 1, 2]: ideal_t = 20   # Summer
        elif target_month in [6, 7, 8]: ideal_t = 5   # Winter
        elif target_month in [3, 4, 5]: ideal_t = 15  # Spring
        else: ideal_t = 12                            # Autumn

    #Data Collection Loop
    all_years = []
    current_year = 2026
    
    for year in range(current_year - 3, current_year):
        last_day = calendar.monthrange(year, target_month)[1] #Find the last day based on the month
        start_date = f"{year}-{target_month:02d}-01"
        end_date = f"{year}-{target_month:02d}-{last_day}"
        
        w_url = "https://archive-api.open-meteo.com/v1/archive"
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        w_params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "daily": ["temperature_2m_max", "precipitation_sum", "cloud_cover_mean", "wind_speed_10m_max"],
            "timezone": "auto"}
        
        aq_params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "hourly": ["pm2_5", "ozone"],# PM2_5 is the main indicator of atmospheric haze. Particulate matter particles with a diameter of 2.5 micrometers or less
            "timezone": "auto"}
        
        try:
            #Get the Response objects
            w_res = requests.get(w_url, params=w_params)
            aq_res = requests.get(aq_url, params=aq_params)
            
            #Check the status codes on the Response objects
            if w_res.status_code == 200:
                #Convert to json dictionaries
                w_data = w_res.json()
                aq_data = aq_res.json()
                
                #Conver the JSON into a DataFrame
                df_w = pd.DataFrame(w_data['daily'])
                
                #Check if Air Quality data exists in the response
                if 'hourly' in aq_data:
                    #Convert the JSON into a DataFrame
                    df_aq = pd.DataFrame(aq_data['hourly'])
                    #Group into a daily data
                    df_aq['date'] = pd.to_datetime(df_aq['time']).dt.date
                    
                    #Put pm2_5 into the column name in the response
                    if 'pm2_5' in df_aq.columns:
                        daily_pm = df_aq.groupby('date')['pm2_5'].mean().reset_index()
                        df_w['pm2_5'] = daily_pm['pm2_5']
                        
                    #If no enough data on the air quality default it to "10.0" which is equal clean air    
                    else:
                        df_w['pm2_5'] = 10.0
                        
                    #Check if Ozone data exists in the response
                    if 'ozone' in df_aq.columns:
                        #Create the variable daily_ozone with time and ozone data
                        daily_ozone = df_aq.groupby('date')['ozone'].mean().reset_index()
                        
                        #Store the averages only
                        df_w['ozone'] = daily_ozone['ozone']
                    else:
                        df_w['ozone'] = 300.0
                        
                #If the 'hourly' is missing from the API
                else:
                    df_w['pm2_5'] = 10.0
                    df_w['ozone'] = 300.0
                
                
                # Add to the list as dataframes
                all_years.append(df_w)
                
        #Catch any errors to prevent crashing
        except Exception as e:
            print(f"Error fetching data for {year}: {e}")
            
    #If the list is empty then notify 
    if not all_years:
        return "NO WEATHER DATA WAS COLLECTED!"
    
    #Data Processing
    
    #Combine all DataFrames row-wise
    hist_weather_info = pd.concat(all_years)
    
    #Convert into python date time object and the focus is on the number (day) of the month
    hist_weather_info['day'] = pd.to_datetime(hist_weather_info['time']).dt.day
    
    #Find the average based on the number (day) of the month
    daily_avg = hist_weather_info.groupby('day').mean(numeric_only=True).reset_index() 

    #Weather Scoring + Fortune Cookie/Vibe Formula
    def calc_score(row):
        #Cloudiness (70% weight) - remove from 100% because the lower amount is better
        c_score = 100 - row['cloud_cover_mean']
        
        #Temperature (30% weight) - Gaussian Bell Curve
        #The number "5" was used for standard deviation
        t_score = np.exp(-((row['temperature_2m_max'] - ideal_t)**2) / (2 * 5**2)) * 100
        
        #Higher Ozone (Chappuis Effect) enhances purple/blue hues.
        # Divide it by 350 because the layer is thick enough to act as a filter for the sun's rays as they pass sideways
        # Ozone layer can not contribute more than 10 points
        color_bonus = min(10, (row['ozone'] / 350) * 10)
        
        #The multipliers (the penalties)
        r_mult = 1.0 if row['precipitation_sum'] < 0.1 else max(0, 1 - (row['precipitation_sum'] / 3))
        w_mult = 1.0 if row['wind_speed_10m_max'] < 12 else max(0.1, 1 - (row['wind_speed_10m_max'] / 18))
        #Ignore pollution until it crosses 10 then penalize
        p_mult = 1.0 if row['pm2_5'] < 10 else max(0, 1 - (row['pm2_5'] - 10) / 40)
        
        #The Final Score Calculation
        final_score = ((c_score * 0.7) + (t_score * 0.3) + color_bonus) * r_mult * w_mult * p_mult


        #Fortune Cookie-like Addition
        if r_mult < 0.9:
            vibe = "💧 WATER | Trust the flow; clarity comes after the soak."
        elif w_mult < 0.9:
            vibe = "🌬️ AIR | Fresh perspectives are heading your way."
        elif color_bonus > 8.5:
            vibe = "🩷 Cotton-candy sky | The ozone is electric. Expect majestic sky hues."
        elif final_score > 80:
            vibe = "✨ ETHER | THE Perfect Match. The sky is cosmically aligned."
        elif t_score > 80:
            vibe = "🔥 FIRE | Fortune favors the bold—make that big move today."
        else:
            vibe = "🌿 EARTH | Grounded and stable. A day for steady progress."
       
        #Return the Calculated Numbers
        return pd.Series({
            'score': round(final_score, 1),
            'vibe': vibe
        })
    # Apply calc_score to each row
    new_cols = daily_avg.apply(calc_score, axis=1)  # returns DataFrame with 'score' & 'vibe'
    full_results = pd.concat([daily_avg, new_cols], axis=1)

    # Define which columns we want to see
    display_columns = ['day', 'score', 'vibe']


# Now you can safely select it
    top_3 = full_results.sort_values(by='score', ascending=False)[
    ['day', 'score', 'vibe']
    ].head(3)

    print(top_3)


#Execute the formula
#today = datetime.now()
#next_month = today.month + 1 if today.month < 12 else 1
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
result = get_romantic_weather_prediction("Bristol, UK", 4, 2026)
# Print it so you can see it in the Spyder console
print(result)





