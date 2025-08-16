import pandas as pd
import numpy as np

from daily_to_hourly_sunshine import daily_to_hourly_tsun
from daily_to_hourly_precip import daily_to_hourly_prcp
from daily_to_hourly_rhum import extrapolate_hourly_RH

# Read data from the CSV file into a DataFrame
data = pd.read_csv('extrapolate_raw_3.csv')

expanded_rows = []
for _, row in data.iterrows():
    date_str = str(row.iloc[0]).split()[0]  # Get date part only

    #Estimate hourly total sunshine using daily_to_hourly_tsun function
    #Your site-specific latitude and longitude
    lat, lon = -4.033, 39.617   # Mombasa example
    tz = 3.0                    # Africa/Nairobi

    if 'tsun' in row:
        tsun_arr = daily_to_hourly_tsun(row[['date','tmin','tmax', 'tsun']], 
                                        lat=lat, 
                                        lon=lon, 
                                        tz_offset_hours=tz)
    else:
        tsun_arr = [0.0] * 24

    # Estimate hourly precipitation using daily_to_hourly_prcp function
    # Ensure 'prcp' column exists in the row
    if 'prcp' in row:
        prcp_arr = daily_to_hourly_prcp(row[['prcp']], method="diurnal", seed=42)
            
    for hour in range(24):
        new_row = row.copy()
        # Format as YYYYMMDDHH
        formatted = pd.to_datetime(date_str, format='%d-%m-%y').strftime('%Y%m%d') + f"{hour:02d}"
        # Parton–Logan Method for hourly temperature extrapolation
        tmin = row['tmin'] if 'tmin' in row else None
        tmax = row['tmax'] if 'tmax' in row else None

        # Round off hourly temperature to 1 decimal place after calculation
        tavg = row['tavg'] if 'tavg' in row else None

        if tmin is not None and tmax is not None:
            # Assume sunrise at 6:00 and sunset at 18:00 for simplicity
            sunrise = 6
            sunset = 18
            if hour < sunrise:
                # Before sunrise: temperature is tmin
                temp_hour = tmin
            elif hour < sunset:
                # Daytime: sinusoidal rise from tmin to tmax
                temp_hour = tmin + (tmax - tmin) * \
                    (np.sin(np.pi * (hour - sunrise) / (sunset - sunrise)))
            else:
                # After sunset: exponential decay towards tmin
                decay_rate = 0.5  # can be tuned
                temp_hour = tmax - (tmax - tmin) * \
                    (1 - np.exp(-decay_rate * (hour - sunset + 1)))
        else:
            temp_hour = tavg  # fallback to daily average

        # Fill in from the hours
        hourly_tsun =  tsun_arr[hour]*100 if len(tsun_arr) > hour else None
        if hourly_tsun is None or np.isnan(hourly_tsun):    
            # If no sunshine data available, set to 0
            hourly_tsun = 0.0
        # Fill in from the hours
        hourly_prcp = prcp_arr[hour] if len(prcp_arr) > hour else 0.0

        # Estimate hourly relative humidity using extrapolate_hourly_RH function
        rh = extrapolate_hourly_RH({
            "tavg": tavg,
            "rh": 79,  # Default to 79% if not available
            "thour": temp_hour
        })

        new_row_dict = {
            'date': formatted,
            'tavg': round(temp_hour, 1) if temp_hour is not None else None,
            'wspd': row['wspd'] if 'wspd' in row else None,
            'tsun': round(hourly_tsun,0),
            'prcp': round(hourly_prcp, 2),
            'rhum': round(rh, 2) if rh is not None else 79
        }
        # Estimate hourly wind speed using normalized multipliers
        wspd_multipliers = [
            0.70, 0.65, 0.65, 0.70, 0.75, 0.80,  # 0–5h
            0.90, 1.00, 1.10, 1.20, 1.30, 1.40,  # 6–11h
            1.45, 1.50, 1.45, 1.40, 1.35, 1.20,  # 12–17h
            1.00, 0.90, 0.85, 0.80, 0.75, 0.70   # 18–23h
        ]
        mean_multiplier = np.mean(wspd_multipliers)
        normalized_multipliers = [m / mean_multiplier for m in wspd_multipliers]
        daily_wspd = row['wspd'] if 'wspd' in row else None
        if daily_wspd is not None:
            new_row_dict['wspd'] = round(daily_wspd * normalized_multipliers[hour], 2)
        expanded_rows.append(new_row_dict)

    
    # Create DataFrame after all rows are processed
    output_data = pd.DataFrame(expanded_rows, columns=['date', 'tavg', 'wspd', 'tsun', 'prcp', 'rhum'])
    
    # Write to CSV once
    # Round off wind speed to 0 decimal places before saving
    output_data['wspd'] = output_data['wspd'].round(0)
    output_data.to_csv('extrapolate_output.csv', index=False)