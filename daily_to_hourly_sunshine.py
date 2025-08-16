
import numpy as np
import pandas as pd
from math import radians, pi

GSC = 0.0820  # MJ m^-2 min^-1

def day_of_year(dt: pd.Timestamp) -> int:
    return int(dt.day_of_year)

def ra_extraterrestrial(lat_deg: float, J: int) -> float:
    phi = radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * J / 365.0 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    Ra = (24 * 60 / np.pi) * GSC * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return float(Ra)

def possible_sunshine_hours(lat_deg: float, J: int) -> float:
    phi = radians(lat_deg)
    delta = 0.409 * np.sin(2 * np.pi * J / 365.0 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    return float(24.0 / np.pi * ws)

def hargreaves_rs(Ra_MJ, tmax, tmin, krs=0.19):
    dtr = max(0.0, tmax - tmin)
    return krs * np.sqrt(dtr) * Ra_MJ

def angstrom_sunshine_hours(Rs_MJ, Ra_MJ, S0, a=0.25, b=0.5):
    if Ra_MJ <= 0 or S0 <= 0:
        return 0.0
    S = S0 * (Rs_MJ / Ra_MJ - a) / b
    return float(max(0.0, min(S, S0)))

def solar_elevation_weight(lat_deg: float, lon_deg: float, dt_local: pd.Timestamp, tz_offset_hours: float) -> float:
    lst_correction = (lon_deg / 15.0) - tz_offset_hours
    t_local = dt_local + pd.Timedelta(hours=lst_correction)
    J = day_of_year(dt_local)
    phi = radians(lat_deg)
    delta = 0.409 * np.sin(2 * np.pi * J / 365.0 - 1.39)
    h = (t_local.hour + 0.5)
    omega = np.radians(15.0 * (h - 12.0))
    sin_alpha = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
    return max(0.0, float(sin_alpha))

def distribute_sunshine_hourly(date, S_daily, lat, lon, tz_offset_hours=3.0):
    weights = []
    for h in range(24):
        dt = pd.Timestamp(date) + pd.Timedelta(hours=h)
        lst_correction = (lon / 15.0) - tz_offset_hours
        t_local = dt + pd.Timedelta(hours=lst_correction)
        J = day_of_year(dt)
        phi = np.radians(lat)
        delta = 0.409 * np.sin(2 * np.pi * J / 365.0 - 1.39)
        hh = (t_local.hour + 0.5)
        omega = np.radians(15.0 * (hh - 12.0))
        sin_alpha = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
        weights.append(max(0.0, float(sin_alpha)))
    weights = np.array(weights, dtype=float)
    daylight_mask = weights > 0
    if daylight_mask.sum() == 0:
        return [0.0]*24
    weights_norm = weights[daylight_mask] / weights[daylight_mask].sum()
    tsun = np.zeros(24, dtype=float)
    tsun[daylight_mask] = S_daily * weights_norm
    # Cap at 1 hour per clock hour
    tsun = np.minimum(tsun, 1.0)
    # (Optional) could redistribute leftover here if needed
    return tsun.tolist()

def daily_to_hourly_tsun(row, lat, lon, tz_offset_hours=3.0, a=0.25, b=0.5, krs=0.19):
    date = pd.Timestamp(row['date'])
    J = day_of_year(date)
    Ra = ra_extraterrestrial(lat, J)
    S0 = possible_sunshine_hours(lat, J)
    tmin = float(row['tmin'])
    tmax = float(row['tmax'])
    Rs = hargreaves_rs(Ra, tmax, tmin, krs=krs)
    S_daily = angstrom_sunshine_hours(Rs, Ra, S0, a=a, b=b)
    tsun = distribute_sunshine_hourly(date, S_daily, lat, lon, tz_offset_hours=tz_offset_hours)
    return tsun
