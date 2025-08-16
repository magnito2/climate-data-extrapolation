import numpy as np

def saturation_vapor_pressure(T):
    """Saturation vapor pressure over water (hPa), T in °C"""
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))

def dew_point_from_rh(Tavg, RH_mean):
    """Estimate dew point from daily mean T and RH"""
    es = saturation_vapor_pressure(Tavg)
    ea = RH_mean/100 * es
    ln_ratio = np.log(ea/6.112)
    return (243.5 * ln_ratio) / (17.67 - ln_ratio)

def extrapolate_hourly_RH(row):
    """
    Input row with columns: tavg, rh (daily mean %), thour
    Returns hourly dataframe with Temp and RH
    """
    Tavg, RH_mean, T = row["tavg"], row["rh"], row['thour']
        
    # Step 2: dew point from Tavg + RHmean
    Tdew = dew_point_from_rh(Tavg, RH_mean)
        
    # Step 3: hourly RH
    es_T = saturation_vapor_pressure(T)
    es_Td = saturation_vapor_pressure(Tdew)
    RH = 100 * es_Td / es_T
    RH = np.clip(RH, 0, 100)
        
    return RH
