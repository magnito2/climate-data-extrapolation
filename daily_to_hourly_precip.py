import numpy as np
import pandas as pd

def daily_to_hourly_prcp(row, method="diurnal", seed=None,
                         wet_hours=6, duration_hours=6,
                         a=2.5, b=4.0,
                         diurnal_probs=None):
    """
    Disaggregate daily precipitation (mm/day) into hourly (mm/hour).
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ["date", "prcp"] where prcp is daily total in mm.
    method : str
        "uniform"    – spread over N wet hours (randomly placed).
        "diurnal"    – allocate using diurnal weights (default: tropical coastal).
        "single_storm" – single storm of given duration with beta shape.
    seed : int or None
        Random seed for reproducibility.
    wet_hours : int
        For "uniform", number of hours with rain.
    duration_hours : int
        For "single_storm", storm duration in hours.
    a, b : float
        Beta distribution shape parameters for "single_storm".
    diurnal_probs : list or None
        Length-24 list of probabilities (sum=1) for "diurnal". If None, uses a
        tropical profile peaking late afternoon/evening.
    """
    rng = np.random.default_rng(seed)

    if diurnal_probs is None:
        # Default tropical coastal diurnal profile
        # (afternoon/evening convective max, little night rain)
        diurnal_probs = np.array([
            0.01,0.005,0.005,0.005,0.005,0.01,0.015,0.02,
            0.03,0.05,0.06,0.08,0.10,0.12,0.12,0.11,
            0.09,0.06,0.035,0.025,0.015,0.01,0.005,0.005
        ])
        diurnal_probs = diurnal_probs / diurnal_probs.sum()

    total = row["prcp"] if row["prcp"] > 0 else 0.0
    hourly = np.zeros(24)

    if total <= 0 or pd.isna(total):
        # No rain
        pass

    elif method == "uniform":
        wet_idx = rng.choice(24, size=wet_hours, replace=False)
        hourly[wet_idx] = total / wet_hours

    elif method == "diurnal":
        weights = rng.dirichlet(diurnal_probs*10)  # add small randomness
        hourly = total * weights

    elif method == "single_storm":
        # Choose random storm start hour
        start_hour = rng.choice(24)
        end_hour = min(start_hour + duration_hours, 24)

        storm_hours = end_hour - start_hour
        if storm_hours > 0:
            x = np.linspace(0, 1, storm_hours)
            shape = (x**(a-1)) * ((1-x)**(b-1))
            shape /= shape.sum()
            hourly[start_hour:end_hour] = total * shape
    
    return hourly.tolist()