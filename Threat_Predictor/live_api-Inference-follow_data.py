# ============================================================
# 🔥 Wildfire Threat Model – Deterministic Formulas
# Inputs: Temperature, RH, WindSpeed, Precipitation, DaysSinceRain,
#         VPD, FWI, NDVI, NDMI, FuelMoisture, Elevation, Slope, Aspect, ROS, lat, lon
# No other variables allowed
# ============================================================

import math

# ---------- 1. WEATHER DERIVED METRICS ----------

def vapour_pressure_deficit(temp_c, rh_pct):
    """VPD (kPa) from temperature (°C) and relative humidity (%)"""
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    ea = es * (rh_pct / 100.0)
    return es - ea


def fire_weather_index(temp_c, rh_pct, wind_ms, precip_mm):
    """Simplified FWI (dimensionless)"""
    return 0.1 * math.exp(0.05039 * temp_c) * (100 - rh_pct) * wind_ms * math.exp(-0.3 * precip_mm)


# ---------- 2. VEGETATION & FUEL ----------

def fuel_load_from_ndvi(ndvi):
    """Fuel load (kg/m²) from NDVI [0.2–0.8 → 0.3–1.2 kg/m²]"""
    n = min(max(ndvi, 0.2), 0.8)
    return 0.3 + ((n - 0.2) / 0.6) * (1.2 - 0.3)


def fuel_moisture_from_ndvi_ndmi(ndvi, ndmi):
    """Fuel moisture (%) as a linear proxy from NDVI and NDMI"""
    return max(0, min(200, 200 * ndmi + 50 * (ndvi - 0.3)))


# ---------- 3. TERRAIN MODIFIERS ----------

def slope_multiplier(slope_pct):
    """Slope factor for ROS"""
    return min(1 + 0.02 * slope_pct, 3.0)


def aspect_factor(aspect_deg):
    """Dryness boost for sun-facing slopes (135°–225° in N hemisphere)"""
    return 1.2 if 135 <= aspect_deg <= 225 else 1.0


# ---------- 4. FIRE BEHAVIOR PHYSICS ----------

def effective_ros(ros_m_min, slope_pct, aspect_deg):
    """Slope- and aspect-adjusted ROS (m/min)"""
    return ros_m_min * slope_multiplier(slope_pct) * aspect_factor(aspect_deg)


def byram_intensity_kWm(ros_m_min, fuel_load_kg_m2):
    """Fireline intensity (kW/m)"""
    return 1000 * 18.0 * fuel_load_kg_m2 * (ros_m_min / 60.0)


def flame_length_m(intensity_kWm):
    """Flame length (m)"""
    return 0.0775 * (intensity_kWm ** 0.46)


def severity_index(flame_length_m):
    """Severity index scaled 0–1"""
    return min(1.0, flame_length_m / 4.0)


def severity_class(flame_length_m):
    """Severity class label"""
    if flame_length_m < 1.2:
        return "Low"
    elif flame_length_m < 2.4:
        return "Moderate"
    elif flame_length_m < 3.5:
        return "High"
    else:
        return "Extreme"


# ---------- 5. CROWN & SPOTTING ----------

def crown_fire_score(intensity_kWm, wind_ms, ndvi):
    """Crown fire potential score (0–100)"""
    wind_kmh = wind_ms * 3.6
    base = (intensity_kWm / 2000.0) * 20.0
    wind_term = min(30.0, wind_kmh / 2.0)
    canopy_term = 20.0 if ndvi >= 0.6 else 0.0
    return int(min(100.0, base + wind_term + canopy_term))


def crown_fire_class(score):
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Passive"
    else:
        return "Active"


def spotting_distance_km(wind_ms, flame_length_m):
    """Spotting distance (km)"""
    wind_kmh = wind_ms * 3.6
    return 0.02 * wind_kmh + 0.1 * flame_length_m


# ---------- 6. OCCURRENCE × CONSEQUENCE ----------

def expected_threat(occurrence_prob, severity_index):
    """Expected threat fusion metric [0–1]"""
    return occurrence_prob * severity_index


# ---------- 7. CONTAINMENT DIFFICULTY ----------

def containment_difficulty(flame_length_m, slope_pct, distance_to_road_km):
    """Qualitative difficulty class"""
    if flame_length_m < 1.2:
        base = "Easy"
    elif flame_length_m < 2.4:
        base = "Moderate"
    elif flame_length_m < 3.5:
        base = "Hard"
    else:
        base = "Very difficult"

    order = ["Easy", "Moderate", "Hard", "Very difficult"]
    idx = order.index(base)

    if slope_pct > 30:
        idx = min(idx + 1, len(order) - 1)
    if distance_to_road_km > 5:
        idx = min(idx + 1, len(order) - 1)

    return order[idx]


# ---------- 8. DAMAGE & WINDOW TIME ----------

def time_to_burn_window_hours(ros_m_min, area_km2=5.0):
    """
    Time (hours) required to consume a fixed area window (km²)
    using elliptical growth: Area = 0.3*(ROSEff*60*T)^2 / 10,000
    Solve for T when Area = area_km2 * 100 ha.
    """
    A_ha = area_km2 * 100
    k = ros_m_min * 60.0
    T_hours = math.sqrt((A_ha * 10_000) / (0.3 * k ** 2))
    return T_hours


def damage_in_window_rs(ros_m_min, asset_value_per_ha, area_km2=5.0):
    """Damage (Rs) for fixed area window"""
    A_ha = area_km2 * 100
    return A_ha * asset_value_per_ha


# ============================================================
# END OF FORMULAS
# ============================================================
