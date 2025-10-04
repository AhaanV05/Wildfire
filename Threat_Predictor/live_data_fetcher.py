#!/usr/bin/env python3
"""
Simple Live Data Fetcher for Wildfire Prediction
Just gets live weather data and returns features.
"""

import requests
import math
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    print("⚠️ Google Earth Engine not available")

# Load environment variables
load_dotenv()

def get_live_features(lat, lon):
    """
    Simple function to get live weather features for a location.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Weather features ready for prediction
    """
    print(f"🌍 Fetching live data for ({lat}, {lon})...")
    
    # Get weather from NASA POWER API
    weather = _get_nasa_weather(lat, lon)
    
    # Get terrain from OpenTopography API
    terrain = _get_opentopo_terrain(lat, lon)
    
    # Get vegetation from Google Earth Engine
    vegetation = _get_gee_vegetation(lat, lon)
    
    # Calculate derived features
    features = {
        'temp_c': weather['temp_c'],
        'rel_humidity_pct': weather['rel_humidity_pct'], 
        'wind_speed_ms': weather['wind_speed_ms'],
        'precip_mm': weather['precip_mm'],
        'vpd_kpa': _calculate_vpd_from_dewpoint(weather['temp_c'], weather['dew_point_c']),
        'fwi': _calculate_fwi(weather['temp_c'], weather['rel_humidity_pct'], 
                             weather['wind_speed_ms'], weather['precip_mm']),
        'ndvi': vegetation['ndvi'],
        'ndmi': vegetation['ndmi'],
        'lfmc_proxy_pct': vegetation['lfmc_proxy_pct'],
        'elevation_m': terrain['elevation_m'],
        'slope_pct': terrain['slope_pct'],
        'aspect_deg': terrain['aspect_deg']
    }
    
    print(f"✅ Got features: Temp={features['temp_c']:.1f}°C, "
          f"Wind={features['wind_speed_ms']:.1f}m/s, "
          f"Humidity={features['rel_humidity_pct']:.1f}%, "
          f"VPD={features['vpd_kpa']:.3f}kPa, "
          f"Elevation={features['elevation_m']:.0f}m")
    
    return features

def _get_nasa_weather(lat, lon):
    """Get weather data from NASA POWER API."""
    print("🛰️ Fetching weather from NASA POWER...")
    
    # Get data from last 7 days to find valid values
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    
    url = os.getenv('POWER_BASE', 'https://power.larc.nasa.gov/api/temporal/daily/point')
    params = {
        'parameters': 'T2M,RH2M,WS2M,PRECTOTCORR,T2MDEW,PS',
        'community': os.getenv('POWER_COMMUNITY', 'AG'),
        'longitude': lon,
        'latitude': lat,
        'start': week_ago.strftime("%Y%m%d"),
        'end': today.strftime("%Y%m%d"),
        'format': 'JSON'
    }
    
    response = requests.get(url, params=params, timeout=15)
    
    if response.status_code == 200:
        data = response.json()
        props = data['properties']['parameter']
        
        # Find the most recent valid data (not -999)
        for i in range(7):  # Check last 7 days
            date_key = (today - timedelta(days=i)).strftime("%Y%m%d")
            
            temp = props['T2M'].get(date_key)
            humidity = props['RH2M'].get(date_key)
            wind = props['WS2M'].get(date_key)
            precip = props['PRECTOTCORR'].get(date_key)
            dew_point = props['T2MDEW'].get(date_key)
            pressure = props['PS'].get(date_key)
            
            # Check if values are valid (not -999 or None)
            if (temp is not None and temp != -999 and 
                humidity is not None and humidity != -999 and
                wind is not None and wind != -999 and
                precip is not None and precip != -999 and
                dew_point is not None and dew_point != -999 and
                pressure is not None and pressure != -999):
                
                weather = {
                    'temp_c': temp,
                    'rel_humidity_pct': humidity,
                    'wind_speed_ms': wind,
                    'precip_mm': precip,
                    'dew_point_c': dew_point,
                    'pressure_kpa': pressure
                }
                print(f"✅ NASA POWER data retrieved (from {date_key}) - includes dew point & pressure")
                return weather
    
    raise Exception(f"NASA POWER API failed: status {response.status_code} or no valid data found")

def _get_opentopo_terrain(lat, lon):
    """Get terrain data from OpenTopography API."""
    print("🏔️ Fetching terrain from OpenTopography...")
    
    # Use OpenTopoData API for elevation
    elevation_url = "https://api.opentopodata.org/v1/srtm30m"
    elevation_params = {'locations': f"{lat},{lon}"}
    
    elevation_response = requests.get(elevation_url, params=elevation_params, timeout=10)
    
    if elevation_response.status_code == 200:
        elevation_data = elevation_response.json()
        elevation = elevation_data['results'][0]['elevation']
        
        if elevation is not None:
            # Calculate slope and aspect using nearby points
            slope, aspect = _calculate_slope_and_aspect(lat, lon)
            
            terrain = {
                'elevation_m': float(elevation),
                'slope_pct': slope,
                'aspect_deg': aspect
            }
            
            print(f"✅ OpenTopography elevation: {elevation}m, slope: {slope:.1f}%, aspect: {aspect:.1f}°")
            return terrain
    
    raise Exception(f"OpenTopography API failed: status {elevation_response.status_code}")

def _calculate_slope_and_aspect(lat, lon):
    """Calculate slope and aspect from real elevation differences."""
    delta = 0.002  # ~200m spacing
    
    # Get elevations for 4 cardinal points
    points = [
        f"{lat + delta},{lon}",    # North
        f"{lat - delta},{lon}",    # South
        f"{lat},{lon + delta}",    # East
        f"{lat},{lon - delta}"     # West
    ]
    
    locations = "|".join(points)
    url = "https://api.opentopodata.org/v1/srtm30m"
    params = {'locations': locations}
    
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        elevations = [r['elevation'] for r in data['results']]
        
        # Check if all elevations are valid
        if all(e is not None for e in elevations):
            north, south, east, west = elevations
            
            # Calculate gradients (rise/run)
            dz_dy = (north - south) / (2 * delta * 111000)  # N-S gradient (m/m)
            dz_dx = (east - west) / (2 * delta * 111000 * math.cos(math.radians(lat)))  # E-W gradient (m/m)
            
            # Calculate slope as percentage
            slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
            slope_pct = math.tan(slope_rad) * 100
            slope_pct = min(max(slope_pct, 0), 100)  # Clamp between 0-100%
            
            # Calculate aspect (direction of steepest descent)
            # Aspect is measured clockwise from north (0°)
            if dz_dx == 0 and dz_dy == 0:
                aspect_deg = 0.0  # Flat terrain, no aspect
            else:
                aspect_rad = math.atan2(-dz_dx, dz_dy)  # Note: -dz_dx for proper direction
                aspect_deg = math.degrees(aspect_rad)
                
                # Convert to 0-360° range
                if aspect_deg < 0:
                    aspect_deg += 360.0
            
            return slope_pct, aspect_deg
    
    raise Exception(f"Slope and aspect calculation failed: status {response.status_code}")

def _get_gee_vegetation(lat, lon):
    """Get vegetation data from Google Earth Engine."""
    if not GEE_AVAILABLE:
        raise Exception("Google Earth Engine not available - install earthengine-api")
    
    print("🛰️ Fetching vegetation from Google Earth Engine...")
    
    # Initialize GEE
    ee.Initialize(project=os.getenv('GEE_PROJECT'))
    
    # Create point geometry
    point = ee.Geometry.Point([lon, lat])
    
    # Get recent MODIS data (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # MODIS Terra Surface Reflectance
    collection = ee.ImageCollection('MODIS/061/MOD09A1') \
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
        .filterBounds(point)
    
    if collection.size().getInfo() > 0:
        # Get most recent image
        image = collection.first()
        
        # Calculate NDVI (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
        
        # Calculate NDMI (NIR - SWIR) / (NIR + SWIR)
        ndmi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b06'])
        
        # Combine bands
        vegetation_image = ndvi.addBands(ndmi).rename(['ndvi', 'ndmi'])
        
        # Sample at point location
        sample = vegetation_image.sample(point, 500).first()
        
        ndvi_val = sample.get('ndvi').getInfo()
        ndmi_val = sample.get('ndmi').getInfo()
        
        if ndvi_val is not None and ndmi_val is not None:
            # Calculate fuel moisture proxy
            lfmc_proxy = max(0, min(200, 200 * ndmi_val + 50 * (ndvi_val - 0.3)))
            
            vegetation = {
                'ndvi': float(ndvi_val),
                'ndmi': float(ndmi_val),
                'lfmc_proxy_pct': float(lfmc_proxy)
            }
            
            print(f"✅ GEE vegetation: NDVI={ndvi_val:.3f}, NDMI={ndmi_val:.3f}")
            return vegetation
    
    raise Exception("No recent MODIS data available for this location")

def _calculate_vpd_from_dewpoint(temp_c, dew_point_c):
    """Calculate Vapour Pressure Deficit (kPa) from temperature and dew point."""
    # Saturation vapor pressure at air temperature
    es = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    # Actual vapor pressure at dew point temperature  
    ea = 0.6108 * math.exp((17.27 * dew_point_c) / (dew_point_c + 237.3))
    return es - ea

def _calculate_fwi(temp_c, rh_pct, wind_ms, precip_mm):
    """Calculate Fire Weather Index."""
    return 0.1 * math.exp(0.05039 * temp_c) * (100 - rh_pct) * wind_ms * math.exp(-0.3 * precip_mm)

# Test the function
if __name__ == "__main__":
    # Test with Delhi coordinates
    test_lat = 19.05822
    test_lon = 72.87781
    
    features = get_live_features(test_lat, test_lon)
    
    print("\n" + "="*50)
    print("🔥 LIVE WEATHER FEATURES")
    print("="*50)
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")