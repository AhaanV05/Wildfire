"""
OpenTopography Terrain Data Module

Fetches DEM data from OpenTopography API and computes terrain features:
- Elevation (m)
- Slope (%)
- Aspect (degrees)

Uses NASADEM 30m data with tile-based caching for efficiency.
"""

import os
import math
import logging
import requests
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from typing import Tuple, Optional
from pathlib import Path

from config import (
    OPENTOPO_API_KEY, 
    OPENTOPO_BASE_URL,
    TERRAIN_DEM_TYPE,
    TERRAIN_TILE_SIZE,
    CACHE_DIR
)

log = logging.getLogger(__name__)

# Cache directory for terrain tiles
TERRAIN_CACHE_DIR = Path(CACHE_DIR) / "terrain"
TERRAIN_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_tile_bounds(lat: float, lon: float, tile_size: float = None) -> Tuple[float, float, float, float]:
    """
    Get tile bounds for a given lat/lon coordinate.
    
    Args:
        lat: Latitude
        lon: Longitude  
        tile_size: Tile size in degrees (default from config)
        
    Returns:
        (south, north, west, east) bounds
    """
    if tile_size is None:
        tile_size = TERRAIN_TILE_SIZE
        
    # Snap to tile grid
    south = math.floor(lat / tile_size) * tile_size
    north = south + tile_size
    west = math.floor(lon / tile_size) * tile_size
    east = west + tile_size
    
    return south, north, west, east


def _get_tile_cache_path(south: float, north: float, west: float, east: float) -> Path:
    """Get cache file path for a tile."""
    filename = f"{TERRAIN_DEM_TYPE}_{south:.1f}_{north:.1f}_{west:.1f}_{east:.1f}.tif"
    return TERRAIN_CACHE_DIR / filename


def _download_dem_tile(south: float, north: float, west: float, east: float) -> Path:
    """
    Download DEM tile from OpenTopography API.
    
    Args:
        south, north, west, east: Tile bounds in WGS84 degrees
        
    Returns:
        Path to cached GeoTIFF file
        
    Raises:
        requests.RequestException: If download fails
    """
    cache_path = _get_tile_cache_path(south, north, west, east)
    
    # Return cached file if exists
    if cache_path.exists():
        log.debug(f"Using cached DEM tile: {cache_path}")
        return cache_path
    
    # Download from OpenTopography
    url = f"{OPENTOPO_BASE_URL}/globaldem"
    params = {
        "demtype": TERRAIN_DEM_TYPE,
        "south": south,
        "north": north, 
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY
    }
    
    log.info(f"Downloading DEM tile: {south:.1f}-{north:.1f}, {west:.1f}-{east:.1f}")
    
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    
    if response.status_code == 204:
        raise ValueError(f"No DEM data available for bounds: {south}, {north}, {west}, {east}")
    
    # Save to cache
    with open(cache_path, 'wb') as f:
        f.write(response.content)
    
    log.info(f"Cached DEM tile: {cache_path}")
    return cache_path


def _compute_slope_aspect(elevation_array: np.ndarray, 
                         transform: rasterio.Affine) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect from elevation using Horn's method.
    
    Args:
        elevation_array: 2D elevation array
        transform: Rasterio affine transform for pixel size
        
    Returns:
        (slope_percent, aspect_degrees) arrays
    """
    # Get pixel size in meters (approximate for lat/lon)
    pixel_size_x = abs(transform[0]) * 111000  # ~111km per degree at equator
    pixel_size_y = abs(transform[4]) * 111000
    
    # Pad elevation array to handle edges
    padded = np.pad(elevation_array, 1, mode='edge')
    
    # Horn's method: compute gradients using 3x3 neighborhood
    rows, cols = elevation_array.shape
    slope = np.zeros((rows, cols), dtype=np.float32)
    aspect = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # 3x3 window around pixel
            z1, z2, z3 = padded[i-1, j-1], padded[i-1, j], padded[i-1, j+1]
            z4, z5, z6 = padded[i, j-1],   padded[i, j],   padded[i, j+1]
            z7, z8, z9 = padded[i+1, j-1], padded[i+1, j], padded[i+1, j+1]
            
            # Gradients (Horn's method)
            dz_dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * pixel_size_x)
            dz_dy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * pixel_size_y)
            
            # Slope in radians, then convert to percent
            slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
            slope[i-1, j-1] = math.tan(slope_rad) * 100
            
            # Aspect in radians, then convert to degrees (0-360, N=0)
            if dz_dx == 0 and dz_dy == 0:
                aspect[i-1, j-1] = -1  # Flat area
            else:
                aspect_rad = math.atan2(dz_dy, -dz_dx)
                aspect_deg = math.degrees(aspect_rad)
                # Convert to geographic aspect (0=North, 90=East, 180=South, 270=West)
                aspect_deg = 90 - aspect_deg
                if aspect_deg < 0:
                    aspect_deg += 360
                aspect[i-1, j-1] = aspect_deg
    
    return slope, aspect


def _sample_raster_at_point(raster_path: Path, lat: float, lon: float) -> Tuple[float, float, float]:
    """
    Sample elevation, slope, and aspect at a specific lat/lon point.
    
    Args:
        raster_path: Path to GeoTIFF file
        lat: Latitude 
        lon: Longitude
        
    Returns:
        (elevation, slope_percent, aspect_degrees)
    """
    with rasterio.open(raster_path) as src:
        # Read elevation
        elevation = src.read(1)
        
        # Get pixel coordinates
        row, col = src.index(lon, lat)
        
        # Check bounds
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(f"Point ({lat}, {lon}) outside raster bounds")
        
        # Get elevation value
        elev_value = float(elevation[row, col])
        
        # Compute slope and aspect for entire tile (cached computation)
        slope, aspect = _compute_slope_aspect(elevation, src.transform)
        
        slope_value = float(slope[row, col])
        aspect_value = float(aspect[row, col])
        
        return elev_value, slope_value, aspect_value


def lookup_terrain(lat: float, lon: float) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Lookup terrain features (elevation, slope, aspect) for a lat/lon point.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        (elevation_m, slope_pct, aspect_deg, source) or (None, None, None, None) if failed
    """
    try:
        # Get tile bounds
        south, north, west, east = _get_tile_bounds(lat, lon)
        
        # Download/get cached tile
        tile_path = _download_dem_tile(south, north, west, east)
        
        # Sample at point
        elevation, slope, aspect = _sample_raster_at_point(tile_path, lat, lon)
        
        source = f"OpenTopo_{TERRAIN_DEM_TYPE}"
        
        return elevation, slope, aspect, source
        
    except Exception as e:
        log.warning(f"Terrain lookup failed for ({lat}, {lon}): {e}")
        return None, None, None, None


def test_terrain_lookup():
    """Test terrain lookup for a known location."""
    # Test location: Mount Whitney, CA (highest point in contiguous US)
    lat, lon = 36.5786, -118.2926
    
    print(f"Testing terrain lookup for Mount Whitney ({lat}, {lon})...")
    
    elevation, slope, aspect, source = lookup_terrain(lat, lon)
    
    if elevation is not None:
        print(f"✅ Elevation: {elevation:.1f} m")
        print(f"✅ Slope: {slope:.1f}%") 
        print(f"✅ Aspect: {aspect:.1f}°")
        print(f"✅ Source: {source}")
    else:
        print("❌ Terrain lookup failed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_terrain_lookup()