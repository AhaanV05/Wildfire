# Performance Optimization Guide for Historical Dataset Building

Your pipeline is working perfectly! Here are recommendations to optimize for the 1.3M records:

## Current Performance
- **Rate**: ~10 records per 5-10 minutes 
- **Bottleneck**: SWI API calls (20-60 seconds each)
- **Estimated time for full dataset**: ~200-400 hours

## Optimization Strategies

### 1. Spatial Caching (Biggest Win)
The SWI data has 12.5km resolution, so many nearby fires share the same data:

```python
# Modify dataset_build_feature.py to group by spatial grid
def spatial_grid_key(lat, lon, resolution_km=12.5):
    """Group coordinates into spatial grid cells"""
    grid_deg = resolution_km / 111.0  # ~12.5km ≈ 0.11 degrees
    grid_lat = round(lat / grid_deg) * grid_deg
    grid_lon = round(lon / grid_deg) * grid_deg
    return (grid_lat, grid_lon)

# Cache SWI data by (grid_cell, date) instead of (exact_lat, exact_lon, date)
```

### 2. Batch Processing by Date
Current: Processes records one by one
Better: Process all fires for a single date together

```python
# Group by date first, then process all locations for that date
# This reuses the same NetCDF file for multiple locations
```

### 3. Disable Slow Features Initially
Set these in .env for faster initial build:
```bash
USE_GEE_VEG=false          # Skip vegetation initially  
USE_COPERNICUS_SWI=false   # Skip moisture initially
```
Build core weather features first, add expensive features later.

### 4. Parallel Processing
```bash
# Process different date ranges in parallel
python dataset_build_feature.py --start 2024-10-01 --end 2024-12-31 &
python dataset_build_feature.py --start 2025-01-01 --end 2025-03-31 &
python dataset_build_feature.py --start 2025-04-01 --end 2025-06-30 &
```

### 5. Increase Batch Sizes
```bash
# Current: --batch-size 10 (for testing)
# Production: --batch-size 1000
```

## Recommended Approach

### Phase 1: Core Features (Fast)
```bash
# Disable expensive features
USE_GEE_VEG=false
USE_COPERNICUS_SWI=false

# Build weather + terrain features only
python dataset_build_feature.py --start 2024-10-01 --end 2025-06-30 --batch-size 1000
```
**Estimated time**: 4-8 hours

### Phase 2: Add Moisture (Medium)
```bash
# Enable SWI with spatial optimization
USE_COPERNICUS_SWI=true

# Update existing records with moisture data
python dataset_build_feature.py --start 2024-10-01 --end 2025-06-30 --force-missing-only false
```
**Estimated time**: 20-40 hours

### Phase 3: Add Vegetation (Optional)
```bash
# Enable GEE after authentication
USE_GEE_VEG=true
earthengine authenticate

# Update existing records with vegetation data  
python dataset_build_feature.py --start 2024-10-01 --end 2025-06-30 --force-missing-only false
```

## Quick Wins for Tonight

1. **Update .env**:
```bash
USE_GEE_VEG=false
USE_COPERNICUS_SWI=false  # Disable for now
```

2. **Run core features**:
```bash
python dataset_build_feature.py --start 2024-10-01 --end 2024-10-31 --batch-size 500
```

3. **Monitor progress**:
```sql
SELECT COUNT(*) FROM features_daily;  -- Check progress
SELECT date, COUNT(*) FROM features_daily GROUP BY date ORDER BY date;
```

This will give you a working dataset much faster, then you can add the expensive features incrementally.