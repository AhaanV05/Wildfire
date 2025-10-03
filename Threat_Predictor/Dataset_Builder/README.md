# Wildfire Threat Predictor - Dataset Builder

This directory contains tools for building a historical wildfire dataset by combining:
- Fire detection data (FIRMS/MODIS/VIIRS) → Rate of Spread (ROS) targets
- Weather data (NASA POWER) → Temperature, humidity, wind, precipitation, FWI
- Vegetation data (MODIS/GEE) → NDVI, NDMI 
- Moisture data (Copernicus SWI) → Live Fuel Moisture Content proxy
- Terrain data → Elevation, slope, aspect

## Quick Start

1. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials and settings
   ```

2. **Setup database**:
   ```bash
   psql -U postgres -f Pg_schema.sql
   ```

3. **Test integration**:
   ```bash
   python test_integration.py
   ```

4. **Ingest fire data** (creates ROS targets):
   ```bash
   python ingest_ros_targets.py --help
   # Example:
   python ingest_ros_targets.py --pg-dsn "dbname=wildfire user=postgres password=xxx" --paths "data/MODIS_C6_Global_*.csv" --date-from 2020-01-01 --date-to 2020-12-31
   ```

5. **Build features** (historical backfill):
   ```bash
   python dataset_build_feature.py --help
   # Example:
   python dataset_build_feature.py --start 2020-01-01 --end 2020-12-31
   ```

## Files

- `Pg_schema.sql` - Database schema (ros_targets + features_daily tables)
- `config.py` - Configuration loader (reads from .env)
- `ingest_ros_targets.py` - Process fire detection data → ROS targets
- `dataset_build_feature.py` - Build feature dataset from multiple data sources  
- `cdse_swi.py` - Copernicus Data Space Ecosystem SWI data fetcher
- `test_integration.py` - Integration test suite

## Data Sources

### Weather (NASA POWER)
- **Variables**: Temperature, humidity, wind speed, precipitation
- **Resolution**: Daily, 0.5° × 0.625°
- **Derived**: VPD, FWI, days since rain
- **API**: Free, no authentication required

### Vegetation (MODIS via Google Earth Engine) 
- **Variables**: NDVI, NDMI
- **Resolution**: 8-day composite, 500m
- **Source**: MOD09A1 Surface Reflectance
- **Auth**: Requires GEE account (set USE_GEE_VEG=true)

### Moisture (Copernicus SWI)
- **Variables**: Soil Water Index → LFMC proxy
- **Resolution**: Daily, 12.5km
- **Source**: CGLS SWI v3 
- **Auth**: Requires CDSE account (set USE_COPERNICUS_SWI=true)

### Terrain (Placeholder)
- **Variables**: Elevation, slope, aspect
- **Note**: Currently returns None - integrate your preferred DEM source

## Configuration

Key settings in `.env`:

```bash
# Database
PG_DSN=dbname=wildfire user=postgres password=*** host=localhost

# Data source toggles
USE_GEE_VEG=false          # Google Earth Engine vegetation 
USE_COPERNICUS_SWI=true    # Copernicus SWI moisture data

# CDSE credentials (if USE_COPERNICUS_SWI=true)
CDSE_USERNAME=your_email
CDSE_PASSWORD=your_password
```

## Pipeline Workflow

1. **Fire Detection → ROS Targets**:
   - Load FIRMS CSV data
   - Cluster detections spatially (DBSCAN)
   - Link clusters temporally across days  
   - Compute Rate of Spread from spatial progression
   - Store in `ros_targets` table

2. **Feature Engineering**:
   - For each (fire_id, date) in ros_targets:
     - Fetch weather data (NASA POWER)
     - Fetch vegetation data (MODIS/GEE) 
     - Fetch moisture data (SWI)
     - Compute derived features (VPD, FWI, etc.)
     - Store in `features_daily` table

3. **Training Data**:
   - Join `features_daily` with `ros_targets` on (fire_id, date)
   - Features (X): weather, vegetation, moisture, terrain, temporal
   - Target (y): target_ros_m_min

## Troubleshooting

- **CDSE Authentication**: Register at https://dataspace.copernicus.eu/
- **GEE Authentication**: Run `earthengine authenticate` first
- **Database Errors**: Check PG_DSN format and PostgreSQL is running
- **Memory Issues**: Reduce batch sizes or date ranges
- **API Rate Limits**: Built-in retry logic, but increase delays if needed

## Performance Tips

- Use date ranges for historical backfill (`--start`, `--end`)
- Enable CSV catalogue files for faster SWI lookups
- Cache directories avoid redundant API calls
- Batch processing minimizes database round-trips
- Consider running overnight for large date ranges