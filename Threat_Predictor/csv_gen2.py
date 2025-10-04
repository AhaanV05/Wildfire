#!/usr/bin/env python3
"""
Export Real Feature Data for Synthetic Generation

Extracts specific columns from features_daily table along with corresponding
target_ros_m_min values from ros_targets table to create a CSV for synthetic data generation.
"""

import pandas as pd
import psycopg2
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_real_features():
    """Export real feature data with ROS targets to CSV."""
    
    # Load environment variables
    load_dotenv()
    
    # Database connection parameters from .env file
    DB_PARAMS = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    
    logger.info("🔗 Connecting to database...")
    conn = psycopg2.connect(**DB_PARAMS)
    
    # SQL query to get specific columns with ROS targets
    sql_query = """
    SELECT 
        f.date,
        f.lat,
        f.lon,
        f.temp_c,
        f.rel_humidity_pct,
        f.wind_speed_ms,
        f.precip_mm,
        f.vpd_kpa,
        f.fwi,
        f.ndvi,
        f.ndmi,
        f.lfmc_proxy_pct,
        f.elevation_m,
        f.slope_pct,
        f.aspect_deg,
        r.target_ros_m_min
    FROM features_daily f
    JOIN ros_targets r ON f.fire_id = r.fire_id AND f.date = r.date
    WHERE f.temp_c IS NOT NULL 
      AND f.ndvi IS NOT NULL 
      AND f.elevation_m IS NOT NULL
      AND r.target_ros_m_min IS NOT NULL
    ORDER BY f.date, f.fire_id;
    """
    
    logger.info("📊 Executing query to extract feature data...")
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    
    logger.info(f"✅ Extracted {len(df)} complete records")
    
    # Show basic statistics
    logger.info("📈 Data Statistics:")
    logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"   Lat range: {df['lat'].min():.3f} to {df['lat'].max():.3f}")
    logger.info(f"   Lon range: {df['lon'].min():.3f} to {df['lon'].max():.3f}")
    logger.info(f"   ROS range: {df['target_ros_m_min'].min():.3f} to {df['target_ros_m_min'].max():.3f}")
    logger.info(f"   Temperature range: {df['temp_c'].min():.1f}°C to {df['temp_c'].max():.1f}°C")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning("⚠️  Found null values:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"   {col}: {count} nulls")
    else:
        logger.info("✨ No null values found in extracted data")
    
    # Save to CSV
    output_file = "real_features_with_ros.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"💾 Saved data to {output_file}")
    
    return df

def show_column_distributions(df):
    """Show distributions of key columns for synthetic generation."""
    logger.info("📊 Column Distributions (for synthetic generation reference):")
    
    numeric_cols = [
        'temp_c', 'rel_humidity_pct', 'wind_speed_ms', 'precip_mm',
        'vpd_kpa', 'fwi', 'ndvi', 'ndmi', 'lfmc_proxy_pct',
        'elevation_m', 'slope_pct', 'aspect_deg', 'target_ros_m_min'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            stats = df[col].describe()
            logger.info(f"   {col}:")
            logger.info(f"     Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            logger.info(f"     Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
            logger.info(f"     25%: {stats['25%']:.3f}, 75%: {stats['75%']:.3f}")

if __name__ == "__main__":
    try:
        # Export the real data
        df = export_real_features()
        
        # Show distributions for reference
        show_column_distributions(df)
        
        logger.info("🎯 Ready for synthetic data generation!")
        logger.info("📝 Next steps:")
        logger.info("   1. Use 'real_features_with_ros.csv' as base for synthetic generation")
        logger.info("   2. Generate 2,850 additional synthetic rows")
        logger.info("   3. Combine real + synthetic for 3,000 total rows")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise
