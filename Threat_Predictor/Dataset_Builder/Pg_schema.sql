-- Run this in psql as a superuser (e.g., postgres)
CREATE DATABASE wildfire;

\c wildfire

-- Minimal table for training targets (one row per fire_id + date)
CREATE TABLE IF NOT EXISTS ros_targets (
  fire_id           TEXT NOT NULL,
  date              DATE NOT NULL,
  lat               DOUBLE PRECISION NOT NULL,
  lon               DOUBLE PRECISION NOT NULL,
  target_ros_m_min  DOUBLE PRECISION NOT NULL,
  label_source      TEXT NOT NULL,
  notes             TEXT,
  PRIMARY KEY (fire_id, date)
);

-- Helpful index for time-slicing later (optional)
CREATE INDEX IF NOT EXISTS idx_ros_targets_date ON ros_targets(date);


-- Database: wildfire  (same DB as ros_targets)

-- 1) Ensure ros_targets exists (from your previous setup)
--   ros_targets(fire_id TEXT, date DATE, lat DOUBLE PRECISION, lon DOUBLE PRECISION, ...)

-- 2) Feature table: one row per fire_id + date
CREATE TABLE IF NOT EXISTS features_daily (
  fire_id               TEXT      NOT NULL,
  date                  DATE      NOT NULL,

  -- location (copied from ros_targets for convenience; use FK to enforce alignment)
  lat                   DOUBLE PRECISION NOT NULL,
  lon                   DOUBLE PRECISION NOT NULL,

  -- 🌦 Weather & Atmosphere (daily)
  temp_c                DOUBLE PRECISION,   -- 2m air temperature (°C)
  rel_humidity_pct      DOUBLE PRECISION,
  wind_speed_ms         DOUBLE PRECISION,
  precip_mm             DOUBLE PRECISION,
  vpd_kpa               DOUBLE PRECISION,   -- derived
  fwi                   DOUBLE PRECISION,   -- derived (Canadian FWI)
  -- days_since_rain       INTEGER,            -- derived (≥0)

  -- 🌱 Vegetation & Fuels (closest composite to date)
  ndvi                  DOUBLE PRECISION,
  ndmi                  DOUBLE PRECISION,
  lfmc_proxy_pct        DOUBLE PRECISION,   -- proxy from NDVI/SMAP blend (optional)

  -- 🏔 Terrain (static)
  elevation_m           DOUBLE PRECISION,
  slope_pct             DOUBLE PRECISION,
  aspect_deg            DOUBLE PRECISION,

  -- 🏙 Human influence (static)
  -- dist_road_km          DOUBLE PRECISION,
  -- dist_settlement_km    DOUBLE PRECISION,

  -- 📅 Temporal helpers
  doy                   SMALLINT,           -- 1..366

  -- bookkeeping
  weather_source        TEXT,               -- e.g., 'NASA_POWER'
  veg_source            TEXT,               -- e.g., 'MODIS/VIIRS AppEEARS'
  moisture_source       TEXT,               -- e.g., 'SMAP_NRT'
  terrain_source        TEXT,               -- e.g., 'SRTM_30m'
  -- human_source          TEXT,               -- e.g., 'OSM_2025-08'
  updated_at            TIMESTAMPTZ DEFAULT now(),

  -- relational constraints
  CONSTRAINT features_daily_pk PRIMARY KEY (fire_id, date),
  CONSTRAINT features_daily_fk FOREIGN KEY (fire_id, date)
    REFERENCES ros_targets(fire_id, date) ON DELETE CASCADE
);

-- Helpful indexes for lookups by date or spatial-ish filtering
CREATE INDEX IF NOT EXISTS idx_features_daily_date ON features_daily(date);
CREATE INDEX IF NOT EXISTS idx_features_daily_latlon ON features_daily(lat, lon);

-- Optional: materialized view to pair features with label for training in one select
CREATE MATERIALIZED VIEW IF NOT EXISTS training_ros_features AS
SELECT
  f.*,
  r.target_ros_m_min
FROM features_daily f
JOIN ros_targets r
  ON (f.fire_id = r.fire_id AND f.date = r.date);

-- Refresh helper (run after loading features)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY training_ros_features;
