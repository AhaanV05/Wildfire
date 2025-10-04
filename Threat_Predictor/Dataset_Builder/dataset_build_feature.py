#!/usr/bin/env python3

from __future__ import annotations
import os
import math
import time
import argparse
import datetime as dt
import logging
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

import psycopg2
import psycopg2.extras
import requests

# ---- local config loader (reads .env) ----
from config import (
    PG_DSN,
    POWER_BASE, POWER_COMMUNITY,
    CACHE_DIR,
    USE_GEE_VEG, GEE_PROJECT,
    LOG_LEVEL,
)

from cdse_swi import lfmc_from_swi

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("dataset_build_target")

# ----------------------------
# Tunables (historical backfill)
# ----------------------------
WINDOW_DSR_DAYS = 30           # lookback for Days Since Rain
REQUEST_SLEEP_SEC = 0.3        # politeness to APIs
POWER_PARAMS = ["T2M", "RH2M", "WS2M", "PRECTOT"]

# ----------------------------
# HTTP session with retry
# ----------------------------
def http_session() -> requests.Session:
    s = requests.Session()
    a = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=10, pool_maxsize=50)
    s.mount("http://", a)
    s.mount("https://", a)
    s.headers.update({"User-Agent": "WildfireThreatPredictor/1.0"})
    return s

SESSION = http_session()

# ----------------------------
# Utilities / cache
# ----------------------------
def cache_path(*parts) -> str:
    path = os.path.join(CACHE_DIR or ".cache/wildfire", *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def read_json(path: str):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)

# ----------------------------
# Derived metrics
# ----------------------------
def vpd_kpa(temp_c: float | None, rh_pct: float | None) -> Optional[float]:
    if temp_c is None or rh_pct is None:
        return None
    es = 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))  # kPa
    ea = es * (rh_pct / 100.0)
    return max(es - ea, 0.0)

def fwi_from_daily(temp_c, rh_pct, wind_ms, precip_mm,
                   prev_ffmc=None, prev_dmc=None, prev_dc=None,
                   lat=None, doy=None):
    """
    Compact daily Canadian FWI. Returns (FFMC, DMC, DC, ISI, BUI, FWI).
    Intended for historical sequences; carries minimal state by (lat,lon).
    """
    if None in (temp_c, rh_pct, wind_ms, precip_mm, doy):
        return (None, None, None, None, None, None)

    ffmc = 85.0 if prev_ffmc is None else prev_ffmc
    dmc  = 6.0  if prev_dmc  is None else prev_dmc
    dc   = 15.0 if prev_dc   is None else prev_dc

    # FFMC update
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    if precip_mm > 0.5:
        rf = precip_mm - 0.5
        mo = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        mo = min(mo, 250.0)
    ed = 0.942 * (rh_pct ** 0.679) + (11.0 * math.exp((rh_pct - 100.0) / 10.0)) + 0.18 * (21.1 - temp_c) * (1.0 - math.exp(-0.115 * rh_pct))
    if mo < ed:
        # drying
        ko = 0.424 * (1.0 - ((rh_pct / 100.0) ** 1.7)) + 0.0694 * math.sqrt(max(wind_ms, 0.0)) * (1.0 - (rh_pct / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp_c)
        mo = ed + (mo - ed) * math.exp(-kd)
    else:
        # wetting
        ew = 0.618 * (rh_pct ** 0.753) + (10.0 * math.exp((rh_pct - 100.0) / 10.0)) + 0.18 * (21.1 - temp_c) * (1.0 - math.exp(-0.115 * rh_pct))
        kw = 0.424 * (1.0 - ((100.0 - rh_pct) / 100.0) ** 1.7) + 0.0694 * math.sqrt(max(wind_ms, 0.0)) * (1.0 - ((100.0 - rh_pct) / 100.0) ** 8)
        kw = kw * 0.581 * math.exp(0.0365 * temp_c)
        mo = ew - (ew - mo) * math.exp(-kw)
    ffmc = (59.5 * (250.0 - mo)) / (147.2 + mo)

    # DMC (very compact approx)
    le = (1.894 * (math.sin(math.radians(0.0172 * (doy - 80))) + 1.0) + 0.5)
    if precip_mm > 1.5:
        rw = 0.92 * precip_mm - 1.27
        ra = dmc
        dmc = max(0.0, ra + 1000.0 * rw / (48.77 + ra))
    dmc = (dmc + (0.1 * (temp_c + 1.1) * (100.0 - rh_pct) * le * 0.0001)) * 0.92

    # DC
    if precip_mm > 2.8:
        ra = dc
        rw = 0.83 * precip_mm - 1.27
        try:
            dc = ra - 400.0 * math.log(1.0 + (3.937 * rw) / (ra * 0.92 + 1e-6))
        except ValueError:
            pass
    v = 0.36 * (temp_c + 2.8) + le
    dc = max(0.0, dc + v)

    # ISI/BUI/FWI
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    fw = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / (4.93e7))
    isi = fw * (0.208 * math.exp(0.05039 * max(wind_ms, 0.0)))
    bui = (0.8 * dmc * dc) / (dmc + 0.4 * dc) if (dmc + 0.4 * dc) > 1e-9 else 0.0
    fwi_val = 0.1 * isi * (bui + 0.4) * (1.0 + (2.72 * (isi ** 0.647))) / (1.0 + (bui ** 0.77))
    return (ffmc, dmc, dc, isi, bui, fwi_val)

# ----------------------------
# POWER fetch (historical ranges)
# ----------------------------
def power_daily(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, Dict[str, float]]:
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": POWER_COMMUNITY,
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    key = f"{lat:.4f}_{lon:.4f}_{params['start']}_{params['end']}.json"
    cpath = cache_path("power", key)
    if os.path.exists(cpath):
        return read_json(cpath)

    r = SESSION.get(POWER_BASE, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"POWER {r.status_code}: {r.text[:200]}")
    data = r.json()
    series = data.get("properties", {}).get("parameter", {})
    out: Dict[str, Dict[str, float]] = {}
    for var, d in series.items():
        for ymd, val in d.items():
            out.setdefault(ymd, {})[var] = val
    write_json(cpath, out)
    time.sleep(REQUEST_SLEEP_SEC)
    return out

def days_since_rain_from_series(series: Dict[str, Dict[str, float]], ymd: str) -> Optional[int]:
    keys = sorted(series)
    if ymd not in keys:
        return None
    i = keys.index(ymd)
    cnt = 0
    for j in range(i, -1, -1):
        day_data = series[keys[j]]
        p = day_data.get("PRECTOTCORR") or day_data.get("PRECTOT")
        if p is None:
            return None
        if p > 0.0:
            return cnt
        cnt += 1
    return cnt

# ----------------------------
# GEE: batched vegetation via reduceRegions (MODIS MOD09A1, 8-day SR, 500m)
# ----------------------------
def init_gee():
    if not USE_GEE_VEG:
        return
    import ee
    from config import GEE_PROJECT
    
    # Try to initialize with project if specified
    project = GEE_PROJECT
    if not project:
        # Try to get project from available projects
        try:
            ee.Authenticate()
            projects = ee.data.getProjects()
            if projects:
                project = projects[0]['id']
                print(f"Using Earth Engine project: {project}")
        except Exception as e:
            print(f"Could not get GEE projects: {e}")
            return
    
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        print("Please set GEE_PROJECT in your .env file or run: earthengine authenticate")
        raise

def _gee_image_for_date_mod09a1(date_obj: dt.date):
    import ee
    start = date_obj - dt.timedelta(days=8)
    end   = date_obj + dt.timedelta(days=8)
    ic = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(str(start), str(end))
    img = ic.sort("system:time_start").first()
    return img

def fetch_ndvi_ndmi_batch(points_for_date: List[dict], date_obj: dt.date) -> Dict[str, tuple[Optional[float], Optional[float], str]]:
    """
    Batch-fetch NDVI/NDMI for many points on a single date using GEE reduceRegions.
    Returns: { fire_id: (ndvi, ndmi, "MODIS_MOD09A1_GEE"), ... }
    """
    if not USE_GEE_VEG or not points_for_date:
        return {}

    import ee
    img = _gee_image_for_date_mod09a1(date_obj)
    if img is None:
        return {}

    red  = img.select("sur_refl_b01").multiply(0.0001)
    nir  = img.select("sur_refl_b02").multiply(0.0001)
    swir = img.select("sur_refl_b06").multiply(0.0001)

    ndvi_img = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndmi_img = nir.subtract(swir).divide(nir.add(swir)).rename("NDMI")
    veg_img  = ndvi_img.addBands(ndmi_img)

    feats = []
    for p in points_for_date:
        geom = ee.Geometry.Point([p["lon"], p["lat"]])
        feats.append(ee.Feature(geom, {"fire_id": p["fire_id"]}))
    fc = ee.FeatureCollection(feats)

    result_fc = veg_img.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=500
    )

    try:
        features = result_fc.getInfo().get("features", [])
    except Exception as e:
        log.warning(f"GEE getInfo failed for {date_obj}: {e}")
        return {}

    out: Dict[str, tuple[Optional[float], Optional[float], str]] = {}
    for f in features:
        props = f.get("properties", {})
        fid = props.get("fire_id")
        ndvi = props.get("NDVI")
        ndmi = props.get("NDMI")
        out[fid] = (ndvi, ndmi, "MODIS_MOD09A1_GEE")
    return out

# ----------------------------
# Terrain sampler (OpenTopography)
# ----------------------------
def lookup_terrain(lat: float, lon: float) -> tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Lookup terrain features using OpenTopography API.
    Returns (elevation_m, slope_pct, aspect_deg, source).
    """
    try:
        from opentopo_terrain import lookup_terrain as _lookup_terrain
        return _lookup_terrain(lat, lon)
    except Exception as e:
        log.warning(f"Terrain lookup failed for ({lat}, {lon}): {e}")
        return (None, None, None, None)

# ----------------------------
# DB ops
# ----------------------------
def get_missing_targets(conn, start: dt.date | None = None, end: dt.date | None = None, only_missing: bool = True, limit: int | None = None):
    """
    Returns list of (fire_id, date, lat, lon) to process from ros_targets.
    
    Args:
        conn: Database connection
        start: Optional start date filter
        end: Optional end date filter  
        only_missing: If True, only get records not in features_daily
        limit: Optional limit on number of records to return
        
    Returns:
        List of (fire_id, date, lat, lon) tuples
    """
    with conn.cursor() as cur:
        if only_missing:
            # Get ros_targets records that don't have corresponding features_daily entries
            sql = """
                SELECT r.fire_id, r.date, r.lat, r.lon
                FROM ros_targets r
                LEFT JOIN features_daily f
                  ON f.fire_id = r.fire_id AND f.date = r.date
                WHERE f.fire_id IS NULL
            """
            params = {}
            
            if start:
                sql += " AND r.date >= %(start)s"
                params["start"] = start
            if end:
                sql += " AND r.date <= %(end)s" 
                params["end"] = end
                
            sql += " ORDER BY r.date, r.fire_id"
            
            if limit:
                sql += f" LIMIT {limit}"
                
            cur.execute(sql, params)
        else:
            # Get all ros_targets records (optionally filtered by date)
            sql = "SELECT fire_id, date, lat, lon FROM ros_targets WHERE 1=1"
            params = {}
            
            if start:
                sql += " AND date >= %(start)s"
                params["start"] = start
            if end:
                sql += " AND date <= %(end)s"
                params["end"] = end
                
            sql += " ORDER BY date, fire_id"
            
            cur.execute(sql, params)
            
        return cur.fetchall()

def upsert_features(conn, rows: list[dict]):
    if not rows:
        return
    # Removed human influence columns
    cols = [
        "fire_id","date","lat","lon",
        "temp_c","rel_humidity_pct","wind_speed_ms","precip_mm","vpd_kpa","fwi","days_since_rain",
        "ndvi","ndmi","lfmc_proxy_pct",
        "elevation_m","slope_pct","aspect_deg",
        "doy","weather_source","veg_source","moisture_source","terrain_source"
    ]
    tpl = "(" + ",".join(["%s"]*len(cols)) + ")"
    sql = f"""
        INSERT INTO features_daily ({",".join(cols)})
        VALUES {tpl}
        ON CONFLICT (fire_id, date) DO UPDATE SET
          lat = EXCLUDED.lat,
          lon = EXCLUDED.lon,
          temp_c = EXCLUDED.temp_c,
          rel_humidity_pct = EXCLUDED.rel_humidity_pct,
          wind_speed_ms = EXCLUDED.wind_speed_ms,
          precip_mm = EXCLUDED.precip_mm,
          vpd_kpa = EXCLUDED.vpd_kpa,
          fwi = EXCLUDED.fwi,
          days_since_rain = EXCLUDED.days_since_rain,
          ndvi = COALESCE(EXCLUDED.ndvi, features_daily.ndvi),
          ndmi = COALESCE(EXCLUDED.ndmi, features_daily.ndmi),
          lfmc_proxy_pct = COALESCE(EXCLUDED.lfmc_proxy_pct, features_daily.lfmc_proxy_pct),
          elevation_m = COALESCE(EXCLUDED.elevation_m, features_daily.elevation_m),
          slope_pct = COALESCE(EXCLUDED.slope_pct, features_daily.slope_pct),
          aspect_deg = COALESCE(EXCLUDED.aspect_deg, features_daily.aspect_deg),
          doy = EXCLUDED.doy,
          weather_source = EXCLUDED.weather_source,
          veg_source = COALESCE(EXCLUDED.veg_source, features_daily.veg_source),
          moisture_source = COALESCE(EXCLUDED.moisture_source, features_daily.moisture_source),
          terrain_source = COALESCE(EXCLUDED.terrain_source, features_daily.terrain_source),
          updated_at = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur, sql,
            [tuple(r.get(c) for c in cols) for r in rows],
            page_size=500
        )
    conn.commit()

# ----------------------------
# Parallel processing helpers
# ----------------------------

def process_batch_parallel(batch_records, max_workers=4):
    """Process a batch of records in parallel."""
    
    def process_single_record(record_data):
        """Process a single record with all data sources."""
        fire_id, date_obj, lat, lon = record_data
        
        try:
            # Weather data (cached)
            pw = power_daily(lat, lon, date_obj, date_obj)
            if date_obj.strftime("%Y%m%d") in pw:
                day_data = pw[date_obj.strftime("%Y%m%d")]
                temp_c = day_data.get("T2M")
                rh = day_data.get("RH2M")
                wind = day_data.get("WS2M") 
                prcp = day_data.get("PRECTOTCORR", day_data.get("PRECTOT"))
                
                vpd = vpd_kpa(temp_c, rh) if temp_c and rh else None
                dsr = days_since_rain_from_series(pw, date_obj.strftime("%Y%m%d"))
                ffmc, dmc, dc, isi, bui, fwi_val = fwi_from_daily(temp_c, rh, wind, prcp or 0.0, ffmc_prev=85.0, dmc_prev=6.0, dc_prev=15.0)
            else:
                temp_c = rh = wind = prcp = vpd = dsr = fwi_val = None
            
            # Moisture data (parallel safe)
            try:
                lfmc_proxy, moisture_src = lfmc_from_swi(lat, lon, date_obj)
            except Exception as e:
                log.warning(f"LFMC(SWI) failed {fire_id} {date_obj}: {e}")
                lfmc_proxy, moisture_src = None, None
            
            # Terrain data (parallel safe)
            elev, slope, aspect, terr_src = lookup_terrain(lat, lon)
            
            return {
                'fire_id': fire_id,
                'date': date_obj, 
                'lat': lat,
                'lon': lon,
                'temp_c': temp_c,
                'rel_humidity_pct': rh,
                'wind_speed_ms': wind,
                'precip_mm': prcp,
                'vpd_kpa': vpd,
                'fwi': fwi_val,
                'days_since_rain': dsr,
                'lfmc_proxy_pct': lfmc_proxy,
                'elevation_m': elev,
                'slope_pct': slope,
                'aspect_deg': aspect,
                'doy': date_obj.timetuple().tm_yday,
                'weather_source': "NASA_POWER" if temp_c else None,
                'moisture_source': moisture_src,
                'terrain_source': terr_src
            }
            
        except Exception as e:
            log.error(f"Failed to process {fire_id} at {date_obj}: {e}")
            return None
    
    # Process records in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {
            executor.submit(process_single_record, record): record 
            for record in batch_records
        }
        
        for future in as_completed(future_to_record):
            result = future.result()
            if result:
                results.append(result)
    
    return results

# ----------------------------
# Main loop (historical)
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Build features for wildfire ROS targets from multiple data sources")
    ap.add_argument("--batch-size", type=int, default=500, help="DB upsert batch size")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of records to process (for testing)")
    ap.add_argument("--start", type=lambda s: dt.date.fromisoformat(s), default=None, 
                    help="Optional start date filter (YYYY-MM-DD)")
    ap.add_argument("--end", type=lambda s: dt.date.fromisoformat(s), default=None, 
                    help="Optional end date filter (YYYY-MM-DD)")
    ap.add_argument("--force-rebuild", action="store_true", default=False,
                    help="Rebuild features for records that already exist (default: only missing records)")
    ap.add_argument("--parallel", action="store_true", default=False,
                    help="Enable parallel processing for faster execution")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of parallel workers (default: 4)")
    ap.add_argument("--fast", action="store_true", default=False,
                    help="Fast mode: weather + terrain only (skip vegetation and moisture)")
    ap.add_argument("--instance", type=int, default=0,
                    help="Instance number (0-based) for parallel processing")
    ap.add_argument("--total-instances", type=int, default=1,
                    help="Total number of parallel instances running")
    args = ap.parse_args()

    # Initialize data sources
    init_gee()  # No-op if USE_GEE_VEG=false

    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = False

    # Get records to process from ros_targets
    only_missing = not args.force_rebuild
    all_todo = get_missing_targets(conn, args.start, args.end, only_missing=only_missing, limit=args.limit)
    
    # Split work across instances
    if args.total_instances > 1:
        total_records = len(all_todo)
        records_per_instance = total_records // args.total_instances
        start_idx = args.instance * records_per_instance
        
        if args.instance == args.total_instances - 1:
            # Last instance gets any remaining records
            end_idx = total_records
        else:
            end_idx = start_idx + records_per_instance
            
        todo = all_todo[start_idx:end_idx]
        log.info(f"Instance {args.instance + 1}/{args.total_instances}: Processing records {start_idx + 1}-{end_idx} ({len(todo)} records)")
    else:
        todo = all_todo
        log.info(f"Processing {len(todo)} records from ros_targets")
    
    if args.limit:
        log.info(f"Limited to {args.limit} records")
    if only_missing:
        log.info("Building features for missing records only")
    else:
        log.info("Rebuilding features for all matching records")

    # Group all points by date for GEE batching
    points_by_date: Dict[dt.date, List[dict]] = defaultdict(list)
    for (fire_id, date_, lat, lon) in todo:
        date_obj: dt.date = date_ if isinstance(date_, dt.date) else dt.date.fromisoformat(str(date_))
        points_by_date[date_obj].append({"fire_id": fire_id, "lat": lat, "lon": lon})

    # Pre-fetch vegetation per date (one reduceRegions call per day)
    veg_map: Dict[tuple, tuple[Optional[float], Optional[float], str]] = {}
    if USE_GEE_VEG:
        log.info(f"GEE vegetation batching enabled for {len(points_by_date)} unique dates")
        for d, pts in points_by_date.items():
            res = fetch_ndvi_ndmi_batch(pts, d)  # {fire_id: (ndvi, ndmi, src)}
            for fid, tup in res.items():
                veg_map[(fid, d)] = tup

    power_cache: Dict[Tuple[float,float,dt.date,dt.date], Dict[str, Dict[str, float]]] = {}
    prev_codes_by_point: Dict[Tuple[float,float], Tuple[float,float,float]] = {}
    done = 0

    for (fire_id, date_, lat, lon) in todo:
        date_obj: dt.date = date_ if isinstance(date_, dt.date) else dt.date.fromisoformat(str(date_))
        doy = int(date_obj.strftime("%j"))

        # ---- WEATHER: fetch 30-day window to compute DSR + same-day variables
        win_start = date_obj - dt.timedelta(days=WINDOW_DSR_DAYS - 1)
        pkey = (round(lat,4), round(lon,4), win_start, date_obj)
        if pkey not in power_cache:
            try:
                power_cache[pkey] = power_daily(lat, lon, win_start, date_obj)
            except Exception as e:
                log.warning(f"POWER fetch failed for {fire_id} {date_obj} ({lat},{lon}): {e}")
                power_cache[pkey] = {}

        series = power_cache[pkey]
        ymd = date_obj.strftime("%Y%m%d")
        pw = series.get(ymd, {})
        temp_c = pw.get("T2M")
        rh     = pw.get("RH2M")
        wind   = pw.get("WS2M")
        prcp   = pw.get("PRECTOTCORR") or pw.get("PRECTOT")  # Try PRECTOTCORR first, fallback to PRECTOT
        vpd    = vpd_kpa(temp_c, rh)
        dsr    = days_since_rain_from_series(series, ymd)

        # FWI using minimal memory per (lat,lon)
        prev_ffmc, prev_dmc, prev_dc = prev_codes_by_point.get((lat, lon), (None, None, None))
        ffmc, dmc, dc, isi, bui, fwi_val = fwi_from_daily(temp_c, rh, wind, prcp,
                                                          prev_ffmc, prev_dmc, prev_dc,
                                                          lat=lat, doy=doy)
        if ffmc is not None:
            prev_codes_by_point[(lat, lon)] = (ffmc, dmc, dc)

        # ---- VEGETATION: from batch map (if present), else None
        ndvi = ndmi = veg_src = None
        tup = veg_map.get((fire_id, date_obj))
        if tup:
            ndvi, ndmi, veg_src = tup

        # ---- Copernicus SWI → LFMC proxy (historical)
        try:
            lfmc_proxy, moisture_src = lfmc_from_swi(lat, lon, date_obj)
        except Exception as e:
            log.warning(f"LFMC(SWI) failed {fire_id} {date_obj}: {e}")
            lfmc_proxy, moisture_src = (None, None)

        # ---- TERRAIN: optional static sampling (kept as placeholders)
        elev, slope, aspect, terr_src = lookup_terrain(lat, lon)

        row = dict(
            fire_id=fire_id, date=date_obj, lat=lat, lon=lon,
            temp_c=temp_c, rel_humidity_pct=rh, wind_speed_ms=wind, precip_mm=prcp,
            vpd_kpa=vpd, fwi=fwi_val, days_since_rain=dsr,
            ndvi=ndvi, ndmi=ndmi, lfmc_proxy_pct=lfmc_proxy,
            elevation_m=elev, slope_pct=slope, aspect_deg=aspect,
            doy=doy,
            weather_source="NASA_POWER" if pw else None,
            veg_source=veg_src,
            moisture_source=moisture_src,
            terrain_source=terr_src
        )
        
        # Insert each record immediately to avoid losing progress
        upsert_features(conn, [row])
        done += 1
        log.info(f"✅ Processed {fire_id} ({done}/{len(todo)}) - Elev: {elev}m, LFMC: {lfmc_proxy}%, Temp: {temp_c}°C")

    log.info(f"✅ All {done} records processed successfully!")

    conn.close()
    log.info("All done.")

if __name__ == "__main__":
    main()
