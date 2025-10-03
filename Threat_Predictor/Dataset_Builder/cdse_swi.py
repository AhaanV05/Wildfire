# cdse_swi.py
from __future__ import annotations
import os
import csv
import time, json, datetime as dt, logging, random
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import zipfile

import requests
import xarray as xr
import numpy as np

from config import (
    CDSE_ODATA_BASE, CDSE_DOWNLOAD_BASE, CDSE_TOKEN_URL,
    CDSE_USERNAME, CDSE_PASSWORD, CDSE_TOTP, CDSE_GRANT_TYPE,
    SWI_COLLECTION_NAME, SWI_DATASET_IDENTIFIER, SWI_VARIABLE,
    SWI_CACHE_DIR, USE_COPERNICUS_SWI,
    SWI_CSV_NC, SWI_CSV_COG,
)

log = logging.getLogger("cdse_swi")
log.setLevel(logging.INFO)

# Token cache & simple CSV index cache
_TOKEN_CACHE = Path(SWI_CACHE_DIR) / "cdse_token.json"
_INDEX_CACHE_NC  = Path(SWI_CACHE_DIR) / "swi_nc_date_index.json"
_INDEX_CACHE_COG = Path(SWI_CACHE_DIR) / "swi_cog_date_index.json"

# Pick a download base (default to CDSE download host; fall back to catalogue host)
_DL_BASE = (os.getenv("CDSE_DOWNLOAD_BASE") or CDSE_DOWNLOAD_BASE or CDSE_ODATA_BASE or "").rstrip("/")
if not _DL_BASE:
    _DL_BASE = "https://download.dataspace.copernicus.eu/odata/v1"

# -------------------------
# OAuth2 token (cached)
# -------------------------
def _get_token() -> str:
    if not (CDSE_TOKEN_URL and CDSE_USERNAME and CDSE_PASSWORD):
        raise RuntimeError("CDSE username/password not configured in .env")

    # Use cached token if still valid
    if _TOKEN_CACHE.exists():
        try:
            data = json.loads(_TOKEN_CACHE.read_text())
            if data.get("expires_at", 0) > time.time() + 60:
                return data["access_token"]
        except Exception:
            pass

    grant_type = (CDSE_GRANT_TYPE or "password").lower()
    if grant_type != "password":
        raise RuntimeError(f"Unsupported CDSE_GRANT_TYPE={grant_type}; expected 'password'")

    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": CDSE_USERNAME,
        "password": CDSE_PASSWORD,
    }
    if CDSE_TOTP:
        form["totp"] = CDSE_TOTP  # only if you enabled 2FA

    resp = requests.post(CDSE_TOKEN_URL, data=form, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"CDSE token error {resp.status_code}: {resp.text[:200]}")

    tok = resp.json()
    tok["expires_at"] = time.time() + tok.get("expires_in", 600)
    _TOKEN_CACHE.write_text(json.dumps(tok))
    return tok["access_token"]


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_get_token()}"}


# -------------------------
# Rate-limit aware GET
# -------------------------
def _get(url: str, params=None, stream=False, max_retries=6):
    back = 1.0
    for _ in range(max_retries):
        r = requests.get(url, params=params, headers=_auth_headers(), timeout=60, stream=stream)
        if r.status_code in (200, 206):
            return r
        if r.status_code in (401, 403):
            # force token refresh next iteration
            _TOKEN_CACHE.unlink(missing_ok=True)
        if r.status_code in (429, 502, 503, 504):
            sleep = back * (1 + 0.2 * random.random())
            log.warning(f"OData throttled ({r.status_code}). Retrying in {sleep:.1f}s…")
            time.sleep(sleep)
            back = min(back * 2, 64)
            continue
        raise RuntimeError(f"OData GET {r.status_code}: {r.text[:200]}")
    raise RuntimeError("OData GET: max retries exceeded")


# ============================================================================
# CSV CATALOGUE SUPPORT
# Build a map: YYYY-MM-DD -> (product_id, product_name)
# ============================================================================

def _parse_iso(s: str) -> dt.datetime | None:
    try:
        # CSV uses e.g. 2007-06-30T12:00:00.000
        return dt.datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        return None

def _csv_index_path(which: str) -> Path:
    if which == "nc":
        return _INDEX_CACHE_NC
    else:
        return _INDEX_CACHE_COG

def _csv_source_path(which: str) -> Optional[str]:
    return SWI_CSV_NC if which == "nc" else SWI_CSV_COG

def _load_csv_rows(src_csv: str) -> List[dict]:
    # CSV is semicolon-delimited
    with open(src_csv, "r", encoding="utf-8") as f:
        sniffer = csv.Sniffer()
        sample = f.read(1024)
        dialect = sniffer.sniff(sample, delimiters=";,")
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        return list(reader)

def _build_date_to_name_index(which: str) -> Dict[str, Dict[str, str]]:
    """
    Returns dict: { 'YYYY-MM-DD': {'id': <uuid>, 'name': <product_name>} }
    Caches to JSON for speed.
    """
    p = _csv_index_path(which)
    src = _csv_source_path(which)
    if not src:
        return {}
    if not Path(src).exists():
        raise RuntimeError(f"SWI_CSV_{which.upper()} set to '{src}' but file not found.")

    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass

    rows = _load_csv_rows(src)
    out: Dict[str, Dict[str, str]] = {}

    # Columns we rely on: id, name, nominal_date (or content_date_start/end)
    for r in rows:
        pid  = r.get("id") or r.get("Id") or r.get("ID")
        name = r.get("name") or r.get("Name")
        nd   = r.get("nominal_date") or r.get("NominalDate")
        if not (pid and name and nd):
            # fall back: try content_date_end
            nd = r.get("content_date_end") or r.get("ContentDate/End")
            if not (pid and name and nd):
                continue
        t = _parse_iso(nd)
        if not t:
            continue
        key = t.date().isoformat()
        # keep the first (or overwrite; either is fine – there is 1 per day globally)
        out[key] = {"id": pid, "name": name}

    p.write_text(json.dumps(out))
    return out

def _csv_find_id_for_date(date_obj: dt.date) -> Tuple[Optional[str], str]:
    """
    Look up by NC index first; if not present and a COG CSV exists, try that.
    Returns (product_id, which_index) where which_index in {'nc','cog'}
    """
    for which in ("nc", "cog"):
        src = _csv_source_path(which)
        if not src:
            continue
        idx = _build_date_to_name_index(which)
        # Exact day
        k = date_obj.isoformat()
        if k in idx:
            return idx[k]["id"], which
        # ±3-day window
        for off in (-3, -2, -1, 1, 2, 3):
            kk = (date_obj + dt.timedelta(days=off)).isoformat()
            if kk in idx:
                log.info(f"CSV: using {kk} for requested {date_obj}")
                return idx[kk]["id"], which
    return (None, "nc")

# ============================================================================
# ODATA LOOKUP (fallback if you want pure API)
# ============================================================================

def _odata_find_product_by_date(date_obj: dt.date, lat: float = None, lon: float = None):
    if not CDSE_ODATA_BASE:
        raise RuntimeError("CDSE_ODATA_BASE not configured.")

    collection_name = os.environ.get("SWI_COLLECTION_NAME", SWI_COLLECTION_NAME or "CLMS")
    configured_id   = os.environ.get("SWI_DATASET_IDENTIFIER", SWI_DATASET_IDENTIFIER)

    CANDIDATE_IDS = [i for i in [
        configured_id,
        "swi-timeseries_global_12.5km_daily_v3",
        "swi_global_12.5km_daily_v3",
        "swi_global_12.5km_10daily_v3",
        "swi_global_12.5km_daily_v4",
    ] if i]

    def _iso(d: dt.date, h=0, m=0, s=0): return dt.datetime(d.year, d.month, d.day, h, m, s).isoformat() + "Z"
    url = f"{CDSE_ODATA_BASE}/Products"

    day_start = _iso(date_obj, 0, 0, 0)
    next_day  = _iso(date_obj + dt.timedelta(days=1), 0, 0, 0)

    geo = ""
    if lat is not None and lon is not None:
        geo = f" and OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})')"

    def _run(q):
        return _get(url, params={
            "$filter": q,
            "$orderby": "ContentDate/Start desc",
            "$expand": "Assets",
            "$top": 1
        }).json().get("value", [])

    for did in CANDIDATE_IDS:
        q = (f"Collection/Name eq '{collection_name}' and "
             f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'datasetIdentifier' and "
             f"att/OData.CSC.StringAttribute/Value eq '{did}'){geo} and "
             f"ContentDate/Start lt {next_day} and ContentDate/End ge {day_start}")
        items = _run(q)
        if items:
            log.info(f"OData matched datasetIdentifier={did} for {date_obj}")
            return items[0]

        for off in (-3,-2,-1,1,2,3):
            d0 = _iso(date_obj + dt.timedelta(days=off))
            d1 = _iso(date_obj + dt.timedelta(days=off+1))
            q2 = (f"Collection/Name eq '{collection_name}' and "
                  f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'datasetIdentifier' and "
                  f"att/OData.CSC.StringAttribute/Value eq '{did}'){geo} and "
                  f"ContentDate/Start ge {d0} and ContentDate/Start lt {d1}")
            items = _run(q2)
            if items:
                log.info(f"OData matched datasetIdentifier={did} near {date_obj}")
                return items[0]

    log.warning(f"No SWI product for {date_obj}")
    return None


# ============================================================================
# Download helpers (Assets → $value → Nodes)
# ============================================================================

def _download_via_assets(product_id: str) -> Optional[Path]:
    url = f"{_DL_BASE}/Products({product_id})/Assets"
    r = _get(url)
    assets = r.json().get("value", [])
    if not assets:
        return None

    # Prefer NetCDF, then GeoTIFF
    def _score(a):
        name = (a.get("Name") or "").lower()
        ctype = (a.get("ContentType") or "").lower()
        if name.endswith(".nc") or "netcdf" in ctype:
            return 0
        if name.endswith(".tif") or name.endswith(".tiff") or "tiff" in ctype:
            return 1
        return 9

    assets.sort(key=_score)
    a0 = assets[0]
    if _score(a0) == 9:
        return None

    aid  = a0.get("Id")
    name = a0.get("Name") or f"{product_id}.bin"
    ext  = name.split(".")[-1].lower()
    if ext not in ("nc","nc4","tif","tiff"):
        ext = "nc"

    out = Path(SWI_CACHE_DIR) / f"{product_id}.{ext}"
    if out.exists() and out.stat().st_size > 0:
        return out

    vurl = f"{_DL_BASE}/Assets({aid})/$value"
    with _get(vurl, stream=True) as resp:
        resp.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out)
    return out

def _extract_from_zip(zip_path: Path, product_id: str) -> Optional[Path]:
    """Extract NetCDF or TIFF files from ZIP archive."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Look for NetCDF or TIFF files
            for name in zf.namelist():
                if name.lower().endswith(('.nc', '.nc4', '.tif', '.tiff')):
                    # Extract to cache directory
                    ext = name.split('.')[-1].lower()
                    extracted_path = Path(SWI_CACHE_DIR) / f"{product_id}.{ext}"
                    with zf.open(name) as src, open(extracted_path, 'wb') as dst:
                        dst.write(src.read())
                    return extracted_path
        return None
    except Exception:
        return None

def _download_via_value(product_id: str) -> Optional[Path]:
    """Direct product download ($value). CLMS often serves the NC directly."""
    url = f"{_DL_BASE}/Products({product_id})/$value"
    out = Path(SWI_CACHE_DIR) / f"{product_id}.nc"
    try:
        with _get(url, stream=True) as resp:
            if resp.status_code not in (200, 206):
                return None
            # Try to guess extension from headers
            ctype = (resp.headers.get("Content-Type") or "").lower()
            
            # Download to temporary file first
            tmp = out.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            
            # Check if it's a ZIP file by reading the first few bytes
            with open(tmp, 'rb') as f:
                header = f.read(4)
            
            if header.startswith(b'PK'):
                # It's a ZIP file, extract the NetCDF/TIFF from it
                extracted = _extract_from_zip(tmp, product_id)
                tmp.unlink()  # Remove the ZIP file
                return extracted
            else:
                # Assume it's a direct NetCDF file
                ext = ".nc" if "netcdf" in ctype or "application/octet-stream" in ctype else ".bin"
                final = out.with_suffix(ext)
                tmp.replace(final)
                return final
    except Exception:
        return None

def _download_via_nodes(product_id: str) -> Optional[Path]:
    url = f"{_DL_BASE}/Products({product_id})/Nodes"
    r = _get(url, params={"$filter": "IsLeaf eq true"})
    nodes = r.json().get("value", [])
    if not nodes:
        return None

    def _score(n):
        name = (n.get("Name") or "").lower()
        if name.endswith(".nc") or name.endswith(".nc4"):
            return 0
        if name.endswith(".tif") or name.endswith(".tiff"):
            return 1
        return 9

    nodes.sort(key=_score)
    n0 = nodes[0]
    if _score(n0) == 9:
        return None

    nid  = n0.get("Id")
    name = n0.get("Name") or f"{product_id}.bin"
    ext  = name.split(".")[-1].lower()
    if ext not in ("nc","nc4","tif","tiff"):
        ext = "nc"
    out = Path(SWI_CACHE_DIR) / f"{product_id}.{ext}"
    if out.exists() and out.stat().st_size > 0:
        return out

    vurl = f"{_DL_BASE}/Products({product_id})/Nodes('{nid}')/$value"
    with _get(vurl, stream=True) as resp:
        resp.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out)
    return out

def _download_product_asset(product_id: str) -> Path:
    """
    Try Assets → $value → Nodes in that order.
    """
    # 1) Assets
    try:
        p = _download_via_assets(product_id)
        if p:
            return p
    except Exception as e:
        log.debug(f"Assets path failed: {e}")

    # 2) Direct $value
    try:
        p = _download_via_value(product_id)
        if p:
            return p
    except Exception as e:
        log.debug(f"$value path failed: {e}")

    # 3) Nodes
    p = _download_via_nodes(product_id)
    if p:
        return p

    raise RuntimeError("No downloadable .nc/.tif asset found via Assets/$value/Nodes.")


# -------------------------
# Sample SWI at (lat, lon)
# -------------------------
def _sample_swi(nc_or_tif: Path, lat: float, lon: float, target_date: dt.date = None) -> float | None:
    suf = nc_or_tif.suffix.lower()
    if suf in (".tif", ".tiff"):
        try:
            import rioxarray as rxr
            da = rxr.open_rasterio(nc_or_tif)
            if "band" in getattr(da, "dims", []):
                da = da.isel(band=0)
            val = da.sel(x=lon, y=lat, method="nearest").values
            if isinstance(val, np.ndarray):
                val = float(np.squeeze(val).item())
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return None
            return float(max(0.0, min(100.0, val)))
        except Exception as e:
            raise RuntimeError("GeoTIFF sampling requires rioxarray/rasterio. Install: pip install rioxarray rasterio") from e

    # NetCDF
    try:
        ds = xr.open_dataset(nc_or_tif, engine="netcdf4")
    except ValueError:
        # Fallback to auto-detection if netcdf4 engine fails
        ds = xr.open_dataset(nc_or_tif)
    try:
        lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
        lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
        if not (lat_name and lon_name):
            lat_name = lat_name or ("lat" if "lat" in ds.variables else "latitude")
            lon_name = lon_name or ("lon" if "lon" in ds.variables else "longitude")

        var = SWI_VARIABLE if SWI_VARIABLE in ds.variables else next(
            (v for v in ["SWI_010", "SWI", "swi", "SWI1"] if v in ds.variables), None
        )
        if not var:
            raise RuntimeError(f"SWI variable not found in {nc_or_tif.name}")

        da = ds[var]
        if "time" in da.dims and da.sizes.get("time", 1) == 1:
            da = da.isel(time=0)

        lons = ds[lon_name]
        lats = ds[lat_name]
        Lon = lon
        if (float(lons.max()) > 180) and Lon < 0:
            Lon = Lon % 360

        # Check if lat/lon are dimension coordinates or location-indexed coordinates
        if lat_name in da.dims and lon_name in da.dims:
            # Traditional grid structure
            val = da.sel({lat_name: lat, lon_name: Lon}, method="nearest").values
        else:
            # Location-indexed structure - find nearest location
            # Calculate distances to all locations
            lat_diff = (lats - lat) ** 2
            lon_diff = (lons - Lon) ** 2
            distances = np.sqrt(lat_diff + lon_diff)
            nearest_idx = distances.argmin()
            selected = da.isel(locations=nearest_idx)
            
            # Handle time dimension if present
            if "time" in selected.dims:
                if target_date is not None:
                    # Select the time closest to our target date
                    target_datetime = np.datetime64(target_date.strftime('%Y-%m-%d'))
                    try:
                        selected = selected.sel(time=target_datetime, method="nearest")
                    except (KeyError, ValueError):
                        # If exact date selection fails, find closest date
                        time_diffs = np.abs(selected.time - target_datetime)
                        closest_idx = time_diffs.argmin()
                        selected = selected.isel(time=closest_idx)
                else:
                    # If no target date specified, take the most recent
                    selected = selected.isel(time=-1)
            
            val = selected.values
        
        if isinstance(val, np.ndarray):
            if val.size == 1:
                val = float(val.item())
            else:
                # If still multi-dimensional, take the first value
                val = float(val.flat[0])

        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return None
        return float(max(0.0, min(100.0, val)))
    finally:
        ds.close()


# -------------------------
# Public entry point for builder
# -------------------------
def lfmc_from_swi(lat: float, lon: float, date_obj: dt.date) -> tuple[Optional[float], Optional[str]]:
    """
    Returns (LFMC_proxy_pct, source_string) or (None, None) if unavailable.
    """
    if not USE_COPERNICUS_SWI:
        return (None, None)

    # 1) CSV path: fast, avoids catalogue quirks
    pid = None
    which = "nc"
    try:
        pid, which = _csv_find_id_for_date(date_obj)
    except Exception as e:
        log.debug(f"CSV lookup failed: {e}")

    prod = None
    if not pid:
        # 2) Fallback to OData discovery (slower)
        prod = _odata_find_product_by_date(date_obj, lat=lat, lon=lon)
        if not prod:
            log.warning(f"No SWI product for {date_obj}")
            return (None, None)
        pid = prod.get("Id") or prod.get("@iot.id") or prod.get("id")
        if not pid:
            log.warning(f"SWI product missing Id for {date_obj}")
            return (None, None)

    # Download & sample
    try:
        asset_path = _download_product_asset(str(pid))
        swi = _sample_swi(asset_path, lat, lon, date_obj)
    except Exception as e:
        log.warning(f"SWI sample failed {date_obj} ({lat},{lon}): {e}")
        return (None, None)

    if swi is None:
        return (None, None)

    lfmc = float(max(0.0, min(200.0, swi * 1.2)))
    return (lfmc, "CDSE_CLMS_SWI_DAILY")