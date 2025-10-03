# config.py
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load .env (auto-discover from project root or CWD); OS env still wins
load_dotenv(find_dotenv(usecwd=True), override=False)

def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def _opt(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

# --- Database ---
PG_DSN = _req("PG_DSN")

# --- POWER (weather) ---
POWER_BASE = _opt("POWER_BASE", "https://power.larc.nasa.gov/api/temporal/daily/point")
POWER_COMMUNITY = _opt("POWER_COMMUNITY", "AG")

# --- Cache ---
CACHE_DIR = _opt("CACHE_DIR", ".cache/wildfire")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# --- GEE vegetation switch ---
USE_GEE_VEG = os.getenv("USE_GEE_VEG", "false").lower() in ("1", "true", "yes")
GEE_PROJECT = _opt("GEE_PROJECT")  # Google Cloud Project ID for Earth Engine

# --- OpenTopography Terrain ---
OPENTOPO_API_KEY = _req("OPENTOPO_API_KEY")
OPENTOPO_BASE_URL = _opt("OPENTOPO_BASE_URL", "https://cloud.sdsc.edu/v1/products")
TERRAIN_DEM_TYPE = _opt("TERRAIN_DEM_TYPE", "NASADEM")
TERRAIN_TILE_SIZE = float(_opt("TERRAIN_TILE_SIZE", "0.1"))

# --- Optional logging level ---
LOG_LEVEL = _opt("LOG_LEVEL", "INFO")

# --- CDSE / OData (password grant) ---
CDSE_ODATA_BASE    = _opt("CDSE_ODATA_BASE")
CDSE_DOWNLOAD_BASE = _opt("CDSE_DOWNLOAD_BASE")
CDSE_TOKEN_URL     = _opt("CDSE_TOKEN_URL")
CDSE_USERNAME      = _opt("CDSE_USERNAME")
CDSE_PASSWORD      = _opt("CDSE_PASSWORD")
CDSE_TOTP          = _opt("CDSE_TOTP")  # optional
CDSE_GRANT_TYPE    = _opt("CDSE_GRANT_TYPE", "password")

# Switch for using Copernicus SWI at all (OData or CSV)
USE_COPERNICUS_SWI = os.getenv("USE_COPERNICUS_SWI", "false").lower() in ("1","true","yes")

# --- SWI settings ---
SWI_COLLECTION_NAME     = _opt("SWI_COLLECTION_NAME", "CLMS")
SWI_DATASET_IDENTIFIER  = _opt("SWI_DATASET_IDENTIFIER")
SWI_VARIABLE            = _opt("SWI_VARIABLE", "SWI")

# Optional CSV catalogue paths (semicolon-delimited lists from CLMS “CSV Catalogue”)
# Use absolute Windows paths with either double backslashes or forward slashes.
SWI_CSV_NC  = _opt("SWI_CSV_NC")   # e.g. C:/Users/ahaan/Downloads/swi_global_12.5km_daily_v3_nc.csv
SWI_CSV_COG = _opt("SWI_CSV_COG")  # e.g. C:/Users/ahaan/Downloads/swi_global_12.5km_daily_v3_cog.csv

SWI_CACHE_DIR = _opt("SWI_CACHE_DIR", ".cache/swi_daily")
Path(SWI_CACHE_DIR).mkdir(parents=True, exist_ok=True)
