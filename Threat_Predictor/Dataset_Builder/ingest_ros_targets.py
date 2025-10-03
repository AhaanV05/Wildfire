# ingest_ros_targets.py  (streaming per-day)
import os, math, argparse, glob
import pandas as pd
import numpy as np
from datetime import date as date_cls
from collections import defaultdict
import psycopg2
from psycopg2.extras import execute_values
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))

def day_str(d: date_cls) -> str: return d.strftime("%Y-%m-%d")

def parse_date_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    dmy = s.str.match(r"^\d{2}-\d{2}-\d{4}$")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any(): out.loc[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d", errors="coerce")
    if dmy.any(): out.loc[dmy] = pd.to_datetime(s[dmy], format="%d-%m-%Y", errors="coerce")
    rem = out.isna()
    if rem.any(): out.loc[rem] = pd.to_datetime(s[rem], errors="coerce", infer_datetime_format=True)
    return out.dt.date

def load_detection_file(path: str, date_from=None, date_to=None) -> pd.DataFrame:
    print(f"[INFO] Loading {path}")
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    lat  = cols.get('latitude') or 'latitude'
    lon  = cols.get('longitude') or 'longitude'
    date = cols.get('acq_date') or 'acq_date'
    time = cols.get('acq_time') or 'acq_time'
    frp  = cols.get('frp') or 'frp'
    out = pd.DataFrame({
        'lat': pd.to_numeric(df[lat], errors='coerce'),
        'lon': pd.to_numeric(df[lon], errors='coerce'),
        'acq_date': parse_date_column(df[date]),
        'acq_time': pd.to_numeric(df.get(time, np.nan), errors='coerce'),
        'frp': pd.to_numeric(df.get(frp, np.nan), errors='coerce'),
    }).dropna(subset=['lat','lon','acq_date'])
    out = out[(out.lat.between(-90,90)) & (out.lon.between(-180,180))]
    if date_from: out = out[out.acq_date >= pd.to_datetime(date_from).date()]
    if date_to:   out = out[out.acq_date <= pd.to_datetime(date_to).date()]
    out = out.drop_duplicates(subset=["lat","lon","acq_date","acq_time"]).reset_index(drop=True)
    print(f"[INFO] Loaded {len(out):,} detections, dates {out['acq_date'].min()} → {out['acq_date'].max()}")
    return out

class Cluster:
    __slots__=("day","id_day","points","centroid","fire_id")
    def __init__(self, day, id_day, lat, lon):
        self.day=day; self.id_day=id_day
        self.points=[(float(lat),float(lon))]
        self.centroid=(float(lat),float(lon))
        self.fire_id=None
    def add(self, lat, lon):
        self.points.append((float(lat),float(lon)))
        lats=[p[0] for p in self.points]; lons=[p[1] for p in self.points]
        self.centroid=(float(np.mean(lats)), float(np.mean(lons)))

def cluster_day(points_df: pd.DataFrame, day: date_cls, eps_km=5.0) -> list:
    """
    Cluster detections for a single day using DBSCAN with haversine distance.
    Much faster than nested loops.
    """
    if points_df.empty:
        return []

    # Convert to radians for haversine
    coords = np.radians(points_df[['lat', 'lon']].values)

    # eps in radians (km -> radians on Earth's sphere)
    kms_per_radian = 6371.0088
    eps_rad = eps_km / kms_per_radian

    # Run DBSCAN
    db = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine').fit(coords)
    labels = db.labels_

    # Build Cluster objects
    clusters = []
    for cid in np.unique(labels):
        mask = labels == cid
        subset = points_df[mask]
        lat_mean = subset['lat'].mean()
        lon_mean = subset['lon'].mean()
        c = Cluster(day, f"{day_str(day)}_{cid+1}", lat_mean, lon_mean)
        for r in subset.itertuples(index=False):
            c.add(r.lat, r.lon)
        clusters.append(c)

    return clusters


def link_days(prev_clusters, curr_clusters, link_km=10.0, next_id=1):
    if not prev_clusters:
        for c in curr_clusters:
            c.fire_id = f"F{next_id:07d}"; next_id += 1
        return next_id

    # Build KD-tree of prev centroids
    prev_coords = np.radians([[c.centroid[0], c.centroid[1]] for c in prev_clusters])
    tree = BallTree(prev_coords, metric='haversine')

    # Query nearest prev cluster for each curr cluster
    curr_coords = np.radians([[c.centroid[0], c.centroid[1]] for c in curr_clusters])
    dists, idxs = tree.query(curr_coords, k=1)
    dists_km = dists[:,0] * 6371.0088

    for c, dist_km, idx in zip(curr_clusters, dists_km, idxs[:,0]):
        if dist_km <= link_km and prev_clusters[idx].fire_id:
            c.fire_id = prev_clusters[idx].fire_id
        else:
            c.fire_id = f"F{next_id:07d}"; next_id += 1

    return next_id

def ros_from_spread(prev_pts, curr_pts, kappa=2.0) -> float:
    if not prev_pts or not curr_pts: return np.nan
    dists=[]
    for (clat,clon) in curr_pts:
        best=1e9
        for (plat,plon) in prev_pts:
            d=haversine_km(clat,clon,plat,plon)
            if d<best: best=d
        dists.append(best)
    if not dists: return np.nan
    adv_km = float(np.percentile(dists,95)) * kappa
    return float(np.clip((adv_km*1000.0)/1440.0, 0.01, 60.0))

def upsert_ros_targets(conn, rows):
    if not rows:
        return

    # Deduplicate by (fire_id, date) → keep max ROS per day
    dedup = {}
    for r in rows:
        key = (r[0], r[1])  # (fire_id, date)
        if key not in dedup:
            dedup[key] = r
        else:
            # Compare ROS (index 4 in tuple)
            if r[4] > dedup[key][4]:
                dedup[key] = r
    rows = list(dedup.values())

    sql = """
    INSERT INTO ros_targets (fire_id, date, lat, lon, target_ros_m_min, label_source, notes)
    VALUES %s
    ON CONFLICT (fire_id, date) DO UPDATE
      SET lat = EXCLUDED.lat,
          lon = EXCLUDED.lon,
          target_ros_m_min = EXCLUDED.target_ros_m_min,
          label_source = EXCLUDED.label_source,
          notes = EXCLUDED.notes;
    """
    with conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=20000)
    print(f"[INFO] Upserted {len(rows):,} rows into ros_targets")


def process_file_stream(conn, detections: pd.DataFrame, eps_km=5.0, link_km=10.0, kappa=2.0, print_every=1):
    days = sorted(detections.acq_date.unique())
    prev_clusters=[]; next_id=1
    total_days = len(days)
    inserted_total=0

    for idx, d in enumerate(days, start=1):
        day_df = detections[detections.acq_date == d][["lat","lon"]]
        cs = cluster_day(day_df, d, eps_km=eps_km)
        next_id = link_days(prev_clusters, cs, link_km, next_id)

        # Build map of prev-day clusters by fire_id (for ROS)
        prev_by_id = defaultdict(list)
        for pc in prev_clusters: prev_by_id[pc.fire_id].append(pc)

        # Compute ROS rows for this day
        batch=[]
        for c in cs:
            if not prev_clusters or c.fire_id not in prev_by_id:  # first day for this fire
                continue
            prev_pts=[]
            for pc in prev_by_id[c.fire_id]:
                prev_pts.extend(pc.points)
            ros = ros_from_spread(prev_pts, c.points, kappa=kappa)
            if np.isnan(ros): continue
            lat_c, lon_c = c.centroid
            batch.append((
                c.fire_id, d, round(lat_c,6), round(lon_c,6),
                float(round(ros,4)), "spread_max_daily_advance",
                f"eps_km={eps_km},link_km={link_km},kappa={kappa}"
            ))

        upsert_ros_targets(conn, batch)
        inserted_total += len(batch)

        # progress & roll window
        if idx % print_every == 0:
            print(f"[INFO] {idx}/{total_days} days | {day_str(d)} | clusters={len(cs)} | ROS rows+={len(batch)} (total {inserted_total})")
        prev_clusters = cs

def main():
    ap = argparse.ArgumentParser(description="Stream per-day ROS into Postgres (with progress).")
    ap.add_argument("--pg-dsn", required=True,
                    help="e.g. 'dbname=wildfire user=postgres password=YOURPWD host=localhost port=5432'")
    ap.add_argument("--paths", nargs="+", required=True,
                    help="File paths or globs to FIRMS archive CSVs.")
    ap.add_argument("--date-from", help="Optional YYYY-MM-DD filter start")
    ap.add_argument("--date-to", help="Optional YYYY-MM-DD filter end")
    ap.add_argument("--eps-km", type=float, default=5.0)
    ap.add_argument("--link-km", type=float, default=10.0)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--print-every", type=int, default=1)
    args = ap.parse_args()

    files=[]
    for p in args.paths:
        expanded = glob.glob(p)
        if expanded: files.extend(expanded)
        elif os.path.isfile(p): files.append(p)
        else: print(f"[WARN] No files matched: {p}")
    files = sorted(set(files))
    if not files:
        print("[ERROR] No input files found."); return

    conn = psycopg2.connect(args.pg_dsn)
    try:
        for f in files:
            det = load_detection_file(f, date_from=args.date_from, date_to=args.date_to)
            if det.empty:
                print("[INFO] No detections after filtering; skipping.")
                continue
            process_file_stream(conn, det, eps_km=args.eps_km, link_km=args.link_km,
                                kappa=args.kappa, print_every=args.print_every)
    finally:
        conn.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()
