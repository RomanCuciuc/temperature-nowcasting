# This module fetches live observations from NOAA/NWS (api.weather.gov) for each city.
# It resolves the nearest station and downloads the latest observation payloads and converts
# units, aligns timestamps, and assembles the last L-hour window per node for inference.
# Fallback to a local backup JSON can be used when the network or fields are unavailable.

import os, json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from math import hypot
import pytz
import requests
import pandas as pd
from weather_gcn_citynodes import add_calendar_features, FEATURES

@dataclass
class CitySpec:
    name: str
    lat: float
    lon: float


# This function: Return ISO8601 UTC bounds for 'today' in America/New_York: [00:00, 24:00).
def _today_utc_bounds_ny():
    ny = pytz.timezone("America/New_York")
    now_ny = datetime.now(ny)
    start_ny = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    end_ny = start_ny + timedelta(days=1)
    to_iso = lambda dt: dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return to_iso(start_ny), to_iso(end_ny)

# This function: Convert NWS 'stations/{id}/observations' FeatureCollection to a tidy hourly DataFrame:
# columns: ['time','temp','wspd','rhum','snow'] in model units (C, km/h, %, mm/h).
def _obs_featurecollection_to_df(fc_json):

    rows = []
    feats = fc_json.get("features", []) or []
    for feat in feats:
        p = (feat or {}).get("properties", {})
        t = p.get("timestamp")

        # Units: temperature degC; windSpeed may be m/s or km/h; RH in percent
        temp_c = p.get("temperature", {}).get("value")
        wind_val = p.get("windSpeed", {}).get("value")
        wind_unit = p.get("windSpeed", {}).get("unitCode")  # 'wmoUnit:m_s-1' or 'wmoUnit:km_h-1'
        if wind_val is None:
            wspd_kmh = None
        else:
            if wind_unit and "m_s-1" in wind_unit:
                wspd_kmh = float(wind_val) * 3.6
            else:
                wspd_kmh = float(wind_val)
        rh = p.get("relativeHumidity", {}).get("value")
        # NWS obs rarely include snowfall rate. 0 if missing
        snow_mm_h = 0.0

        if t is not None:
            rows.append({
                "time": t,
                "temp": temp_c,
                "wspd": wspd_kmh,
                "rhum": rh,
                "snow": snow_mm_h
            })
    if not rows:
        return pd.DataFrame(columns=["time","temp","wspd","rhum","snow"])

    df = pd.DataFrame(rows)
    # Normalize to top-of-hour and drop duplicate minutes inside hour keeping the last
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("H")
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df

# This function: Resolve stations for a given (lat, lon) using points API. return list of tuples:
# [(station_id, distance_meters), ...] sorted by distance.
def _stations_by_point(lat, lon, session: requests.Session, user_agent: str, max_list=5):
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
    r = session.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", headers=headers, timeout=15)
    r.raise_for_status()
    pt = r.json()

    obs_url = (pt.get("properties") or {}).get("observationStations")
    if not obs_url:
        return []

    r2 = session.get(obs_url, headers=headers, timeout=15)
    r2.raise_for_status()
    data = r2.json()

    out = []
    feats = data.get("features") or []
    if feats:
        for f in feats:
            pid = (f.get("properties") or {}).get("stationIdentifier")
            dist = (f.get("properties") or {}).get("distance", {}).get("value")
            if pid:
                out.append((pid, float(dist) if dist is not None else 1e12))
    else:
        # Fallback: older structure observationStations = [urls]
        for url in data.get("observationStations", []):
            pid = url.split("/")[-1]
            out.append((pid, 1e12))

    out.sort(key=lambda x: x[1])
    return out[:max_list]

# This function: Build hourly observations for a geo point by blending top-K nearest stations
# with inverse-distance weights. Returns DataFrame ['time','temp','wspd','rhum','snow'].
def fetch_hourly_df_by_latlon(lat: float, lon: float, session: requests.Session, user_agent: str,
                               blend_k: int = 2, max_stations: int = 5):
    cand = _stations_by_point(lat, lon, session, user_agent=user_agent, max_list=max_stations)
    if not cand:
        return pd.DataFrame(columns=["time","temp","wspd","rhum","snow"])

    # Take top-K with finite distances
    chosen = [(sid, dist) for sid, dist in cand if dist is not None and dist < 1e11][:max(1, blend_k)]

    dfs, weights = [], []
    for sid, dist in chosen:
        df = _fetch_station_today_obs(sid, session, user_agent=user_agent)
        if not df.empty:
            dfs.append(df.set_index("time"))
            # inverse-distance weight. add small epsilon to avoid div-by-zero
            w = 1.0 / max(dist, 1.0)
            weights.append(w)

    if not dfs:
        return pd.DataFrame(columns=["time","temp","wspd","rhum","snow"])

    # Align by time (outer join), then weighted average across stations
    base = pd.concat(dfs, axis=1, keys=range(len(dfs)))  # multicolumn: (k, var)
    # Compute weighted average per variable
    wsum = sum(weights)
    out = pd.DataFrame(index=base.index.unique().sort_values())
    for var in ["temp", "wspd", "rhum", "snow"]:
        cols = [base[(k, var)] for k in range(len(dfs)) if (k, var) in base]
        if not cols:
            out[var] = pd.Series(dtype=float)
            continue
        M = pd.concat(cols, axis=1)
        # Weighted average row-wise, ignoring NaNs
        w = pd.Series(weights[:M.shape[1]], index=M.columns, dtype=float)
        num = (M * w.values).sum(axis=1, skipna=True)
        den = (~M.isna() * w.values).sum(axis=1)
        out[var] = num / den.replace(0, pd.NA)

    out = out.reset_index().rename(columns={"index": "time"})
    # Drop rows where everything is NA
    out = out.dropna(how="all", subset=["temp","wspd","rhum","snow"]).reset_index(drop=True)
    return out

# This function: Download all observations for 'today' (NY local) for a single station.
def _fetch_station_today_obs(station_id: str, session: requests.Session, user_agent: str):
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
    start_utc, end_utc = _today_utc_bounds_ny()
    url = f"https://api.weather.gov/stations/{station_id}/observations?start={start_utc}&end={end_utc}"
    r = session.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return _obs_featurecollection_to_df(r.json())


class NWSClient:
    def __init__(self, user_agent: str):
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/geo+json"
        })

    # This function: Pick the station's data by HTTP
    def pick_station_for_point(self, lat: float, lon: float) -> str:
        r = self.sess.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=20)
        r.raise_for_status()
        obs_url = r.json()["properties"]["observationStations"]
        r2 = self.sess.get(obs_url, timeout=30)
        r2.raise_for_status()
        feats = r2.json().get("features", [])
        if not feats:  #
            raise RuntimeError(f"No NWS stations for {lat},{lon}")
        return feats[0]["properties"]["stationIdentifier"]  #

    @staticmethod
    def _to_mm(value, unitCode):
        if value is None: return None
        u = (unitCode or "").lower()
        if "unit:mm" in u: return float(value)
        if "unit:m"  in u: return float(value) * 1000.0
        return float(value)

    @staticmethod
    def _to_ms(value, unitCode):
        if value is None: return None  #
        u = (unitCode or "").lower()
        if "unit:m_s-1" in u:   return float(value)
        if "unit:km_h-1" in u:  return float(value) / 3.6
        return float(value)

    # This function: Project-specific helper function.
    @staticmethod
    def _to_hpa(value, unitCode):
        if value is None: return None
        u = (unitCode or "").lower()
        if "unit:pa" in u:   return float(value) / 100.0
        if "unit:hpa" in u:  return float(value)
        return float(value)

    # This function: Fetch and normalize live NWS observations for the selected cities.
    def fetch_hourly_df(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        time.sleep(1)

        start_s = start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        end_s   = end.replace(microsecond=0).isoformat().replace("+00:00", "Z")

        url = f"https://api.weather.gov/stations/{station_id}/observations?start={start_s}&end={end_s}"
        r = self.sess.get(url, timeout=40)
        r.raise_for_status()

        feats = r.json().get("features", [])
        rows = []

        for f in feats:
            p = f.get("properties", {})
            ts = p.get("timestamp")

            if not ts:
                continue

            t = pd.to_datetime(ts, utc=True)

            temp = p.get("temperature", {})
            rh   = p.get("relativeHumidity", {})
            wsp  = p.get("windSpeed", {})
            prs1 = p.get("barometricPressure", {})
            prs2 = p.get("seaLevelPressure", {})
            prcp1h = p.get("precipitationLastHour", {})
            prcp3h = p.get("precipitationLast3Hours", {})
            snow_depth = p.get("snowDepth", {})
            snowfall1h = p.get("snowfallLastHour", {})

            prcp = None
            if prcp1h.get("value") is not None:
                prcp = self._to_mm(prcp1h.get("value"), prcp1h.get("unitCode"))
            elif prcp3h.get("value") is not None:
                prcp = self._to_mm(prcp3h.get("value"), prcp3h.get("unitCode"))
                if prcp is not None:
                    prcp = prcp / 3.0

            snow = None
            if snowfall1h.get("value") is not None:
                snow = self._to_mm(snowfall1h.get("value"), snowfall1h.get("unitCode"))
            elif snow_depth.get("value") is not None:
                snow = float(snow_depth.get("value")) * 1000.0

            row = {
                "time": t,
                "temp": temp.get("value"),
                "rhum": rh.get("value"),
                "wspd": self._to_ms(wsp.get("value"),  wsp.get("unitCode")),
                "pres": self._to_hpa(prs1.get("value") if prs1.get("value") is not None else prs2.get("value"), prs1.get("unitCode") or prs2.get("unitCode")),
                "prcp": prcp,
                "snow": snow,
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["time","temp","rhum","pres","wspd","prcp","snow"])  #

        df = pd.DataFrame(rows).sort_values("time")
        df["time"] = df["time"].dt.floor("h")
        df = df.groupby("time").last().reset_index()
        return df[["time","temp","rhum","pres","wspd","prcp","snow"]]  #

# This class: To handle errors
class LiveDataError(Exception):
    pass

# This function: Assemble time-windowed node feature tensors (L×features per city).
def build_live_window(cities: List[CitySpec], lags: int, user_agent: str,
                      backup_json: Optional[str] = None):
    client = NWSClient(user_agent=user_agent)
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now_utc - timedelta(hours=lags-1)
    end   = now_utc

    per_city = {}
    station_ids = []

    for c in cities:
        sid = client.pick_station_for_point(c.lat, c.lon)
        station_ids.append(sid)

    for c, sid in zip(cities, station_ids):
        df = client.fetch_hourly_df(sid, start, end)
        rng = pd.date_range(start, end, freq="H", tz="UTC")
        df = df.set_index("time").reindex(rng)

        for col in ["prcp","snow"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        df.fillna(0.0, inplace=True) #todo

        if df[["temp","rhum","pres","wspd"]].isna().any().any():  #
            if backup_json and os.path.exists(backup_json):  #
                return build_window_from_backup(cities, lags, backup_json)  #
            raise LiveDataError(f"Missing critical values for {c.name}")

        per_city[c.name] = df

    feat_order = FEATURES + ["hod_sin","hod_cos","doy_sin","doy_cos"]
    mats = []
    for c in cities:
        df = per_city[c.name].copy()
        df = add_calendar_features(df)
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        mats.append(df[feat_order].astype("float32").values)
    X_last = np.stack(mats, axis=1)
    return np.expand_dims(X_last, axis=0)

# This function: Assemble time-windowed node feature tensors (L×features per city).
def build_window_from_backup(cities: List[CitySpec], lags: int, backup_json: str):
    with open(backup_json, "r", encoding="utf-8") as f:
        arr = json.load(f)
    df_all = pd.DataFrame(arr)

    if df_all.empty:
        raise LiveDataError("Backup.json is empty")

    df_all["time"] = pd.to_datetime(df_all["time"], utc=True)
    last_common = df_all["time"].max()
    rng = pd.date_range(last_common - timedelta(hours=lags-1), last_common, freq="H", tz="UTC")

    feat_order = FEATURES + ["hod_sin","hod_cos","doy_sin","doy_cos"]
    mats = []

    for c in cities:
        d = df_all[df_all["city"] == c.name].copy()
        d = d.set_index("time").reindex(rng)

        for col in FEATURES:
            if col not in d.columns:
                d[col] = 0.0

        for col in ["prcp","snow"]:
            if col not in d.columns:
                d[col] = 0.0
            d[col] = d[col].fillna(0.0)

        if d[["temp","rhum","pres","wspd"]].isna().any().any():  #
            raise LiveDataError(f"Backup.json missing required values for {c.name}")

        d = add_calendar_features(d)
        mats.append(d[feat_order].astype("float32").values)

    X_last = np.stack(mats, axis=1)
    return np.expand_dims(X_last, axis=0)
