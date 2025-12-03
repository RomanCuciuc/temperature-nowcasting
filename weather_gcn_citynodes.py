# This module implements the end-to-end pipeline for graph-based 1-hour nowcasting.
# It covers: parsing the city list, mapping cities to Meteostat JSON files,
# building time-windowed node features, constructing the weighted k-NN graph
# and normalized adjacency (A), defining the WeatherSTGCN model (temporal CNN
# + city embeddings + GCN), training with checkpointing, and inference that
# writes predictions_next_hour.csv.

import os, re, json, math, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# -------------------------- Config --------------------------
FEATURES = ["temp", "rhum", "pres", "wspd", "prcp", "snow"]
TARGETS  = ["temp", "wspd", "rhum", "snow"]

# ---------------------- Helpers: parsing --------------------
# This function: Parces the cities
def parse_cities(cities_file: str) -> List[Dict]:
    pat = re.compile(r".*?([A-Za-z .'-]+,\s*[A-Z]{2})\s*\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)")
    out = []
    with open(cities_file, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line.strip())
            if not m:
                continue
            out.append({"name": m.group(1).strip(),
                        "lat": float(m.group(2)),
                        "lon": float(m.group(3))})
    return out

def _city_token(name: str) -> str:
    return name.split(',')[0].strip().lower()

# This function: Creates the dictionary for cities and List of JSON files with history for each city
def build_city_to_file_map(data_dir: str, cities: List[Dict], map_json: str=None) -> Dict[str, List[str]]:
    data_dir = Path(data_dir)

    if map_json:  #
        with open(map_json, 'r', encoding='utf-8') as f:
            mapping = json.load(f) if map_json.lower().endswith(".json") else None
            if mapping is None:  #
                import yaml
                mapping = yaml.safe_load(f)
        out = {}
        for c in cities:  #
            name = c['name']
            val = mapping.get(name)
            if not val:  #
                continue
            from pathlib import Path as _P
            if isinstance(val, str):  #
                cand = _P(val)
                pth = cand if cand.is_absolute() else data_dir / cand
                paths = [str(pth)]
            else:
                paths = []
                for x in val:  #
                    cand = _P(x)
                    pth = cand if cand.is_absolute() else data_dir / cand
                    paths.append(str(pth))
            out[name] = [p for p in paths if Path(p).exists()]
        return out  # Return the final value/object from this function

    all_paths = list(data_dir.rglob("*.json")) + list(data_dir.rglob("*.JSON"))
    pat = re.compile(r'^(?P<city>[A-Za-z0-9\- ]+?)_(?P<station>[A-Za-z0-9]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.json$', re.IGNORECASE)

    infos = []

    for p in all_paths:  #
        fname = p.name
        m = pat.match(fname)
        if m:  #
            infos.append({"path": str(p),
                          "city_token": m.group("city").strip().lower(),
                          "start": m.group("start"),
                          "end": m.group("end"),
                          "parsed": True})
        else:
            base = fname
            prefix = base.split("_", 1)[0].strip().lower() if "_" in base else base.split(".",1)[0].strip().lower()
            infos.append({"path": str(p),
                          "city_token": prefix,
                          "start": None, "end": None, "parsed": False})

    result = {}

    for c in cities:
        ctok = _city_token(c['name'])
        group = [i for i in infos if i["city_token"] == ctok]

        if not group:
            group = [i for i in infos if Path(i["path"]).name.lower().startswith(ctok)]
        if group:
            group.sort(key=lambda x: (x["start"] is None, x["start"] or Path(x["path"]).name))
            result[c['name']] = [g["path"] for g in group]
        else:
            print(f"[WARN] No JSON files matched for city '{c['name']}' (token='{ctok}').")
    return result

# This function: Parse and normalize the city list (names, lat/lon) and prepare city tokens.
def load_city_files(paths: List[str]) -> pd.DataFrame:
    frames = []

    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        df = pd.DataFrame(obj.get("data", []))

        if df.empty:
            continue
        for col in ["time","temp","rhum","pres","wspd","prcp","snow"]:  #
            if col not in df.columns:  #
                df[col] = np.nan

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df["rhum"] = pd.to_numeric(df["rhum"], errors="coerce")
        df["pres"] = pd.to_numeric(df["pres"], errors="coerce")
        df["wspd"] = pd.to_numeric(df["wspd"], errors="coerce") / 3.6
        df["prcp"] = pd.to_numeric(df["prcp"], errors="coerce").fillna(0.0)
        df["snow"] = pd.to_numeric(df["snow"], errors="coerce").fillna(0.0)
        frames.append(df[["time","temp","rhum","pres","wspd","prcp","snow"]])

    if not frames:
        return pd.DataFrame(columns=["time","temp","rhum","pres","wspd","prcp","snow"])  # Return the final value/object from this function
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df

# This function: processes hourly data for each city, aligning them to a common timeline and creating a unified data "grid."
def align_and_merge(city_to_files: Dict[str, List[str]]):
    series = {}

    for city, flist in city_to_files.items():
        if not flist:
            print(f"[WARN] Empty file list for {city}")
            continue

        df = load_city_files(flist)

        if df.empty:
            print(f"[WARN] No rows after load for {city}")
            continue
        series[city] = df.set_index("time")
    if not series:
        raise ValueError("No city has non-empty data.")
    common = None
    for df in series.values():  #
        common = df.index if common is None else common.intersection(df.index)

    for city in list(series.keys()):
        series[city] = series[city].loc[common].copy()
    return common, series

# This function: adds temporal features from the calendar, such as hour of the day, day of the week,
# month, weekend/holiday flags (if available),
# and cyclical sin/cos time encodings.
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    hod = idx.hour.values
    doy = idx.day_of_year.values
    df["hod_sin"] = np.sin(2*np.pi*hod/24.0)
    df["hod_cos"] = np.cos(2*np.pi*hod/24.0)
    df["doy_sin"] = np.sin(2*np.pi*doy/366.0)
    df["doy_cos"] = np.cos(2*np.pi*doy/366.0)
    return df

# This function: prepares data in the "input → target" format.
# It creates sliding windows of length L from the aligned hourly series across all cities.
# The output is a feature tensor X with shape (B, L, N, F) and a target tensor y with shape (B, N, out_dim)
# for the next-hour values. It also applies normalization/scaling if needed and selects the relevant target fields.
def make_supervised_tensors(common_idx, series: Dict[str,pd.DataFrame], cities: List[Dict], lags: int):
    city_names = [c["name"] for c in cities if c["name"] in series]
    mats = []

    for name in city_names:  #
        df = series[name].copy()
        df = add_calendar_features(df)
        feat_cols = FEATURES + ["hod_sin","hod_cos","doy_sin","doy_cos"]
        mats.append(df[feat_cols].astype("float32").values)

    M = np.stack(mats, axis=1)   # [T,N,F]
    T, N, Fdim = M.shape
    X_list, y_list = [], []

    for t in range(lags, T-1):
        X_list.append(M[t-lags:t, :, :])
        vals = []
        for name in city_names:
            df = series[name]
            vals.append(df.iloc[t+1][TARGETS].values.astype(np.float32))
        y_list.append(np.stack(vals, axis=0))
    X = torch.tensor(np.stack(X_list, axis=0))
    y = torch.tensor(np.stack(y_list, axis=0))

    return X, y, city_names

# This function: calculates the great-circle distance between 2 points on the Earth's surface
# using the Haversine formula. The result is in kilometers (with R ~ 6371 km).
# Inputs: lat1, lon1, lat2, lon2 in degrees.
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# This function: builds the graph matrix for the Graph Convolutional Network (GCN).
# It calculates pairwise distances between cities based on their coordinates, selects the k-nearest neighbors for each city,
# and converts these distances into edge weights
def build_A_hat(coords: np.ndarray, k: int = 6, sigma_km: float = 50.0):
    N = coords.shape[0]
    W = np.zeros((N,N), dtype=np.float32)
    lat = coords[:,0]; lon = coords[:,1]
    dist = np.zeros((N,N), dtype=np.float32)

    for i in range(N):
        dist[i,:] = haversine_km(lat[i], lon[i], lat, lon)

    for i in range(N):
        idx = np.argsort(dist[i])[1:k+1]
        for j in idx:  #
            w = math.exp(-dist[i,j] / max(1e-6, sigma_km))
            W[i,j] = max(W[i,j], w)
            W[j,i] = max(W[j,i], w)

    W = W + np.eye(N, dtype=np.float32)
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-8))).astype(np.float32)
    A_hat = D_inv_sqrt @ W @ D_inv_sqrt
    return A_hat

# This class: takes node features, applies a linear transformation,
# and mixes them with the features of neighboring nodes using the normalized graph adjacency matrix ((\hat A)).
# This results in a weighted sum of the "messages" from a node’s neighbors.
# It then applies ReLU and, dropout/normalization, producing updated node features for the next layer.
class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A_hat):
        X = self.dropout(X)
        X = A_hat @ X
        return F.relu(self.lin(X))

# This class: Model class: temporal CNN encoder + city embeddings + GCN propagation + output head.
class WeatherSTGCN(nn.Module):

    def __init__(self, num_nodes, in_feat, tcn_channels=64, emb_dim=16,
                 gcn_hidden=64, out_dim=2, gcn_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_feat = in_feat
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels=in_feat, out_channels=tcn_channels,
                      kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=tcn_channels, out_channels=tcn_channels,
                      kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(),
        )
        self.city_emb = nn.Embedding(num_nodes, emb_dim)
        gcn_in = tcn_channels + emb_dim
        self.gcns = nn.ModuleList([
            GCNLayer(gcn_in if i==0 else gcn_hidden, gcn_hidden, dropout=dropout)
            for i in range(gcn_layers)  #
        ])
        self.head = nn.Linear(gcn_hidden, out_dim)

    def forward(self, X, A_hat):
        B,L,N,Fdim = X.shape
        x = X.permute(0,3,1,2)  # [B,F,L,N]
        x = self.temporal(x)    # [B,C,L,N]
        x = x.mean(dim=2)       # [B,C,N]
        x = x.permute(0,2,1)    # [B,N,C]

        device = x.device
        city_idx = torch.arange(N, device=device)

        emb = self.city_emb(city_idx).unsqueeze(0).expand(B, N, -1)
        x = torch.cat([x, emb], dim=-1)  # [B,N,C+emb]
        A = torch.tensor(A_hat, dtype=torch.float32, device=device)

        outs = []
        for b in range(B):
            h = x[b]
            for g in self.gcns:
                h = g(h, A)
            outs.append(self.head(h))
        return torch.stack(outs, dim=0)

# This function: Run inference (optionally with live NWS data) and write predictions to CSV.
def infer_once(model_ckpt: str, X_last: torch.Tensor, A_hat: np.ndarray, device="cpu"):
    model_ckpt = os.path.abspath(model_ckpt)

    if not os.path.exists(model_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {model_ckpt}")
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)  # Load checkpoint weights from disk for inference/resume

    _, L, N, Fdim = X_last.shape
    out_dim = len(TARGETS)
    model = WeatherSTGCN(num_nodes=N, in_feat=Fdim, tcn_channels=64, emb_dim=16,
                         gcn_hidden=64, out_dim=out_dim, gcn_layers=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    A_hat_t = torch.tensor(A_hat, dtype=torch.float32, device=device)
    with torch.no_grad():
        yhat = model(X_last.to(device), A_hat_t).cpu().numpy()
    return yhat[0]

# This function: Train the model with Adam/AdamW,
# evaluate on val split, save checkpoint.
def train_eval(X, y, A_hat, train_frac=0.8, val_frac=0.1, epochs=20, lr=1e-3, batch_size=32, device="cpu", out_path="best_model.pt"):
    B, L, N, Fdim = X.shape
    n_train = int(B * train_frac)
    n_val   = int(B * val_frac)
    n_test  = B - n_train - n_val
    idx_train = slice(0, n_train)
    idx_val   = slice(n_train, n_train + n_val)
    idx_test  = slice(n_train + n_val, B)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    model = WeatherSTGCN(num_nodes=N, in_feat=Fdim, tcn_channels=64, emb_dim=16,
                         gcn_hidden=64, out_dim=y.shape[-1], gcn_layers=2, dropout=0.1).to(device)
    A_hat_t = torch.tensor(A_hat, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = float("inf")
    patience, bad = 5, 0

    def batches(Xt, yt, bs):
        for i in range(0, Xt.shape[0], bs):  #
            yield Xt[i:i+bs], yt[i:i+bs]

    out_path = os.path.abspath(out_path)

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in batches(X_train.to(device), y_train.to(device), batch_size):  #
            opt.zero_grad()
            pred = model(xb, A_hat_t)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, X_train.shape[0])

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val.to(device), A_hat_t)
            val_loss = F.mse_loss(pred_val, y_val.to(device)).item()

        print(f"[Epoch {ep:02d}] train MSE={tr_loss:.4f} | val MSE={val_loss:.4f}")

        if val_loss < best_val - 1e-5:  #
            best_val = val_loss
            bad = 0
            torch.save({"model": model.state_dict(), "A_hat": A_hat, "meta": {"epochs": ep}}, out_path)  # Save checkpoint weights to disk (best_model.pt)
            print(f"[CKPT] saved {out_path}")
        else:
            bad += 1
            if bad >= patience:  #
                print("Early stopping.")
                break

    ckpt = torch.load(out_path, map_location=device, weights_only=False)  # Load checkpoint weights from disk for inference/resume

    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred_test = model(X_test.to(device), A_hat_t).cpu()
    yte = y_test.cpu()

    def mae(a,b): return (a-b).abs().mean().item()
    def rmse(a,b): return torch.sqrt(((a-b)**2).mean()).item()

    name_map = {"temp": "temp_C_next", "wspd": "wind_ms_next", "rhum": "rhum_next", "snow": "snow_next"}

    for j, key in enumerate(TARGETS[:y.shape[-1]]):  #
        m1 = mae(pred_test[...,j], yte[...,j])
        m2 = rmse(pred_test[...,j], yte[...,j])
        print(f"[TEST] {name_map.get(key, key+'_next')}: MAE={m1:.3f}, RMSE={m2:.3f}")
    return model

# This function: Command-line entry point that orchestrates training or inference.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--map_json", type=str, default=None)
    ap.add_argument("--infer_from_checkpoint", type=str, default=None)
    ap.add_argument("--live_nws", action="store_true")
    ap.add_argument("--nws_user_agent", type=str, default="weather-gnn/1.0 (roman@example.com)")
    ap.add_argument("--backup_json", type=str, default=None)

    ap.add_argument("--live_nws_mode", choices=["nearest_station", "latlon_blend"], default="latlon_blend",
                        help="How to assemble live hourly data per city: 'nearest_station' (legacy) or 'latlon_blend' (top-K stations by inverse distance).")

    ap.add_argument("--nws_blend_k", type=int, default=2,
                        help="Top-K nearest stations to blend when --live_nws_mode=latlon_blend.")

    ap.add_argument("--nws_max_stations", type=int, default=5,
                        help="How many candidate stations to fetch per city before selecting top-K to blend.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print("[1/6] Parse city list ...")
    cities = parse_cities(args.cities)
    print(f"  Loaded {len(cities)} cities.")

    print("[2/6] Map cities to JSON files ...")
    c2f = build_city_to_file_map(args.data_dir, cities, map_json=args.map_json)
    cities = [c for c in cities if c['name'] in c2f and c2f[c['name']]]

    if not cities:
        raise SystemExit("No matching JSON files for the provided cities.")
    print(f"  Matched {len(cities)} cities with data.")

    print("[3/6] Align & merge ...")
    common_idx, series = align_and_merge(c2f)
    print(f"  Common hours: {len(common_idx)}")

    print("[4/6] Build kNN graph ...")
    coords = np.array([[c['lat'], c['lon']] for c in cities], dtype=np.float32)
    A_hat = build_A_hat(coords, k=args.k, sigma_km=50.0)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "A_hat.npy"), A_hat)  # Persist normalized adjacency A
    print(f"  A_hat saved: {os.path.join(args.output_dir, 'A_hat.npy')}")

    print("[5/6] Build tensors ...")
    X, y, city_names = make_supervised_tensors(common_idx, series, cities, lags=args.lags)
    print(f"  X={tuple(X.shape)}, y={tuple(y.shape)}, cities={len(city_names)}")

    model_path = os.path.join(os.path.abspath(args.output_dir), "best_model.pt")
    print(f"[INFO] Checkpoint path: {model_path}")

    if args.infer_from_checkpoint:  #
        print("[6/6] Inference from checkpoint ...")
        from live_nws_fetcher import build_live_window, CitySpec

        city_specs = [CitySpec(name=c['name'], lat=c['lat'], lon=c['lon']) for c in cities]
        #city_specs=city_specs[:4]
        if args.live_nws:  #
            print("Fetching live NWS window ...")
            X_last_np = build_live_window(cities=city_specs, lags=args.lags, user_agent=args.nws_user_agent, backup_json=args.backup_json)
            X_last = torch.tensor(X_last_np, dtype=torch.float32)

        else:
            X_last = X[-1:].clone()

        # Run inference
        preds = infer_once(args.infer_from_checkpoint, X_last, A_hat, device=device)

        # Emit CSV with pretty column names
        rows = []

        future_str = utils.get_current_ny_date()
        idx = {k: j for j, k in enumerate(TARGETS)}  # For exmpl. {'temp':0,'wspd':1,'rhum':2,'snow':3}

        for i, name in enumerate(city_names):  #
            # getting predictions based on keys
            temp_c = float(preds[i, idx["temp"]]) if "temp" in idx else None
            wind_ms = float(preds[i, idx["wspd"]]) if "wspd" in idx else None
            rhum = float(preds[i, idx["rhum"]]) if "rhum" in idx else None
            snow = float(preds[i, idx["snow"]]) if "snow" in idx else None

            # from C -> F
            temp_f = (temp_c * 9.0 / 5.0 + 32.0) if temp_c is not None else None
            #temp_f = (temp_c * 8.5 / 37.0)
            temp_f = temp_f -9
            # creating a row in the desired column order
            row = {
                "Node Name": name,
                "Future Date": future_str,
                "Air temperature (F)": None if temp_f is None else round(temp_f, 1),
                "Wind Speed (m/s)": None if wind_ms is None else round(wind_ms, 2),
                "Humidity (%)": None if rhum is None else round(rhum, 1),
                "Snowfall (mm/h)": None if snow is None else round(snow, 2),
            }
            rows.append(row)

        out_csv = os.path.join(args.output_dir, "predictions_next_hour.csv")
        cols = ["Node Name", "Future Date", "Air temperature (F)", "Wind Speed (m/s)", "Humidity (%)", "Snowfall (mm/h)"]
        pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)

        print(f"Saved: {out_csv}")
        return

    print("[6/6] Train/Eval model ...")
    _ = train_eval(X, y, A_hat, train_frac=0.8, val_frac=0.1,
                   epochs=args.epochs, lr=1e-3, batch_size=32,
                   device=device, out_path=model_path)

if __name__ == "__main__":
    main()
