import os
import io
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Geod
import pydeck as pdk
PASSWORD = st.secrets["APP_PASSWORD"]

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pwd = st.text_input("Enter password", type="password")
    if pwd == PASSWORD:
        st.session_state.auth = True
        st.rerun()
    else:
        st.stop()

# -----------------------------
# PATHS (RELATIVE)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

NC_PATH = DATA_DIR / "rainfall" / "rf_daily_1901_2023.nc"
SHAPE_ROOT = DATA_DIR / "shapefiles"

DEFAULT_RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
DEFAULT_DURATIONS_HR = [1, 2, 3, 6, 12, 24]
DURATION_HR_BASE = 24.0


# -----------------------------
# UI SETUP
# -----------------------------
st.set_page_config(page_title="IMD IDF Analysis", layout="centered")
st.title("IMD IDF Analysis")
st.caption("Site boundary → nearest IMD grid → extreme value fit → IDF tables/plots (with optional climate uplift)")


# -----------------------------
# HELPERS
# -----------------------------
def pretty_crs(crs):
    if crs is None:
        return "UNKNOWN"
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else str(crs)


def list_demo_sites(shape_root: Path) -> Dict[str, Path]:
    demos = {}
    if not shape_root.exists():
        return demos
    for shp in shape_root.rglob("*.shp"):
        rel = shp.relative_to(shape_root)
        label = str(rel).replace("\\", "/")
        demos[label] = shp
    return demos


def read_vector_upload(uploaded_file) -> gpd.GeoDataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "upload.zip")
            with open(zpath, "wb") as f:
                f.write(data)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(td)

            shp = None
            for root, _, files in os.walk(td):
                for fn in files:
                    if fn.lower().endswith(".shp"):
                        shp = os.path.join(root, fn)
                        break
                if shp:
                    break

            if not shp:
                raise ValueError("ZIP has no .shp. Zip all parts: .shp .shx .dbf .prj")
            return gpd.read_file(shp)

    if name.endswith(".gpkg"):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, uploaded_file.name)
            with open(fpath, "wb") as f:
                f.write(data)
            return gpd.read_file(fpath)

    if name.endswith(".geojson") or name.endswith(".json"):
        return gpd.read_file(io.BytesIO(data))

    raise ValueError("Unsupported vector format. Use .zip (shp), .gpkg, or .geojson/.json")


@st.cache_resource
def load_ds(nc_path: Path) -> xr.Dataset:
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")
    return xr.open_dataset(nc_path)


def detect_dims_and_var(ds: xr.Dataset) -> Tuple[str, str, str, str]:
    time_name = None
    for c in list(ds.coords):
        if np.issubdtype(ds[c].dtype, np.datetime64):
            time_name = c
            break
    if time_name is None:
        for guess in ["TIME", "time", "Date", "date"]:
            if guess in ds.coords:
                time_name = guess
                break

    lat_name = None
    lon_name = None
    for c in list(ds.coords):
        lc = c.lower()
        if lat_name is None and "lat" in lc:
            lat_name = c
        if lon_name is None and "lon" in lc:
            lon_name = c

    var_name = "RAINFALL" if "RAINFALL" in ds.data_vars else None
    if var_name is None:
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.number):
                var_name = v
                break

    if not (time_name and lat_name and lon_name and var_name):
        raise ValueError(
            f"Could not detect required structure.\n"
            f"time={time_name}, lat={lat_name}, lon={lon_name}, var={var_name}\n"
            f"Coords: {list(ds.coords)}\nVars: {list(ds.data_vars)}"
        )

    return time_name, lat_name, lon_name, var_name


def centroid_latlon(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    if gdf.crs is None:
        raise ValueError("Vector CRS missing. Ensure .prj exists or use GPKG/GeoJSON with CRS.")
    gdf_wgs = gdf.to_crs(epsg=4326)
    c = gdf_wgs.geometry.unary_union.centroid
    return float(c.y), float(c.x)


def nearest_grid(ds: xr.Dataset, lat_name: str, lon_name: str, lat0: float, lon0: float) -> Tuple[float, float]:
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("This app expects 1D lat/lon coordinate arrays.")

    i = int(np.argmin(np.abs(lat_vals - lat0)))
    j = int(np.argmin(np.abs(lon_vals - lon0)))
    return float(lat_vals[i]), float(lon_vals[j])


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def build_ams_and_fit_gev(
    ds: xr.Dataset,
    var_name: str,
    time_name: str,
    lat_name: str,
    lon_name: str,
    lat_g: float,
    lon_g: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rain = ds[var_name].sel({lat_name: lat_g, lon_name: lon_g}, method="nearest")
    df = rain.to_dataframe().reset_index()
    df = df[[time_name, var_name]].rename(columns={time_name: "date", var_name: "rain_mm"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()

    df["year"] = df["date"].dt.year
    ams = df.groupby("year", as_index=False)["rain_mm"].max().rename(columns={"rain_mm": "annual_max_mm"})
    ams = ams.dropna()

    c, loc, scale = genextreme.fit(ams["annual_max_mm"].values)
    fit = {"shape_c": float(c), "loc": float(loc), "scale": float(scale)}
    return ams, fit


def gev_quantiles_24h(fit: Dict[str, float], return_periods: List[int]) -> np.ndarray:
    c, loc, scale = fit["shape_c"], fit["loc"], fit["scale"]
    p = 1 - 1 / np.array(return_periods, dtype=float)
    return genextreme.ppf(p, c, loc=loc, scale=scale)


def idf_from_24h_depths(P24: np.ndarray, return_periods: List[int], durations_hr: List[float], exponent_n: float) -> pd.DataFrame:
    durations_hr = sorted([float(d) for d in durations_hr])
    rows = []
    for rp, p24 in zip(return_periods, P24):
        for d in durations_hr:
            pdur = float(p24) * (d / DURATION_HR_BASE) ** exponent_n
            inten = pdur / d
            rows.append({
                "return_period_yr": int(rp),
                "duration_hr": float(d),
                "P_mm": round(pdur, 3),
                "I_mm_per_hr": round(inten, 3)
            })
    return pd.DataFrame(rows)


def pivot_idf(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df_long.pivot(index="duration_hr", columns="return_period_yr", values=value_col).reset_index()
    out.columns.name = None
    return out


def plot_depth_vs_rp(return_periods: List[int], P24: np.ndarray, title: str):
    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(return_periods, P24, marker="o")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("24-hour rainfall depth (mm)")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_intensity_vs_duration(df_long: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    for rp in sorted(df_long["return_period_yr"].unique()):
        sub = df_long[df_long["return_period_yr"] == rp].sort_values("duration_hr")
        ax.plot(sub["duration_hr"], sub["I_mm_per_hr"], marker="o", label=f"{int(rp)} yr")

    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Intensity (mm/hr)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_xscale("log")
    ax.legend(title="Return period", ncols=2, fontsize=9)
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    buf.seek(0)
    return buf.getvalue()


def choose_map_style() -> str:
    """
    Mapbox styles look best but need MAPBOX_API_KEY.
    If missing, fall back to a token-free CARTO basemap.
    """
    if os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_ACCESS_TOKEN"):
        return "mapbox://styles/mapbox/dark-v11"
    return "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


def make_pydeck_map(gdf, lat0, lon0, lat_g, lon_g, view_mode, show_grid=True):
    gdf_wgs = gdf.to_crs(4326)
    geojson = gdf_wgs.__geo_interface__

    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_line_color=[255, 255, 255],
        get_fill_color=[60, 120, 180, 80],
        line_width_min_pixels=3,
    )

    pts = pd.DataFrame([
        {"name": "Site centroid", "lat": lat0, "lon": lon0, "color": [0, 200, 0]},
        {"name": "Nearest IMD grid", "lat": lat_g, "lon": lon_g, "color": [255, 80, 80]},
    ])

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="[lon, lat]",
        get_radius=650,
        get_fill_color="color",
        pickable=True,
    )

    grid_layers = []
    if show_grid:
        d = 0.125  # half-grid for 0.25°
        grid_poly = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "IMD grid cell"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_g - d, lat_g - d],
                        [lon_g + d, lat_g - d],
                        [lon_g + d, lat_g + d],
                        [lon_g - d, lat_g + d],
                        [lon_g - d, lat_g - d],
                    ]]
                }
            }]
        }

        grid_layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=grid_poly,
                stroked=True,
                filled=False,
                get_line_color=[255, 0, 0],
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    if view_mode == "India":
        view_state = pdk.ViewState(latitude=22.5, longitude=79.0, zoom=3.6, transition_duration=1200)
    elif view_mode == "Grid":
        view_state = pdk.ViewState(latitude=lat_g, longitude=lon_g, zoom=7.5, transition_duration=1200)
    else:
        view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=13.5, transition_duration=1200)

    tooltip = {"text": "{name}\nLat: {lat}\nLon: {lon}"}

    return pdk.Deck(
        layers=[boundary_layer, point_layer] + grid_layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=choose_map_style(),
    )


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    use_demo = st.checkbox("Use demo boundary", value=False)
    demo_sites = list_demo_sites(SHAPE_ROOT)

    demo_choice = None
    if use_demo:
        if not demo_sites:
            st.warning("No demo shapefiles found under data/shapefiles/")
        else:
            demo_choice = st.selectbox("Choose a demo site", list(demo_sites.keys()))
            st.caption(f"Using: {demo_choice}")

    st.divider()
    st.header("IDF settings")

    rp = st.multiselect(
        "Return periods (years)",
        options=[2, 5, 10, 25, 50, 100, 200],
        default=DEFAULT_RETURN_PERIODS,
    )

    durations_hr = st.multiselect(
        "Durations (hours)",
        options=[0.5, 1, 2, 3, 6, 12, 24],
        default=DEFAULT_DURATIONS_HR,
    )

    exponent_n = st.slider(
        "Duration scaling exponent (n)",
        min_value=0.15,
        max_value=0.55,
        value=0.33,
        step=0.01,
        help="Shorter durations are derived using P(d)=P24*(d/24)^n.",
    )

    st.divider()
    st.header("Climate uplift (optional)")
    apply_uplift = st.checkbox("Apply uplift", value=False)
    uplift_pct = st.slider("Uplift (%)", 5, 30, 15, 1) if apply_uplift else 0


# -----------------------------
# STEP 1: SITE BOUNDARY
# -----------------------------
st.subheader("1) Site boundary")

if use_demo and demo_choice and demo_choice in demo_sites:
    try:
        gdf = gpd.read_file(demo_sites[demo_choice])
        st.info("Demo boundary loaded (bundled with this repo).")
        st.caption(f"Repo path: data/shapefiles/{demo_choice}")
    except Exception as e:
        st.error(f"Failed to load demo shapefile: {e}")
        st.stop()
else:
    up = st.file_uploader("Upload boundary (.zip shp / .gpkg / .geojson)", type=["zip", "gpkg", "geojson", "json"])
    if up is None:
        st.stop()
    try:
        gdf = read_vector_upload(up)
    except Exception as e:
        st.error(f"Boundary read failed: {e}")
        st.stop()

if gdf.empty:
    st.error("Boundary file has no features.")
    st.stop()

st.success(f"Loaded boundary | Features: {len(gdf)} | CRS: {pretty_crs(gdf.crs)}")


# -----------------------------
# STEP 2: RAINFALL DATA
# -----------------------------
st.subheader("2) Rainfall dataset")

try:
    ds = load_ds(NC_PATH)
except Exception as e:
    st.error(f"Could not open NetCDF at {NC_PATH}: {e}")
    st.stop()

time_name, lat_name, lon_name, var_name = detect_dims_and_var(ds)

tmin = str(pd.to_datetime(ds[time_name].values[0]).date())
tmax = str(pd.to_datetime(ds[time_name].values[-1]).date())

colA, colB = st.columns(2)
with colA:
    st.write("Variable:", f"**{var_name}**")
    st.write("Time range:", f"**{tmin} → {tmax}**")
with colB:
    st.write("Grid:", f"**{ds[lat_name].size} lat × {ds[lon_name].size} lon**")
    st.write("NetCDF:", f"`{NC_PATH.as_posix()}`")


# -----------------------------
# STEP 3: NEAREST GRID + MAP
# -----------------------------
st.subheader("3) Location check (India → Site → Grid)")

try:
    lat0, lon0 = centroid_latlon(gdf)
    lat_g, lon_g = nearest_grid(ds, lat_name, lon_name, lat0, lon0)
    dkm = distance_km(lat0, lon0, lat_g, lon_g)
except Exception as e:
    st.error(f"Centroid / nearest grid failed: {e}")
    st.stop()

st.write(f"Site centroid: **{lat0:.5f}, {lon0:.5f}**")
st.write(f"Nearest IMD grid: **{lat_g:.5f}, {lon_g:.5f}** (distance ≈ **{dkm:.2f} km**)")

col1, col2 = st.columns(2)
with col1:
    view_mode = st.radio("Map view", ["India", "Site", "Grid"], horizontal=True, index=1, key="map_view_mode")
with col2:
    show_grid = st.checkbox("Show IMD grid cell", value=True, key="show_imd_grid")

st.pydeck_chart(
    make_pydeck_map(gdf, lat0, lon0, lat_g, lon_g, view_mode, show_grid),
    use_container_width=True
)


# -----------------------------
# STEP 4: IDF (BASELINE)
# -----------------------------
st.subheader("4) IDF computation (baseline)")

if not rp:
    st.warning("Pick at least one return period.")
    st.stop()
if not durations_hr:
    st.warning("Pick at least one duration.")
    st.stop()

with st.spinner("Building annual maxima and fitting GEV..."):
    ams_df, fit = build_ams_and_fit_gev(ds, var_name, time_name, lat_name, lon_name, lat_g, lon_g)
    P24 = gev_quantiles_24h(fit, rp)

idf24 = pd.DataFrame({
    "return_period_yr": rp,
    "P24_mm": np.round(P24, 3),
    "I24_mm_per_hr": np.round(P24 / DURATION_HR_BASE, 3),
})

idf_long = idf_from_24h_depths(P24, rp, durations_hr, exponent_n)
idf_depth_matrix = pivot_idf(idf_long, "P_mm")
idf_int_matrix = pivot_idf(idf_long, "I_mm_per_hr")

st.success("Baseline IDF generated")

with st.expander("GEV fit parameters (for reference)"):
    st.write(f"shape(c): **{fit['shape_c']:.4f}**")
    st.write(f"loc: **{fit['loc']:.4f}**")
    st.write(f"scale: **{fit['scale']:.4f}**")

st.markdown("#### 24-hour results")
st.dataframe(idf24, use_container_width=True)

st.markdown("#### Multi-duration IDF (Depth matrix)")
st.dataframe(idf_depth_matrix, use_container_width=True)

st.markdown("#### Multi-duration IDF (Intensity matrix)")
st.dataframe(idf_int_matrix, use_container_width=True)

fig24 = plot_depth_vs_rp(rp, P24, "24-hour IDF (baseline)")
st.pyplot(fig24, clear_figure=True)
png24 = fig_to_png_bytes(fig24)

figI = plot_intensity_vs_duration(idf_long, "IDF curves (Intensity vs Duration, baseline)")
st.pyplot(figI, clear_figure=True)
pngI = fig_to_png_bytes(figI)

st.markdown("#### Downloads (baseline)")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("IDF (long) CSV", data=df_to_csv_bytes(idf_long), file_name="idf_long_baseline.csv", mime="text/csv")
with c2:
    st.download_button("Plot PNG (24h)", data=png24, file_name="idf_24h_baseline.png", mime="image/png")
with c3:
    st.download_button("Plot PNG (IDF)", data=pngI, file_name="idf_intensity_baseline.png", mime="image/png")

excel_baseline = to_excel_bytes({
    "idf_24h": idf24,
    "idf_long": idf_long,
    "depth_matrix": idf_depth_matrix,
    "intensity_matrix": idf_int_matrix,
    "ams": ams_df
})
st.download_button(
    "Download Excel (baseline)",
    data=excel_baseline,
    file_name="idf_baseline.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# -----------------------------
# STEP 5: UPLIFT (OPTIONAL)
# -----------------------------
st.subheader("5) Climate uplift (optional)")

if apply_uplift:
    factor = 1.0 + uplift_pct / 100.0

    idf24_u = idf24.copy()
    idf24_u["P24_mm"] = np.round(idf24_u["P24_mm"] * factor, 3)
    idf24_u["I24_mm_per_hr"] = np.round(idf24_u["I24_mm_per_hr"] * factor, 3)

    idf_long_u = idf_long.copy()
    idf_long_u["P_mm"] = np.round(idf_long_u["P_mm"] * factor, 3)
    idf_long_u["I_mm_per_hr"] = np.round(idf_long_u["I_mm_per_hr"] * factor, 3)

    depth_u = pivot_idf(idf_long_u, "P_mm")
    int_u = pivot_idf(idf_long_u, "I_mm_per_hr")

    st.success(f"Applied uplift: +{uplift_pct}%")

    st.markdown("#### Uplifted 24-hour results")
    st.dataframe(idf24_u, use_container_width=True)

    st.markdown("#### Uplifted multi-duration matrices")
    st.dataframe(depth_u, use_container_width=True)
    st.dataframe(int_u, use_container_width=True)

    fig24u = plot_depth_vs_rp(rp, idf24_u["P24_mm"].values, f"24-hour IDF (+{uplift_pct}% uplift)")
    st.pyplot(fig24u, clear_figure=True)
    png24u = fig_to_png_bytes(fig24u)

    figIu = plot_intensity_vs_duration(idf_long_u, f"IDF curves (Intensity vs Duration, +{uplift_pct}% uplift)")
    st.pyplot(figIu, clear_figure=True)
    pngIu = fig_to_png_bytes(figIu)

    st.markdown("#### Downloads (uplifted)")
    u1, u2, u3 = st.columns(3)
    with u1:
        st.download_button("IDF (long) CSV", data=df_to_csv_bytes(idf_long_u), file_name=f"idf_long_uplift_{uplift_pct}pct.csv", mime="text/csv")
    with u2:
        st.download_button("Plot PNG (24h)", data=png24u, file_name=f"idf_24h_uplift_{uplift_pct}pct.png", mime="image/png")
    with u3:
        st.download_button("Plot PNG (IDF)", data=pngIu, file_name=f"idf_intensity_uplift_{uplift_pct}pct.png", mime="image/png")

    excel_uplift = to_excel_bytes({
        "idf_24h_uplift": idf24_u,
        "idf_long_uplift": idf_long_u,
        "depth_matrix_uplift": depth_u,
        "intensity_matrix_uplift": int_u,
        "ams": ams_df
    })
    st.download_button(
        "Download Excel (uplifted)",
        data=excel_uplift,
        file_name=f"idf_uplift_{uplift_pct}pct.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Uplift is off. Enable it from the sidebar if needed.")

st.caption("Note: This demo uses daily rainfall (24h). Shorter durations are derived via a transparent scaling rule with adjustable exponent n.")
import os
import io
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import geopandas as gpd
from pyproj import Geod
import pydeck as pdk
import requests


# -----------------------------
# PATHS (RELATIVE)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

NC_URL = "https://github.com/prasadOS/imd-idf-app/releases/download/data-v1/rf_daily_1901_2023.nc"
NC_PATH = DATA_DIR / "rainfall" / "rf_daily_1901_2023.nc"

SHAPE_ROOT = DATA_DIR / "shapefiles"

DEFAULT_RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
DEFAULT_DURATIONS_HR = [1, 2, 3, 6, 12, 24]
DURATION_HR_BASE = 24.0


# -----------------------------
# ENSURE NETCDF EXISTS
# -----------------------------
def ensure_nc_available():
    (DATA_DIR / "rainfall").mkdir(parents=True, exist_ok=True)

    if NC_PATH.exists():
        return

    with st.spinner("Downloading IMD rainfall dataset (one-time setup, ~350 MB)..."):
        r = requests.get(NC_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(NC_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


st.set_page_config(page_title="IMD IDF Analysis", layout="centered")
st.title("IMD IDF Analysis")
st.caption(
    "Site boundary → nearest IMD grid → extreme value fit → "
    "IDF tables/plots (with optional climate uplift)"
)

st.markdown("### Choose data source")

use_demo = st.checkbox(
    "Use demo IMD dataset (data-v1, ~350 MB download)",
    value=False
)

uploaded_nc = st.file_uploader(
    "Or upload your own IMD NetCDF (.nc)",
    type=["nc"]
)

nc_path = None

if uploaded_nc is not None:
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(uploaded_nc.read())
        nc_path = Path(tmp.name)

elif use_demo:
    ensure_nc_available()
    nc_path = NC_PATH

else:
    st.info("Select a data source to proceed.")
    st.stop()



NC_PATH = DATA_DIR / "rainfall" / "rf_daily_1901_2023.nc"
SHAPE_ROOT = DATA_DIR / "shapefiles"

DEFAULT_RETURN_PERIODS = [2, 5, 10, 25, 50, 100]
DEFAULT_DURATIONS_HR = [1, 2, 3, 6, 12, 24]
DURATION_HR_BASE = 24.0


# -----------------------------
# UI SETUP
# -----------------------------
st.set_page_config(page_title="IMD IDF Analysis", layout="centered")
st.title("IMD IDF Analysis")
st.caption("Site boundary → nearest IMD grid → extreme value fit → IDF tables/plots (with optional climate uplift)")


# -----------------------------
# HELPERS
# -----------------------------
def pretty_crs(crs):
    if crs is None:
        return "UNKNOWN"
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else str(crs)


def list_demo_sites(shape_root: Path) -> Dict[str, Path]:
    demos = {}
    if not shape_root.exists():
        return demos
    for shp in shape_root.rglob("*.shp"):
        rel = shp.relative_to(shape_root)
        label = str(rel).replace("\\", "/")
        demos[label] = shp
    return demos


def read_vector_upload(uploaded_file) -> gpd.GeoDataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "upload.zip")
            with open(zpath, "wb") as f:
                f.write(data)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(td)

            shp = None
            for root, _, files in os.walk(td):
                for fn in files:
                    if fn.lower().endswith(".shp"):
                        shp = os.path.join(root, fn)
                        break
                if shp:
                    break

            if not shp:
                raise ValueError("ZIP has no .shp. Zip all parts: .shp .shx .dbf .prj")
            return gpd.read_file(shp)

    if name.endswith(".gpkg"):
        with tempfile.TemporaryDirectory() as td:
            fpath = os.path.join(td, uploaded_file.name)
            with open(fpath, "wb") as f:
                f.write(data)
            return gpd.read_file(fpath)

    if name.endswith(".geojson") or name.endswith(".json"):
        return gpd.read_file(io.BytesIO(data))

    raise ValueError("Unsupported vector format. Use .zip (shp), .gpkg, or .geojson/.json")


@st.cache_resource
def load_ds(nc_path: Path) -> xr.Dataset:
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")
    return xr.open_dataset(nc_path)


def detect_dims_and_var(ds: xr.Dataset) -> Tuple[str, str, str, str]:
    time_name = None
    for c in list(ds.coords):
        if np.issubdtype(ds[c].dtype, np.datetime64):
            time_name = c
            break
    if time_name is None:
        for guess in ["TIME", "time", "Date", "date"]:
            if guess in ds.coords:
                time_name = guess
                break

    lat_name = None
    lon_name = None
    for c in list(ds.coords):
        lc = c.lower()
        if lat_name is None and "lat" in lc:
            lat_name = c
        if lon_name is None and "lon" in lc:
            lon_name = c

    var_name = "RAINFALL" if "RAINFALL" in ds.data_vars else None
    if var_name is None:
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.number):
                var_name = v
                break

    if not (time_name and lat_name and lon_name and var_name):
        raise ValueError(
            f"Could not detect required structure.\n"
            f"time={time_name}, lat={lat_name}, lon={lon_name}, var={var_name}\n"
            f"Coords: {list(ds.coords)}\nVars: {list(ds.data_vars)}"
        )

    return time_name, lat_name, lon_name, var_name


def centroid_latlon(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    if gdf.crs is None:
        raise ValueError("Vector CRS missing. Ensure .prj exists or use GPKG/GeoJSON with CRS.")
    gdf_wgs = gdf.to_crs(epsg=4326)
    c = gdf_wgs.geometry.unary_union.centroid
    return float(c.y), float(c.x)


def nearest_grid(ds: xr.Dataset, lat_name: str, lon_name: str, lat0: float, lon0: float) -> Tuple[float, float]:
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("This app expects 1D lat/lon coordinate arrays.")

    i = int(np.argmin(np.abs(lat_vals - lat0)))
    j = int(np.argmin(np.abs(lon_vals - lon0)))
    return float(lat_vals[i]), float(lon_vals[j])


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def build_ams_and_fit_gev(
    ds: xr.Dataset,
    var_name: str,
    time_name: str,
    lat_name: str,
    lon_name: str,
    lat_g: float,
    lon_g: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rain = ds[var_name].sel({lat_name: lat_g, lon_name: lon_g}, method="nearest")
    df = rain.to_dataframe().reset_index()
    df = df[[time_name, var_name]].rename(columns={time_name: "date", var_name: "rain_mm"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()

    df["year"] = df["date"].dt.year
    ams = df.groupby("year", as_index=False)["rain_mm"].max().rename(columns={"rain_mm": "annual_max_mm"})
    ams = ams.dropna()

    c, loc, scale = genextreme.fit(ams["annual_max_mm"].values)
    fit = {"shape_c": float(c), "loc": float(loc), "scale": float(scale)}
    return ams, fit


def gev_quantiles_24h(fit: Dict[str, float], return_periods: List[int]) -> np.ndarray:
    c, loc, scale = fit["shape_c"], fit["loc"], fit["scale"]
    p = 1 - 1 / np.array(return_periods, dtype=float)
    return genextreme.ppf(p, c, loc=loc, scale=scale)


def idf_from_24h_depths(P24: np.ndarray, return_periods: List[int], durations_hr: List[float], exponent_n: float) -> pd.DataFrame:
    durations_hr = sorted([float(d) for d in durations_hr])
    rows = []
    for rp, p24 in zip(return_periods, P24):
        for d in durations_hr:
            pdur = float(p24) * (d / DURATION_HR_BASE) ** exponent_n
            inten = pdur / d
            rows.append({
                "return_period_yr": int(rp),
                "duration_hr": float(d),
                "P_mm": round(pdur, 3),
                "I_mm_per_hr": round(inten, 3)
            })
    return pd.DataFrame(rows)


def pivot_idf(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df_long.pivot(index="duration_hr", columns="return_period_yr", values=value_col).reset_index()
    out.columns.name = None
    return out


def plot_depth_vs_rp(return_periods: List[int], P24: np.ndarray, title: str):
    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(return_periods, P24, marker="o")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("24-hour rainfall depth (mm)")
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_intensity_vs_duration(df_long: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(1, 1, 1)

    for rp in sorted(df_long["return_period_yr"].unique()):
        sub = df_long[df_long["return_period_yr"] == rp].sort_values("duration_hr")
        ax.plot(sub["duration_hr"], sub["I_mm_per_hr"], marker="o", label=f"{int(rp)} yr")

    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Intensity (mm/hr)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_xscale("log")
    ax.legend(title="Return period", ncols=2, fontsize=9)
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    buf.seek(0)
    return buf.getvalue()


def choose_map_style() -> str:
    """
    Mapbox styles look best but need MAPBOX_API_KEY.
    If missing, fall back to a token-free CARTO basemap.
    """
    if os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_ACCESS_TOKEN"):
        return "mapbox://styles/mapbox/dark-v11"
    return "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


def make_pydeck_map(gdf, lat0, lon0, lat_g, lon_g, view_mode, show_grid=True):
    gdf_wgs = gdf.to_crs(4326)
    geojson = gdf_wgs.__geo_interface__

    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_line_color=[255, 255, 255],
        get_fill_color=[60, 120, 180, 80],
        line_width_min_pixels=3,
    )

    pts = pd.DataFrame([
        {"name": "Site centroid", "lat": lat0, "lon": lon0, "color": [0, 200, 0]},
        {"name": "Nearest IMD grid", "lat": lat_g, "lon": lon_g, "color": [255, 80, 80]},
    ])

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="[lon, lat]",
        get_radius=650,
        get_fill_color="color",
        pickable=True,
    )

    grid_layers = []
    if show_grid:
        d = 0.125  # half-grid for 0.25°
        grid_poly = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "IMD grid cell"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_g - d, lat_g - d],
                        [lon_g + d, lat_g - d],
                        [lon_g + d, lat_g + d],
                        [lon_g - d, lat_g + d],
                        [lon_g - d, lat_g - d],
                    ]]
                }
            }]
        }

        grid_layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=grid_poly,
                stroked=True,
                filled=False,
                get_line_color=[255, 0, 0],
                line_width_min_pixels=2,
                pickable=True,
            )
        )

    if view_mode == "India":
        view_state = pdk.ViewState(latitude=22.5, longitude=79.0, zoom=3.6, transition_duration=1200)
    elif view_mode == "Grid":
        view_state = pdk.ViewState(latitude=lat_g, longitude=lon_g, zoom=7.5, transition_duration=1200)
    else:
        view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=13.5, transition_duration=1200)

    tooltip = {"text": "{name}\nLat: {lat}\nLon: {lon}"}

    return pdk.Deck(
        layers=[boundary_layer, point_layer] + grid_layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=choose_map_style(),
    )


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    use_demo = st.checkbox("Use demo boundary", value=False)
    demo_sites = list_demo_sites(SHAPE_ROOT)

    demo_choice = None
    if use_demo:
        if not demo_sites:
            st.warning("No demo shapefiles found under data/shapefiles/")
        else:
            demo_choice = st.selectbox("Choose a demo site", list(demo_sites.keys()))
            st.caption(f"Using: {demo_choice}")

    st.divider()
    st.header("IDF settings")

    rp = st.multiselect(
        "Return periods (years)",
        options=[2, 5, 10, 25, 50, 100, 200],
        default=DEFAULT_RETURN_PERIODS,
    )

    durations_hr = st.multiselect(
        "Durations (hours)",
        options=[0.5, 1, 2, 3, 6, 12, 24],
        default=DEFAULT_DURATIONS_HR,
    )

    exponent_n = st.slider(
        "Duration scaling exponent (n)",
        min_value=0.15,
        max_value=0.55,
        value=0.33,
        step=0.01,
        help="Shorter durations are derived using P(d)=P24*(d/24)^n.",
    )

    st.divider()
    st.header("Climate uplift (optional)")
    apply_uplift = st.checkbox("Apply uplift", value=False)
    uplift_pct = st.slider("Uplift (%)", 5, 30, 15, 1) if apply_uplift else 0


# -----------------------------
# STEP 1: SITE BOUNDARY
# -----------------------------
st.subheader("1) Site boundary")

if use_demo and demo_choice and demo_choice in demo_sites:
    try:
        gdf = gpd.read_file(demo_sites[demo_choice])
        st.info("Demo boundary loaded (bundled with this repo).")
        st.caption(f"Repo path: data/shapefiles/{demo_choice}")
    except Exception as e:
        st.error(f"Failed to load demo shapefile: {e}")
        st.stop()
else:
    up = st.file_uploader("Upload boundary (.zip shp / .gpkg / .geojson)", type=["zip", "gpkg", "geojson", "json"])
    if up is None:
        st.stop()
    try:
        gdf = read_vector_upload(up)
    except Exception as e:
        st.error(f"Boundary read failed: {e}")
        st.stop()

if gdf.empty:
    st.error("Boundary file has no features.")
    st.stop()

st.success(f"Loaded boundary | Features: {len(gdf)} | CRS: {pretty_crs(gdf.crs)}")


# -----------------------------
# STEP 2: RAINFALL DATA
# -----------------------------
st.subheader("2) Rainfall dataset")

try:
    ds = load_ds(NC_PATH)
except Exception as e:
    st.error(f"Could not open NetCDF at {NC_PATH}: {e}")
    st.stop()

time_name, lat_name, lon_name, var_name = detect_dims_and_var(ds)

tmin = str(pd.to_datetime(ds[time_name].values[0]).date())
tmax = str(pd.to_datetime(ds[time_name].values[-1]).date())

colA, colB = st.columns(2)
with colA:
    st.write("Variable:", f"**{var_name}**")
    st.write("Time range:", f"**{tmin} → {tmax}**")
with colB:
    st.write("Grid:", f"**{ds[lat_name].size} lat × {ds[lon_name].size} lon**")
    st.write("NetCDF:", f"`{NC_PATH.as_posix()}`")


# -----------------------------
# STEP 3: NEAREST GRID + MAP
# -----------------------------
st.subheader("3) Location check (India → Site → Grid)")

try:
    lat0, lon0 = centroid_latlon(gdf)
    lat_g, lon_g = nearest_grid(ds, lat_name, lon_name, lat0, lon0)
    dkm = distance_km(lat0, lon0, lat_g, lon_g)
except Exception as e:
    st.error(f"Centroid / nearest grid failed: {e}")
    st.stop()

st.write(f"Site centroid: **{lat0:.5f}, {lon0:.5f}**")
st.write(f"Nearest IMD grid: **{lat_g:.5f}, {lon_g:.5f}** (distance ≈ **{dkm:.2f} km**)")

col1, col2 = st.columns(2)
with col1:
    view_mode = st.radio("Map view", ["India", "Site", "Grid"], horizontal=True, index=1, key="map_view_mode")
with col2:
    show_grid = st.checkbox("Show IMD grid cell", value=True, key="show_imd_grid")

st.pydeck_chart(
    make_pydeck_map(gdf, lat0, lon0, lat_g, lon_g, view_mode, show_grid),
    use_container_width=True
)


# -----------------------------
# STEP 4: IDF (BASELINE)
# -----------------------------
st.subheader("4) IDF computation (baseline)")

if not rp:
    st.warning("Pick at least one return period.")
    st.stop()
if not durations_hr:
    st.warning("Pick at least one duration.")
    st.stop()

with st.spinner("Building annual maxima and fitting GEV..."):
    ams_df, fit = build_ams_and_fit_gev(ds, var_name, time_name, lat_name, lon_name, lat_g, lon_g)
    P24 = gev_quantiles_24h(fit, rp)

idf24 = pd.DataFrame({
    "return_period_yr": rp,
    "P24_mm": np.round(P24, 3),
    "I24_mm_per_hr": np.round(P24 / DURATION_HR_BASE, 3),
})

idf_long = idf_from_24h_depths(P24, rp, durations_hr, exponent_n)
idf_depth_matrix = pivot_idf(idf_long, "P_mm")
idf_int_matrix = pivot_idf(idf_long, "I_mm_per_hr")

st.success("Baseline IDF generated")

with st.expander("GEV fit parameters (for reference)"):
    st.write(f"shape(c): **{fit['shape_c']:.4f}**")
    st.write(f"loc: **{fit['loc']:.4f}**")
    st.write(f"scale: **{fit['scale']:.4f}**")

st.markdown("#### 24-hour results")
st.dataframe(idf24, use_container_width=True)

st.markdown("#### Multi-duration IDF (Depth matrix)")
st.dataframe(idf_depth_matrix, use_container_width=True)

st.markdown("#### Multi-duration IDF (Intensity matrix)")
st.dataframe(idf_int_matrix, use_container_width=True)

fig24 = plot_depth_vs_rp(rp, P24, "24-hour IDF (baseline)")
st.pyplot(fig24, clear_figure=True)
png24 = fig_to_png_bytes(fig24)

figI = plot_intensity_vs_duration(idf_long, "IDF curves (Intensity vs Duration, baseline)")
st.pyplot(figI, clear_figure=True)
pngI = fig_to_png_bytes(figI)

st.markdown("#### Downloads (baseline)")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("IDF (long) CSV", data=df_to_csv_bytes(idf_long), file_name="idf_long_baseline.csv", mime="text/csv")
with c2:
    st.download_button("Plot PNG (24h)", data=png24, file_name="idf_24h_baseline.png", mime="image/png")
with c3:
    st.download_button("Plot PNG (IDF)", data=pngI, file_name="idf_intensity_baseline.png", mime="image/png")

excel_baseline = to_excel_bytes({
    "idf_24h": idf24,
    "idf_long": idf_long,
    "depth_matrix": idf_depth_matrix,
    "intensity_matrix": idf_int_matrix,
    "ams": ams_df
})
st.download_button(
    "Download Excel (baseline)",
    data=excel_baseline,
    file_name="idf_baseline.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


# -----------------------------
# STEP 5: UPLIFT (OPTIONAL)
# -----------------------------
st.subheader("5) Climate uplift (optional)")

if apply_uplift:
    factor = 1.0 + uplift_pct / 100.0

    idf24_u = idf24.copy()
    idf24_u["P24_mm"] = np.round(idf24_u["P24_mm"] * factor, 3)
    idf24_u["I24_mm_per_hr"] = np.round(idf24_u["I24_mm_per_hr"] * factor, 3)

    idf_long_u = idf_long.copy()
    idf_long_u["P_mm"] = np.round(idf_long_u["P_mm"] * factor, 3)
    idf_long_u["I_mm_per_hr"] = np.round(idf_long_u["I_mm_per_hr"] * factor, 3)

    depth_u = pivot_idf(idf_long_u, "P_mm")
    int_u = pivot_idf(idf_long_u, "I_mm_per_hr")

    st.success(f"Applied uplift: +{uplift_pct}%")

    st.markdown("#### Uplifted 24-hour results")
    st.dataframe(idf24_u, use_container_width=True)

    st.markdown("#### Uplifted multi-duration matrices")
    st.dataframe(depth_u, use_container_width=True)
    st.dataframe(int_u, use_container_width=True)

    fig24u = plot_depth_vs_rp(rp, idf24_u["P24_mm"].values, f"24-hour IDF (+{uplift_pct}% uplift)")
    st.pyplot(fig24u, clear_figure=True)
    png24u = fig_to_png_bytes(fig24u)

    figIu = plot_intensity_vs_duration(idf_long_u, f"IDF curves (Intensity vs Duration, +{uplift_pct}% uplift)")
    st.pyplot(figIu, clear_figure=True)
    pngIu = fig_to_png_bytes(figIu)

    st.markdown("#### Downloads (uplifted)")
    u1, u2, u3 = st.columns(3)
    with u1:
        st.download_button("IDF (long) CSV", data=df_to_csv_bytes(idf_long_u), file_name=f"idf_long_uplift_{uplift_pct}pct.csv", mime="text/csv")
    with u2:
        st.download_button("Plot PNG (24h)", data=png24u, file_name=f"idf_24h_uplift_{uplift_pct}pct.png", mime="image/png")
    with u3:
        st.download_button("Plot PNG (IDF)", data=pngIu, file_name=f"idf_intensity_uplift_{uplift_pct}pct.png", mime="image/png")

    excel_uplift = to_excel_bytes({
        "idf_24h_uplift": idf24_u,
        "idf_long_uplift": idf_long_u,
        "depth_matrix_uplift": depth_u,
        "intensity_matrix_uplift": int_u,
        "ams": ams_df
    })
    st.download_button(
        "Download Excel (uplifted)",
        data=excel_uplift,
        file_name=f"idf_uplift_{uplift_pct}pct.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Uplift is off. Enable it from the sidebar if needed.")

st.caption("Note: This demo uses daily rainfall (24h). Shorter durations are derived via a transparent scaling rule with adjustable exponent n.")
