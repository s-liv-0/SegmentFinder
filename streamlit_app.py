# app.py — Strava Segment Finder by Direction & Distance
# -----------------------------------------------------
# What this app does
# • OAuth with Strava
# • Search segments with the /segments/explore endpoint inside a map-sized bounding box
# • Filter results by direction (bearing) and distance
# • Visualize on an interactive map (pydeck)
# • View a table with segment metadata + quick links to Strava
#
# Setup
# 1) In Streamlit Cloud or a local .streamlit/secrets.toml file, set:
#
#   [strava]
#   client_id = "YOUR_CLIENT_ID"
#   client_secret = "YOUR_CLIENT_SECRET"
#   redirect_uri = "http://localhost:8501"   # or your deployed Streamlit URL
#
# 2) In your Strava app settings, add the same Redirect URI.
# 3) `pip install streamlit requests pandas pydeck`
# 4) Run: streamlit run app.py

import math
import time
import json
import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

# ------------------------
# Helpers
# ------------------------

def wrap_deg(x: float) -> float:
    """Wrap degrees to [0,360)."""
    return (x + 360.0) % 360.0


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    brg = math.degrees(math.atan2(y, x))
    return wrap_deg(brg)


def circular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (deg) on a circle."""
    d = abs(wrap_deg(a) - wrap_deg(b))
    return d if d <= 180 else 360 - d


# Minimal polyline decoder (Google Encoded Polyline Algorithm Format)
# So we avoid an extra dependency.
# Returns list of (lat, lon)

def decode_polyline(polyline_str: str):
    coords, lat, lng, idx = [], 0, 0, 0
    length = len(polyline_str)
    while idx < length:
        result, shift = 0, 0
        while True:
            b = ord(polyline_str[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result, shift = 0, 0
        while True:
            b = ord(polyline_str[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng
        coords.append((lat / 1e5, lng / 1e5))
    return coords


def km_to_bbox(lat: float, lon: float, radius_km: float):
    """Return (sw_lat, sw_lon, ne_lat, ne_lon) for a rough circle radius using degrees.
    Approximation: 1 deg lat ~ 111 km; 1 deg lon ~ 111 km * cos(lat).
    """
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def strava_auth_headers(token: str):
    return {"Authorization": f"Bearer {token}"}


def exchange_token(code: str):
    cfg = st.secrets["strava"]
    payload = {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "code": code,
        "grant_type": "authorization_code",
    }
    r = requests.post("https://www.strava.com/oauth/token", data=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def refresh_token(refresh_token: str):
    cfg = st.secrets["strava"]
    payload = {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    r = requests.post("https://www.strava.com/oauth/token", data=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def ensure_token():
    # Refresh if needed (naive: every 5,000s)
    if "token_acquired_at" in st.session_state and time.time() - st.session_state.token_acquired_at > 5000:
        try:
            token_data = refresh_token(st.session_state.refresh_token)
            st.session_state.access_token = token_data["access_token"]
            st.session_state.refresh_token = token_data["refresh_token"]
            st.session_state.token_acquired_at = time.time()
        except Exception as e:
            st.warning(f"Token refresh failed: {e}")


def auth_ui():
    cfg = st.secrets.get("strava", {})
    if not cfg.get("client_id"):
        st.error("Missing Strava credentials in secrets.")
        return False

    st.sidebar.header("Connect Strava")
    params = st.query_params
    code = params.get("code", None)

    if not st.session_state.get("access_token"):
        auth_url = (
            "https://www.strava.com/oauth/authorize?client_id="
            + str(cfg["client_id"]) 
            + "&response_type=code&redirect_uri="
            + cfg["redirect_uri"]
            + "&scope=read,read_all,profile:read_all,activity:read_all"
            + "&approval_prompt=auto"
        )
        st.sidebar.markdown(f"[Authorize with Strava]({auth_url})")

        if code and st.sidebar.button("Complete Sign‑in"):
            try:
                data = exchange_token(code)
                st.session_state.access_token = data["access_token"]
                st.session_state.refresh_token = data["refresh_token"]
                st.session_state.token_acquired_at = time.time()
                st.sidebar.success("Connected to Strava.")
            except Exception as e:
                st.sidebar.error(f"OAuth error: {e}")
                return False
    else:
        st.sidebar.success("Connected to Strava.")

    return bool(st.session_state.get("access_token"))


def explore_segments(token: str, bounds, activity_type="riding", min_cat=None, max_cat=None):
    params = {
        "bounds": ",".join([str(x) for x in bounds]),
        "activity_type": activity_type,
    }
    if min_cat is not None:
        params["min_cat"] = int(min_cat)
    if max_cat is not None:
        params["max_cat"] = int(max_cat)

    r = requests.get(
        "https://www.strava.com/api/v3/segments/explore",
        headers=strava_auth_headers(token),
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("segments", [])


# ------------------------
# UI
# ------------------------

st.set_page_config(page_title="Strava Segment Finder", page_icon="🚴", layout="wide")
st.title("🚴 Strava Segment Finder — Direction & Distance")

connected = auth_ui()

with st.expander("Search Area", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.number_input("Center latitude", value=-27.4698, help="E.g., Brisbane CBD")
    with c2:
        lon = st.number_input("Center longitude", value=153.0251)
    with c3:
        radius_km = st.slider("Search radius (km)", min_value=1, max_value=50, value=8)

    sw_lat, sw_lon, ne_lat, ne_lon = km_to_bbox(lat, lon, radius_km)
    st.caption(f"Bounds (SW→NE): {sw_lat:.5f},{sw_lon:.5f} → {ne_lat:.5f},{ne_lon:.5f}")

with st.expander("Filters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        activity = st.selectbox("Activity type", ["riding", "running"])
    with c2:
        min_km = st.number_input("Min distance (km)", value=0.5, min_value=0.0, step=0.1)
    with c3:
        max_km = st.number_input("Max distance (km)", value=20.0, min_value=0.1, step=0.1)
    with c4:
        tol = st.slider("Direction tolerance (±°)", min_value=5, max_value=90, value=25)

    # Direction options (center bearings)
    direction_map = {
        "N": 0, "NE": 45, "E": 90, "SE": 135,
        "S": 180, "SW": 225, "W": 270, "NW": 315,
    }
    dcol1, dcol2 = st.columns([2,1])
    with dcol1:
        dir_label = st.selectbox("Preferred direction", list(direction_map.keys()))
    with dcol2:
        custom_deg = st.number_input("or custom (0–359°)", min_value=0, max_value=359, value=direction_map[dir_label])
    target_bearing = custom_deg

# ------------------------
# Search
# ------------------------

if st.button("🔎 Find Segments"):
    if not connected:
        st.warning("Please authorize with Strava first (left sidebar).")
        st.stop()

    ensure_token()

    with st.spinner("Querying Strava…"):
        try:
            bounds = (sw_lat, sw_lon, ne_lat, ne_lon)
            raw_segments = explore_segments(st.session_state.access_token, bounds, activity)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    if not raw_segments:
        st.info("No segments found. Try a larger radius or different area.")
        st.stop()

    # Transform & filter
    rows = []
    path_layers = []

    for s in raw_segments:
        sid = s.get("id")
        name = s.get("name")
        distance_m = s.get("distance", 0.0)
        distance_km = distance_m / 1000.0
        start = s.get("start_latlng") or [None, None]
        end = s.get("end_latlng") or [None, None]
        avg_grade = s.get("avg_grade")
        elev_diff = s.get("elev_difference")
        climb_cat = s.get("climb_category")
        points = s.get("points")  # encoded polyline for explorer results

        if None in start or None in end:
            continue

        brg = bearing_deg(start[0], start[1], end[0], end[1])
        if distance_km < min_km or distance_km > max_km:
            continue
        if circular_diff(brg, target_bearing) > tol:
            continue

        url = f"https://www.strava.com/segments/{sid}"

        rows.append({
            "id": sid,
            "name": name,
            "distance_km": round(distance_km, 2),
            "bearing_deg": round(brg, 1),
            "avg_grade_%": round(avg_grade, 1) if isinstance(avg_grade, (int, float)) else None,
            "elev_diff_m": round(elev_diff, 1) if isinstance(elev_diff, (int, float)) else None,
            "climb_cat": climb_cat,
            "strava": url,
            "start_lat": start[0],
            "start_lon": start[1],
        })

        # Build a path layer per segment if we have geometry
        if points:
            coords = decode_polyline(points)
            path_layers.append({
                "path": [{"lat": la, "lon": lo} for la, lo in coords],
                "name": name,
            })

    if not rows:
        st.info("No segments matched your direction + distance filters. Try widening tolerance or distance range.")
        st.stop()

    df = pd.DataFrame(rows)

    st.subheader("Results")
    st.dataframe(
        df[["name", "distance_km", "bearing_deg", "avg_grade_%", "elev_diff_m", "climb_cat", "strava"]],
        use_container_width=True,
    )

    # Map view with polylines (if any)
    st.subheader("Map")

    layers = []

    # Add paths
    if path_layers:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=path_layers,
                get_path="path",
                get_width=4,
                pickable=True,
            )
        )

    # Markers at starts
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[start_lon, start_lat]',
            get_radius=50,
            pickable=True,
        )
    )

    view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=12)

    r = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"})
    st.pydeck_chart(r, use_container_width=True)

    st.caption("Tip: Click a segment name in the table to open it on Strava.")


# Footer / rate limits note
with st.sidebar:
    st.markdown("---")
    st.caption("Note: Strava applies request rate limits per app. If searches fail, try again later.")
