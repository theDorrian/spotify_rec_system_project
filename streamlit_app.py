import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
import joblib
import plotly.express as px

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

# ---------- стили ----------
st.markdown("""
<style>
.topbar{position:sticky; top:0; z-index:999; padding:10px 14px; margin:-10px -14px 16px -14px;
        background:rgba(9,12,18,.8); backdrop-filter: blur(6px); border-bottom:1px solid #1d2636;}
.brand-btn>button{background:transparent;border:none;color:#fff;font-weight:800;font-size:1.2rem;cursor:pointer;}
.brand-btn>button:hover{opacity:.85}
.searchbox input{border-radius:14px !important; height:38px !important; padding-left:14px !important;}
.avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#2b3446,#0f1726);
        border:1px solid #2b3446; display:inline-flex;align-items:center;justify-content:center;font-weight:700}
.avatar small{opacity:.8}
.stImage img { border-radius:12px; }
.card-title{font-weight:600; margin:6px 0 2px 0; line-height:1.2; }
.card-artist{opacity:.85; margin-bottom:4px;}
.pop-pill{display:inline-block;border:1px solid #2a3242;border-radius:999px;padding:1px 8px;font-size:.75rem;opacity:.9}
.open-btn>button{border:1px solid #2a3242;background:#0f1526;color:#e6e6e6;border-radius:10px}
.open-btn>button:hover{border-color:#324158;background:#121a30}
.section-title{font-weight:700;font-size:1.1rem;margin:4px 0 10px 0}
</style>
""", unsafe_allow_html=True)

# ---------- артефакты ----------
ART_DIR = Path("artifacts")
TOP_QUANTILE = 0.70

@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: Path):
    with open(art_dir / "feature_cols.json") as f: cfg = json.load(f)
    with open(art_dir / "meta.json") as f: meta = json.load(f)
    try: joblib.load(art_dir / "scaler.joblib")
    except Exception: pass
    try: joblib.load(art_dir / "svd_64.joblib")
    except Exception: pass
    id_map = None
    pq = art_dir / "id_map.parquet"
    if pq.exists():
        try: id_map = pd.read_parquet(pq)
        except Exception: id_map = None
    if id_map is None:
        id_map = pd.read_csv(art_dir / "id_map.csv")
    dim = int(meta["dim"])
    index = AnnoyIndex(dim, metric="angular"); index.load(str(art_dir / "annoy_index.ann"))
    return meta, id_map, index

meta, id_map_raw, index = load_artifacts(ART_DIR)

name_col    = meta.get("track_name_col")  or "track_name"
artist_col  = meta.get("artist_name_col") or "artist_name"
trackid_col = meta.get("track_id_col")    or "track_id"
pop_col     = "popularity" if "popularity" in id_map_raw.columns else None
img_col     = "image_url"  if "image_url"  in id_map_raw.columns else None
prev_col    = "preview_url"if "preview_url"in id_map_raw.columns else None

if "row_id" not in id_map_raw.columns:
    id_map_raw.insert(0, "row_id", np.arange(len(id_map_raw)))
id_map_raw["row_id"] = id_map_raw["row_id"].astype(int)
IDMAP = id_map_raw.set_index("row_id").sort_index()

POP_CUTOFF = float(IDMAP[pop_col].quantile(TOP_QUANTILE)) if pop_col else None

def only_top(df: pd.DataFrame) -> pd.DataFrame:
    if POP_CUTOFF is None or pop_col not in df.columns: return df
    return df[df[pop_col] >= POP_CUTOFF]

def dedup(df: pd.DataFrame, take: int | None = None) -> pd.DataFrame:
    if trackid_col in df.columns:
        df = df.drop_duplicates(subset=[trackid_col])
    else:
        subs = [c for c in [name_col, artist_col] if c in df.columns]
        df = df.drop_duplicates(subset=subs) if subs else df
    return df.head(take) if take else df

# ---------- query params helpers (новый API) ----------
def get_track_from_qs() -> int | None:
    val = st.query_params.get("track")
    if isinstance(val, list):
        val = val[0] if val else None
    try:
        return int(val) if val is not None else None
    except Exception:
        return None

def set_track_in_qs(row_id: int | None):
    if row_id is None:
        qp = st.query_params
        if "track" in qp: del qp["track"]
    else:
        st.query_params["track"] = str(int(row_id))

# ---------- топбар ----------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
col_brand, col_search, col_avatar = st.columns([0.24, 0.56, 0.20])
with col_brand:
    if st.button("🎧 Spotify Recommender", key="brand_home", use_container_width=True):
        st.session_state["selected_row_id"] = None
        set_track_in_qs(None)
        st.rerun()
with col_search:
    q = st.text_input("Search", key="q", placeholder="Search tracks or artists…", label_visibility="collapsed")
with col_avatar:
    st.markdown('<div style="text-align:right;"><span class="avatar"><small>AK</small></span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- клики / рендер ----------
def open_track(row_id: int):
    rid = int(row_id)
    st.session_state["selected_row_id"] = rid
    set_track_in_qs(rid)
    st.rerun()

def render_card(row: pd.Series, row_id: int, key_prefix: str):
    img = row.get(img_col) or "https://placehold.co/300x300?text=Track"
    st.image(img, width=190)
    st.markdown(f'<div class="card-title">{row.get(name_col,"Unknown")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-artist">{row.get(artist_col,"")}</div>', unsafe_allow_html=True)
    if pop_col and pd.notna(row.get(pop_col, None)):
        st.markdown(f'<span class="pop-pill">pop {int(row[pop_col])}</span>', unsafe_allow_html=True)
    st.markdown('<div class="open-btn">', unsafe_allow_html=True)
    if st.button("▶️ Open", key=f"{key_prefix}_open_{int(row_id)}"):
        open_track(int(row_id))
    st.markdown('</div>', unsafe_allow_html=True)

def render_grid(df: pd.DataFrame, key_prefix: str, take: int = 10, cols: int = 5):
    df = dedup(df, take=take)
    items = list(df.iterrows())
    for i in range(0, len(items), cols):
        chunk = items[i:i+cols]
        columns = st.columns(len(chunk), gap="large")
        for (rid, r), c in zip(chunk, columns):
            with c:
                render_card(r, int(rid), key_prefix)

def hero_card(row: pd.Series):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        img = row.get(img_col) or "https://placehold.co/600x600?text=Album"
        st.image(img, width=240)
    with c2:
        st.subheader(row.get(name_col, "Unknown"))
        st.caption(str(row.get(artist_col, "")))
        if pop_col and pd.notna(row.get(pop_col, None)):
            st.markdown(f'**`pop {int(row[pop_col])}`**')
        if prev_col and pd.notna(row.get(prev_col, None)):
            st.audio(row[prev_col])

def similar_items(row_id: int, k: int = 80) -> pd.DataFrame:
    try:
        idxs, dists = index.get_nns_by_item(int(row_id), k+1, include_distances=True)
        idxs, dists = idxs[1:], dists[1:]
        recs = IDMAP.loc[idxs].copy()
        recs["dist"] = dists
        return recs
    except Exception:
        return pd.DataFrame()

# ---------- состояние из query params ----------
if "selected_row_id" not in st.session_state:
    st.session_state["selected_row_id"] = None
track_qs = get_track_from_qs()
if track_qs is not None:
    st.session_state["selected_row_id"] = track_qs

selected_id = st.session_state.get("selected_row_id")

# ---------- поиск ----------
search_results = pd.DataFrame()
if q and isinstance(q, str) and len(q.strip()) >= 2:
    ql = q.strip().lower()
    mask = IDMAP[name_col].astype(str).str.lower().str.contains(ql, na=False) | \
           IDMAP[artist_col].astype(str).str.lower().str.contains(ql, na=False)
    search_results = only_top(IDMAP[mask].copy())
    if pop_col in search_results.columns:
        search_results = search_results.sort_values(pop_col, ascending=False)
    st.subheader(f"🔎 Search results for: **{q}**")
    if search_results.empty:
        st.info("No matches.")
    else:
        render_grid(search_results, key_prefix="search", take=20, cols=5)
    st.markdown("---")

# ---------- главная / детали ----------
if selected_id is None and search_results.empty:
    if pop_col:
        st.caption(f"Using popularity cutoff at {TOP_QUANTILE:.2f} quantile → {float(IDMAP[pop_col].quantile(TOP_QUANTILE)):.1f}")
    st.subheader("🔥 Top 10 Global")
    top_global = only_top(IDMAP).sort_values(pop_col, ascending=False) if pop_col else IDMAP
    render_grid(top_global, key_prefix="global", take=10, cols=5)

    st.markdown("---")
    st.subheader("🎛️ Top by random artists")
    artists = IDMAP[artist_col].dropna().unique().tolist()
    random.shuffle(artists)
    for ai, a in enumerate(artists[:min(4, len(artists))]):
        adf = only_top(IDMAP[IDMAP[artist_col] == a])
        if adf.empty:
            continue
        st.markdown(f"**{a} — top tracks**")
        adf = adf.sort_values(pop_col, ascending=False) if pop_col else adf
        render_grid(adf, key_prefix=f"artist_{ai}", take=10, cols=5)

elif selected_id is not None:
    seed = IDMAP.loc[selected_id]
    st.subheader("Now playing")
    hero_card(seed)

    st.subheader("Recommended for you")
    same_df = IDMAP[IDMAP[artist_col] == seed.get(artist_col)].copy()
    same_df = same_df.loc[same_df.index != selected_id]
    same_df = only_top(same_df)
    same_df = same_df.sort_values(pop_col, ascending=False) if pop_col else same_df

    sim_df = similar_items(selected_id)
    if not sim_df.empty:
        sim_df = sim_df.loc[sim_df.index != selected_id]
        sim_df = only_top(sim_df)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-title">Same artist (top)</div>', unsafe_allow_html=True)
        if same_df.empty: st.info("No same-artist candidates.")
        else: render_grid(same_df, key_prefix=f"same_{selected_id}", take=7, cols=3)
    with c2:
        st.markdown('<div class="section-title">Similar by audio (top)</div>', unsafe_allow_html=True)
        if sim_df.empty: st.info("No similar top candidates.")
        else: render_grid(sim_df, key_prefix=f"sim_{selected_id}", take=7, cols=3)

st.markdown("---")
with st.expander("📈 Explore snapshot"):
    if artist_col in IDMAP.columns:
        top_art = IDMAP[artist_col].value_counts().head(20).reset_index()
        top_art.columns = ["artist","count"]
        fig = px.bar(top_art, x="artist", y="count")
        st.plotly_chart(fig, use_container_width=True)
    if pop_col in IDMAP.columns:
        fig2 = px.histogram(IDMAP, x=pop_col, nbins=40)
        st.plotly_chart(fig2, use_container_width=True)
