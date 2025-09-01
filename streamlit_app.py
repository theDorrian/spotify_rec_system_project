# streamlit_app.py — fixed hero cover (240x240), fixed-grid, one-click select, Home
import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
import joblib
import plotly.express as px

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

# ---------- Styles ----------
st.markdown("""
<style>
  .pill{display:inline-block;padding:2px 10px;border-radius:999px;border:1px solid #2a3242;
        font-size:.75rem;opacity:.9;margin-right:6px}
  .hero{display:flex;gap:22px;align-items:center;padding:16px;border:1px solid #1f2633;
        border-radius:16px;background:#101620}
  .title{font-weight:700;font-size:1.35rem;margin:2px 0}
  .artist{opacity:.9}

  /* обложка в Now Playing — жёстко 240x240 */
  .hero-cover{
    width:240px !important; height:240px !important;
    max-width:240px !important; max-height:240px !important;
    border-radius:14px; object-fit:cover; display:block;
  }

  /* Ровная фикс-сетка: карточка 210px, без растяжения */
  .grid{
    display:grid; gap:16px;
    grid-template-columns: repeat(auto-fill, 210px);
    justify-content: start; align-content: start;
  }
  .card{
    width:210px; border-radius:14px; background:#11161f; border:1px solid #1f2633; padding:10px;
    transition:transform .12s ease, border-color .12s ease;
  }
  .card:hover{transform:translateY(-3px);border-color:#2b3647}
  .name{
    font-weight:600;margin-top:8px;line-height:1.2;
    display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden
  }
  .artist-s{
    opacity:.85;font-size:.9rem;
    display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden
  }
</style>
""", unsafe_allow_html=True)

ART_DIR = Path("artifacts")
TOP_QUANTILE = 0.70  # «топ» по популярности

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: Path):
    with open(art_dir / "feature_cols.json") as f:
        cfg = json.load(f)
    with open(art_dir / "meta.json") as f:
        meta = json.load(f)

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
    index = AnnoyIndex(dim, metric="angular")
    index.load(str(art_dir / "annoy_index.ann"))
    return meta, id_map, index

meta, id_map_raw, index = load_artifacts(ART_DIR)

# ---------- Columns ----------
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

# ---------- State helpers ----------
def set_selected(rid: int | None):
    if rid is None:
        st.session_state.pop("selected_row_id", None)
        try:
            qp = st.query_params
            if "track" in qp: del qp["track"]
        except Exception:
            try: st.experimental_set_query_params()
            except Exception: pass
    else:
        st.session_state["selected_row_id"] = int(rid)
        try:
            st.query_params["track"] = str(int(rid))
        except Exception:
            try: st.experimental_set_query_params(track=int(rid))
            except Exception: pass
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# ---------- UI pieces ----------
def hero_card(row: pd.Series):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        img = row.get(img_col, None) or "https://placehold.co/600x600?text=Album"
        # HTML-картинка с классом hero-cover (жёстко 240x240)
        st.markdown(f'<img src="{img}" class="hero-cover">', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="hero">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div>
              <div class="title">{row.get(name_col, "Unknown")}</div>
              <div class="artist">{row.get(artist_col, "")}</div>
              <div style="margin-top:10px">
                {f'<span class="pill">pop {int(row[pop_col])}</span>' if pop_col and pd.notna(row.get(pop_col,None)) else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if prev_col and pd.notna(row.get(prev_col, None)):
            st.audio(row[prev_col])

def render_grid(df: pd.DataFrame, key_prefix: str, take: int = 10):
    df = dedup(df, take=take)
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    for rid, r in df.iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(r.get(img_col, "https://placehold.co/300x300?text=Track"),
                     use_container_width=True)
            st.markdown(f'<div class="name">{r.get(name_col,"Unknown")}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="artist-s">{r.get(artist_col,"")}</div>',
                        unsafe_allow_html=True)
            if pop_col and pd.notna(r.get(pop_col, None)):
                st.markdown(f'<span class="pill">pop {int(r[pop_col])}</span>',
                            unsafe_allow_html=True)
            st.button("▶️ Open",
                      key=f"{key_prefix}_open_{rid}",
                      use_container_width=True,
                      on_click=set_selected, args=(int(rid),))
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def similar_items(row_id: int, k: int = 80) -> pd.DataFrame:
    try:
        idxs, dists = index.get_nns_by_item(int(row_id), k+1, include_distances=True)
        idxs,
