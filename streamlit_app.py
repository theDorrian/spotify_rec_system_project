# streamlit_app.py — no genres, unique keys, dedup grids
import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
import joblib
from sklearn.preprocessing import normalize as sk_normalize
from scipy import sparse
import plotly.express as px

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

# ----- small styles -----
st.markdown("""
<style>
  .card {border-radius:14px; padding:12px; background:#11161f; border:1px solid #1f2633;}
  .card img {border-radius:10px; width:100%; object-fit:cover; aspect-ratio:1/1;}
  .title{font-weight:600; margin-top:6px;}
  .artist{opacity:.85; font-size:.9rem;}
  .pill{display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #2a3242; font-size:.75rem; opacity:.9}
</style>
""", unsafe_allow_html=True)

ART_DIR = Path("artifacts")
TOP_QUANTILE = 0.70  # считаем «топом» всё, что >= этого перцентиля popularity

# ----- load artifacts -----
@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: Path):
    with open(art_dir / "feature_cols.json") as f:
        cfg = json.load(f)
    with open(art_dir / "meta.json") as f:
        meta = json.load(f)

    scaler = joblib.load(art_dir / "scaler.joblib")
    mlb = None
    try:
        mlb = joblib.load(art_dir / "mlb_genres.joblib")
    except Exception:
        pass

    svd = None
    if (art_dir / "svd_64.joblib").exists():
        try:
            svd = joblib.load(art_dir / "svd_64.joblib")
        except Exception:
            svd = None

    # id_map parquet -> csv fallback
    id_map = None
    if (art_dir / "id_map.parquet").exists():
        try:
            id_map = pd.read_parquet(art_dir / "id_map.parquet")
        except Exception:
            id_map = None
    if id_map is None:
        id_map = pd.read_csv(art_dir / "id_map.csv")

    dim = int(meta["dim"])
    index = AnnoyIndex(dim, metric="angular")
    index.load(str(art_dir / "annoy_index.ann"))

    return cfg, meta, scaler, mlb, svd, id_map, index

cfg, meta, scaler, mlb, svd, id_map_raw, index = load_artifacts(ART_DIR)

# ----- columns -----
name_col    = meta.get("track_name_col")  or "track_name"
artist_col  = meta.get("artist_name_col") or "artist_name"
trackid_col = meta.get("track_id_col")    or "track_id"
pop_col     = "popularity" if "popularity" in id_map_raw.columns else None
img_col     = "image_url"  if "image_url"  in id_map_raw.columns else None
prev_col    = "preview_url"if "preview_url"in id_map_raw.columns else None

# ensure row_id
if "row_id" not in id_map_raw.columns:
    id_map_raw.insert(0, "row_id", np.arange(len(id_map_raw)))
id_map_raw["row_id"] = id_map_raw["row_id"].astype(int)
IDMAP = id_map_raw.set_index("row_id").sort_index()

# popularity cutoff
if pop_col:
    POP_CUTOFF = float(IDMAP[pop_col].quantile(TOP_QUANTILE))
else:
    POP_CUTOFF = None

def only_top(df: pd.DataFrame) -> pd.DataFrame:
    if POP_CUTOFF is None or pop_col not in df.columns:
        return df
    return df[df[pop_col] >= POP_CUTOFF]

def dedup(df: pd.DataFrame, take: int | None = None) -> pd.DataFrame:
    """убираем дубли по track_id, иначе по (name, artist)."""
    if trackid_col in df.columns:
        df = df.drop_duplicates(subset=[trackid_col])
    else:
        subs = [c for c in [name_col, artist_col] if c in df.columns]
        if subs:
            df = df.drop_duplicates(subset=subs)
    if take is not None:
        df = df.head(take)
    return df

# ----- helpers -----
def track_card(row: pd.Series, key_prefix: str):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c = st.columns([1,2])
    with c[0]:
        if img_col and pd.notna(row.get(img_col, None)):
            st.image(row[img_col])
        else:
            st.image("https://placehold.co/300x300?text=Track")
    with c[1]:
        st.markdown(f'<div class="title">{row.get(name_col, "Unknown")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="artist">{row.get(artist_col, "")}</div>', unsafe_allow_html=True)
        pills = []
        if pop_col and pd.notna(row.get(pop_col, None)):
            pills.append(f'<span class="pill">pop {int(row[pop_col])}</span>')
        if pills:
            st.markdown(" ".join(pills), unsafe_allow_html=True)

        if prev_col and pd.notna(row.get(prev_col, None)):
            st.markdown(f'<a target="_blank" href="{row[prev_col]}">▶️ Preview</a>', unsafe_allow_html=True)

        # уникальный ключ для каждой секции
        if st.button("Open", key=f"{key_prefix}_open_{row.name}", use_container_width=True):
            rid = int(row.name)
            st.session_state["selected_row_id"] = rid
            # попытка обновить query param (новый/старый API)
            try:
                st.query_params["track"] = str(rid)   # >=1.32
            except Exception:
                try:
                    st.experimental_set_query_params(track=rid)  # старый
                except Exception:
                    pass
    st.markdown('</div>', unsafe_allow_html=True)

def similar_items(row_id: int, k: int = 20) -> pd.DataFrame:
    try:
        idxs, dists = index.get_nns_by_item(int(row_id), k+1, include_distances=True)
        idxs, dists = idxs[1:], dists[1:]
        recs = IDMAP.loc[idxs].copy()
        recs["dist"] = dists
        return recs
    except Exception:
        return pd.DataFrame()

# ----- header -----
st.title("🎧 Spotify Recommender — curated homepage (no genres)")
if pop_col:
    st.caption(f"Using popularity cutoff at {TOP_QUANTILE:.2f} quantile → {POP_CUTOFF:.1f}")

# restore selection from URL (new/old APIs)
selected_from_url = None
try:
    qp = st.query_params         # new proxy (dict-like)
    val = qp.get("track")
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if val is not None:
        selected_from_url = int(val)
except Exception:
    try:
        params = st.experimental_get_query_params()  # old API
        if "track" in params and len(params["track"]) > 0:
            selected_from_url = int(params["track"][0])
    except Exception:
        pass
if selected_from_url is not None:
    st.session_state["selected_row_id"] = selected_from_url

# ----- Section A: Top 10 Global (dedup) -----
st.subheader("🔥 Top 10 Global")
if pop_col:
    top_global = only_top(IDMAP).sort_values(pop_col, ascending=False)
else:
    top_global = IDMAP.sample(min(10, len(IDMAP)), random_state=42)
top_global = dedup(top_global, take=10)

cols = st.columns(5)
for i, (rid, r) in enumerate(top_global.iterrows()):
    with cols[i % 5]:
        track_card(r, key_prefix=f"top10_{i}")

# ----- Section B: 4–5 random artists with their top tracks -----
st.markdown("---")
st.subheader("🎛️ Top by random artists")
artists = IDMAP[artist_col].dropna().unique().tolist()
random.shuffle(artists)
for ai, a in enumerate(artists[:min(5, len(artists))]):
    adf = IDMAP[IDMAP[artist_col] == a]
    adf = only_top(adf)
    if adf.empty:
        continue
    st.markdown(f"##### {a} — top tracks")
    adf = adf.sort_values(pop_col, ascending=False) if pop_col else adf
    adf = dedup(adf, take=10)
    cols = st.columns(5)
    for i, (rid, r) in enumerate(adf.iterrows()):
        with cols[i % 5]:
            track_card(r, key_prefix=f"artist_{ai}_{i}")

# ----- Detail view -----
selected_id = st.session_state.get("selected_row_id")
if selected_id is not None and selected_id in IDMAP.index:
    st.markdown("---")
    st.header("🎵 Now playing")
    seed = IDMAP.loc[selected_id]
    track_card(seed, key_prefix="nowplaying")

    st.subheader("✨ Recommendations")
    left, right = st.columns(2)

    # 7 same-artist (top)
    with left:
        st.markdown("**Same artist (top)**")
        same_df = IDMAP[IDMAP[artist_col] == seed.get(artist_col)].copy()
        same_df = same_df.loc[same_df.index != selected_id]
        same_df = only_top(same_df)
        same_df = same_df.sort_values(pop_col, ascending=False) if pop_col else same_df
        same_df = dedup(same_df, take=7)
        if same_df.empty:
            st.info("No same-artist candidates.")
        else:
            cols = st.columns(7)
            for i, (rid, r) in enumerate(same_df.iterrows()):
                with cols[i % 7]:
                    track_card(r, key_prefix=f"same_{selected_id}_{i}")

    # 7 similar by Annoy (top)
    with right:
        st.markdown("**Similar by audio (top)**")
        sim_df = similar_items(selected_id, k=80)
        if not sim_df.empty:
            sim_df = sim_df.loc[sim_df.index != selected_id]
            sim_df = only_top(sim_df)
            sim_df = dedup(sim_df, take=7)
        if sim_df.empty:
            st.info("No similar top candidates.")
        else:
            cols = st.columns(7)
            for i, (rid, r) in enumerate(sim_df.iterrows()):
                with cols[i % 7]:
                    track_card(r, key_prefix=f"sim_{selected_id}_{i}")

# ----- Explore snapshot -----
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
