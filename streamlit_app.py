# streamlit_app.py
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
import joblib
from rapidfuzz import process, fuzz
from sklearn.preprocessing import normalize as sk_normalize
from scipy import sparse
import plotly.express as px

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

# --------- Small style for cards ---------
st.markdown(
    """
    <style>
      .card {border-radius: 14px; padding: 12px; background: #11161f; border: 1px solid #1f2633;}
      .card img {border-radius: 10px; width: 100%; object-fit: cover; aspect-ratio:1/1;}
      .card .title {font-weight: 600; margin-top: 6px;}
      .card .artist {opacity: .85; font-size: .9rem;}
      .sec-title {margin: 8px 0 2px; font-weight: 600; font-size: 1.05rem;}
      .pill {display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #2a3242; font-size:.75rem; opacity:.9}
      .click {margin-top:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

ART_DIR = Path("artifacts")

# --------- Load artifacts ---------
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

    # id_map: parquet → csv fallback
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

# --------- Columns & sanity ---------
name_col   = meta.get("track_name_col")  or "track_name"
artist_col = meta.get("artist_name_col") or "artist_name"
genre_col  = "genre" if "genre" in id_map_raw.columns else None
pop_col    = "popularity" if "popularity" in id_map_raw.columns else None
year_col   = "year" if "year" in id_map_raw.columns else None
img_col    = "image_url" if "image_url" in id_map_raw.columns else None
prev_col   = "preview_url" if "preview_url" in id_map_raw.columns else None

# Guarantee row_id and index by row_id
if "row_id" not in id_map_raw.columns:
    id_map_raw.insert(0, "row_id", np.arange(len(id_map_raw)))
id_map_raw["row_id"] = id_map_raw["row_id"].astype(int)
IDMAP = id_map_raw.set_index("row_id").sort_index()

# Popularity threshold (top-only recommendations)
with st.sidebar:
    st.header("Filters")
    if pop_col:
        q_default = 0.70
        top_q = st.slider("Top popularity quantile (recommend only ≥)", 0.5, 0.95, q_default, 0.01)
        pop_threshold = float(IDMAP[pop_col].quantile(top_q))
        st.write(f"Cutoff: **{pop_threshold:.1f}** (quantile {top_q:.2f})")
    else:
        pop_threshold = None
        st.info("No popularity column detected — recommending without popularity filter.")

# Helper: good tracks filter
def only_top(df: pd.DataFrame) -> pd.DataFrame:
    if pop_threshold is None or pop_col not in df.columns:
        return df
    return df[df[pop_col] >= pop_threshold]

# Helper: draw a track card
def track_card(row: pd.Series):
    with st.container():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cols = st.columns([1, 2])
            with cols[0]:
                if img_col and pd.notna(row.get(img_col, None)):
                    st.image(row[img_col])
                else:
                    st.image("https://placehold.co/300x300?text=Track")
            with cols[1]:
                st.markdown(f'<div class="title">{row.get(name_col, "Unknown")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="artist">{row.get(artist_col, "")}</div>', unsafe_allow_html=True)
                pills = []
                if genre_col and pd.notna(row.get(genre_col, None)):
                    pills.append(f'<span class="pill">{row[genre_col]}</span>')
                if year_col and pd.notna(row.get(year_col, None)):
                    pills.append(f'<span class="pill">{int(row[year_col])}</span>')
                if pop_col and pd.notna(row.get(pop_col, None)):
                    pills.append(f'<span class="pill">pop {int(row[pop_col])}</span>')
                if pills:
                    st.markdown(" ".join(pills), unsafe_allow_html=True)
                if prev_col and pd.notna(row.get(prev_col, None)):
                    st.markdown(f'<a class="preview" target="_blank" href="{row[prev_col]}">▶️ Preview</a>', unsafe_allow_html=True)

                # click-to-open
                if st.button("Open", key=f"open_{row.name}", use_container_width=True):
                    st.session_state["selected_row_id"] = int(row.name)
                    st.query_params(track=int(row.name))
            st.markdown("</div>", unsafe_allow_html=True)

# Helper: get similar by Annoy (vector or item)
def similar_items(row_id: int, k: int = 20) -> pd.DataFrame:
    try:
        idxs, dists = index.get_nns_by_item(int(row_id), k+1, include_distances=True)
        idxs, dists = idxs[1:], dists[1:]
        recs = IDMAP.loc[idxs].copy()
        recs["dist"] = dists
        return recs
    except Exception:
        return pd.DataFrame()

# ------- Landing sections -------
st.title("🎧 Spotify Recommender — curated homepage")

# Restore selection from URL
qp = st.query_params
if "track" in qp:
    try:
        st.session_state["selected_row_id"] = int(qp.get("track"))
    except Exception:
        pass

# Section A: Top 10 popular tracks of different genres
if genre_col and pop_col:
    all_genres = [g for g in IDMAP[genre_col].dropna().unique().tolist() if str(g).strip()]
    random.shuffle(all_genres)
    picked = all_genres[:20]  # take more to filter by top
    cards = []
    for g in picked:
        gdf = IDMAP[IDMAP[genre_col] == g]
        gdf = only_top(gdf)
        if gdf.empty:
            continue
        top1 = gdf.sort_values(pop_col, ascending=False).head(1)
        cards.append(top1)

    grid_df = pd.concat(cards, axis=0).drop_duplicates().head(10) if cards else pd.DataFrame()
    st.subheader("🔥 Top 10 (different genres)")
    if not grid_df.empty:
        cols = st.columns(5)
        for i, (_, r) in enumerate(grid_df.iterrows()):
            with cols[i % 5]:
                track_card(r)
    else:
        st.info("Not enough data with genre/popularity to build Top 10 variety.")
else:
    st.info("Genre or popularity column missing — cannot build curated Top 10 by genre.")

# Section B: 4–5 random genres with their top lists
if genre_col and pop_col:
    st.markdown("---")
    st.subheader("🎛️ Top by random genres")
    sample_n = random.randint(4, 5)
    rg = random.sample(IDMAP[genre_col].dropna().unique().tolist(), k=min(sample_n, IDMAP[genre_col].nunique()))
    for g in rg:
        gdf = only_top(IDMAP[IDMAP[genre_col] == g])
        if gdf.empty:
            continue
        st.markdown(f"##### {g} — top tracks")
        gdft = gdf.sort_values(pop_col, ascending=False).head(10)
        cols = st.columns(5)
        for i, (_, r) in enumerate(gdft.iterrows()):
            with cols[i % 5]:
                track_card(r)

# -------- Detail view (clicked track) --------
selected_id = st.session_state.get("selected_row_id")
if selected_id is not None and selected_id in IDMAP.index:
    st.markdown("---")
    st.header("🎵 Now playing")
    seed = IDMAP.loc[selected_id]
    track_card(seed)

    # Recommendations:
    # 1) same-genre top (7)
    # 2) similar by Annoy (7)
    st.subheader("✨ Recommendations")

    col_left, col_right = st.columns(2)
    # same genre top (excluding seed), only top
    with col_left:
        st.markdown("**Same genre (top)**")
        same_df = pd.DataFrame()
        if genre_col and pd.notna(seed.get(genre_col, None)):
            same_df = IDMAP[IDMAP[genre_col] == seed[genre_col]].copy()
            same_df = same_df.loc[same_df.index != selected_id]
            same_df = only_top(same_df)
            if pop_col in same_df.columns:
                same_df = same_df.sort_values(pop_col, ascending=False)
            same_df = same_df.head(7)

        if same_df.empty:
            st.info("No same-genre top candidates.")
        else:
            cols = st.columns(7)
            for i, (_, r) in enumerate(same_df.iterrows()):
                with cols[i % 7]:
                    track_card(r)

    # similar by Annoy (filter to top)
    with col_right:
        st.markdown("**Similar by audio (top)**")
        sim_df = similar_items(selected_id, k=60)  # fetch more then filter
        if not sim_df.empty:
            sim_df = sim_df.loc[sim_df.index != selected_id]
            sim_df = only_top(sim_df)
            # de-duplicate same artist if you want diversity (optional):
            # sim_df = sim_df[sim_df[artist_col] != seed.get(artist_col)]
            sim_df = sim_df.head(7)
        if sim_df.empty:
            st.info("No similar top candidates.")
        else:
            cols = st.columns(7)
            for i, (_, r) in enumerate(sim_df.iterrows()):
                with cols[i % 7]:
                    track_card(r)

# -------- Explore tab / chart --------
st.markdown("---")
with st.expander("📈 Explore snapshot"):
    if artist_col in IDMAP.columns:
        top_art = IDMAP[artist_col].value_counts().head(20).reset_index()
        top_art.columns = ["artist", "count"]
        fig = px.bar(top_art, x="artist", y="count")
        st.plotly_chart(fig, use_container_width=True)
    if pop_col in IDMAP.columns:
        st.write("Popularity distribution (filtered top zone highlighted by quantile):")
        fig2 = px.histogram(IDMAP, x=pop_col, nbins=40)
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Tip: adjust the popularity quantile in the sidebar to control which tracks are considered 'top'.")
