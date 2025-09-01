# streamlit_app.py — one-click select, Home button, no warnings
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
  .rec-grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:16px}
  .card{border-radius:14px;background:#11161f;border:1px solid #1f2633;padding:10px;transition:transform .12s ease, border-color .12s ease}
  .card:hover{transform:translateY(-3px);border-color:#2b3647}
  .name{font-weight:600;margin-top:8px;line-height:1.2;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .artist-s{opacity:.85;font-size:.9rem;display:-webkit-box;-webkit-line-clamp:1;-webkit-box-orient:vertical;overflow:hidden}
</style>
""", unsafe_allow_html=True)

ART_DIR = Path("artifacts")
TOP_QUANTILE = 0.70  # топ по популярности

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: Path):
    with open(art_dir / "feature_cols.json") as f:
        cfg = json.load(f)
    with open(art_dir / "meta.json") as f:
        meta = json.load(f)

    # предобученные артефакты (в UI не используем, но держим для совместимости)
    try: joblib.load(art_dir / "scaler.joblib")
    except Exception: pass
    try: joblib.load(art_dir / "svd_64.joblib")
    except Exception: pass

    # id_map parquet -> csv fallback
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

# ---------- Helpers ----------
def set_selected(rid: int | None):
    """Set or clear selected track; keep URL in sync if возможно."""
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
            st.query_params["track"] = str(int(rid))      # новый API (>=1.32)
        except Exception:
            try: st.experimental_set_query_params(track=int(rid))  # старый API
            except Exception: pass

def hero_card(row: pd.Series):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.image(
            row.get(img_col, "https://placehold.co/600x600?text=Album"),
            use_container_width=False
        )
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

def rec_grid(df: pd.DataFrame, key_prefix: str):
    """Рендер компактной сетки рекомендаций (7 карточек) с одним кликом."""
    df = dedup(df, take=7)
    st.markdown('<div class="rec-grid">', unsafe_allow_html=True)
    for rid, r in df.iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(r.get(img_col, "https://placehold.co/300x300?text=Track"), use_container_width=True)
            st.markdown(f'<div class="name">{r.get(name_col,"Unknown")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="artist-s">{r.get(artist_col,"")}</div>', unsafe_allow_html=True)
            if pop_col and pd.notna(r.get(pop_col, None)):
                st.markdown(f'<span class="pill">pop {int(r[pop_col])}</span>', unsafe_allow_html=True)
            # один клик — без HTML-кнопок
            if st.button("▶️ Open", key=f"{key_prefix}_open_{rid}", use_container_width=True):
                set_selected(int(rid))
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def similar_items(row_id: int, k: int = 80) -> pd.DataFrame:
    try:
        idxs, dists = index.get_nns_by_item(int(row_id), k+1, include_distances=True)
        idxs, dists = idxs[1:], dists[1:]
        recs = IDMAP.loc[idxs].copy()
        recs["dist"] = dists
        return recs
    except Exception:
        return pd.DataFrame()

# ---------- Header with Home ----------
col_home, col_title = st.columns([0.1, 0.9])
with col_home:
    if st.button("🏠 Home", use_container_width=True):
        set_selected(None)
with col_title:
    st.title("Spotify Recommender")
if pop_col:
    st.caption(f"Using popularity cutoff at {TOP_QUANTILE:.2f} quantile → {POP_CUTOFF:.1f}")

# restore selection from URL (new/old APIs)
sel_from_url = None
try:
    val = st.query_params.get("track")
    if isinstance(val, (list, tuple)): val = val[0] if val else None
    if val is not None: sel_from_url = int(val)
except Exception:
    try:
        params = st.experimental_get_query_params()
        if "track" in params and params["track"]: sel_from_url = int(params["track"][0])
    except Exception:
        pass
if sel_from_url is not None:
    st.session_state["selected_row_id"] = sel_from_url

# ---------- Either Landing or Detail ----------
selected_id = st.session_state.get("selected_row_id")

if selected_id is None:
    # Landing: Top-10 + random artists
    st.subheader("🔥 Top 10 Global")
    top_global = only_top(IDMAP).sort_values(pop_col, ascending=False) if pop_col else IDMAP
    top_global = dedup(top_global, take=10)
    cols = st.columns(5)
    for i, (rid, r) in enumerate(top_global.iterrows()):
        with cols[i % 5]:
            st.image(r.get(img_col, "https://placehold.co/300x300?text=Track"), use_container_width=True)
            st.write(f"**{r.get(name_col,'Unknown')}**")
            st.caption(r.get(artist_col, ""))
            if st.button("▶️ Open", key=f"landing_open_{rid}", use_container_width=True):
                set_selected(int(rid))

    st.markdown("---")
    st.subheader("🎛️ Top by random artists")
    artists = IDMAP[artist_col].dropna().unique().tolist()
    random.shuffle(artists)
    for ai, a in enumerate(artists[:min(5, len(artists))]):
        adf = only_top(IDMAP[IDMAP[artist_col] == a])
        if adf.empty: 
            continue
        st.markdown(f"**{a} — top tracks**")
        adf = dedup(adf.sort_values(pop_col, ascending=False) if pop_col else adf, take=10)
        cols = st.columns(5)
        for i, (rid, r) in enumerate(adf.iterrows()):
            with cols[i % 5]:
                st.image(r.get(img_col, "https://placehold.co/300x300?text=Track"), use_container_width=True)
                st.write(f"**{r.get(name_col,'Unknown')}**")
                st.caption(r.get(artist_col, ""))
                if st.button("▶️ Open", key=f"artist_{ai}_open_{rid}", use_container_width=True):
                    set_selected(int(rid))

else:
    # Detail-only page
    seed = IDMAP.loc[selected_id]
    st.header("Now playing")
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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Same artist (top)**")
        if same_df.empty: st.info("No same-artist candidates.")
        else: rec_grid(same_df, key_prefix=f"same_{selected_id}")
    with col2:
        st.markdown("**Similar by audio (top)**")
        if sim_df.empty: st.info("No similar top candidates.")
        else: rec_grid(sim_df, key_prefix=f"sim_{selected_id}")

# ---------- Explore snapshot ----------
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
