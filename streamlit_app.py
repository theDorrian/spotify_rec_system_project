# streamlit_app.py
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex
import joblib

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

# =====================  STYLES  =====================
st.markdown("""
<style>
.topbar{position:sticky; top:0; z-index:999; padding:10px 14px; margin:-10px -14px 16px -14px;
        background:rgba(9,12,18,.8); backdrop-filter: blur(6px); border-bottom:1px solid #1d2636;}
.searchbox input{border-radius:14px !important; height:38px !important; padding-left:14px !important;}
.avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#2b3446,#0f1726);
        border:1px solid #2b3446; display:inline-flex;align-items:center;justify-content:center;font-weight:700}
.avatar small{opacity:.8}

/* ---- ровные карточки ---- */
.card {
  background: rgba(255,255,255,0.02);
  border: 1px solid #2a3242;
  border-radius: 14px;
  padding: 12px;
  height: 330px;                    /* одинаковая высота */
  display: flex; flex-direction: column; gap: 8px;
}
.card .cover { width:100%; aspect-ratio: 1/1; border-radius:10px; overflow:hidden; background:#1b2335; }
.card .cover img { width:100%; height:100%; object-fit:cover; }
.card .title{
  font-weight:600; line-height:1.15;
  height:42px;                      /* место под 2 строки */
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}
.card .artist{
  opacity:.85; height:20px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.card .meta{ min-height:20px; }
.pop-pill{display:inline-block;border:1px solid #2a3242;border-radius:999px;padding:1px 8px;font-size:.75rem;opacity:.9}

/* кнопки везде одинаковые */
.stButton>button{
  width:100%; border:1px solid #2a3242; background:#0f1526; color:#e6e6e6; border-radius:10px
}
.stButton>button:hover{border-color:#324158;background:#121a30}

.section-title{font-weight:700;font-size:1.1rem;margin:4px 0 10px 0}
</style>
""", unsafe_allow_html=True)

# =====================  LOAD ARTIFACTS  =====================
ART_DIR = Path("artifacts")
TOP_QUANTILE = 0.70

@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: Path):
    with open(art_dir / "feature_cols.json") as f: _ = json.load(f)
    with open(art_dir / "meta.json") as f: meta = json.load(f)
    try: joblib.load(art_dir / "scaler.joblib")
    except Exception: pass
    try: joblib.load(art_dir / "svd_64.joblib")
    except Exception: pass
    df = None
    pq = art_dir / "id_map.parquet"
    if pq.exists():
        try: df = pd.read_parquet(pq)
        except Exception: df = None
    if df is None:
        df = pd.read_csv(art_dir / "id_map.csv")
    dim = int(meta["dim"])
    index = AnnoyIndex(dim, metric="angular"); index.load(str(art_dir / "annoy_index.ann"))
    return meta, df, index

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

# =====================  STATE  =====================
if "selected_row_id" not in st.session_state:
    st.session_state["selected_row_id"] = None
if "q" not in st.session_state:
    st.session_state["q"] = ""
if "rand_cut_ids" not in st.session_state:
    st.session_state["rand_cut_ids"] = None

# =====================  TOP BAR  =====================
st.markdown('<div class="topbar">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([0.28, 0.54, 0.18])
with c1:
    if st.button("🎧 Spotify Recommender", use_container_width=True):
        st.session_state["selected_row_id"] = None
        st.session_state["q"] = ""
        st.rerun()
with c2:
    q = st.text_input("Search", value=st.session_state["q"],
                      placeholder="Search tracks or artists…",
                      label_visibility="collapsed", key="searchbox")
    st.session_state["q"] = q
with c3:
    st.markdown('<div style="text-align:right;"><span class="avatar"><small>AK</small></span></div>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =====================  HELPERS  =====================
def open_track(row_id: int):
    st.session_state["selected_row_id"] = int(row_id)
    st.session_state["q"] = ""   # свернуть прошлые результаты
    st.rerun()

def render_card(row: pd.Series, row_id: int, key_prefix: str):
    img = row.get(img_col) or "https://placehold.co/600x600?text=Track"
    title = row.get(name_col, "Unknown")
    artist = row.get(artist_col, "")
    pop_txt = f'pop {int(row[pop_col])}' if pop_col and pd.notna(row.get(pop_col, None)) else ""

    # фиксированная по высоте карточка -> кнопка окажется на одном уровне у всех
    st.markdown(
        f"""
        <div class="card">
          <div class="cover"><img src="{img}" alt="cover"/></div>
          <div class="title">{title}</div>
          <div class="artist">{artist}</div>
          <div class="meta">{f'<span class="pop-pill">{pop_txt}</span>' if pop_txt else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("▶️ Open", key=f"{key_prefix}_open_{int(row_id)}"):
        open_track(int(row_id))

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

def top_global_ids(n: int = 10) -> list[int]:
    if pop_col:
        return only_top(IDMAP).sort_values(pop_col, ascending=False).head(n).index.tolist()
    return IDMAP.index.tolist()[:n]

def sample_random_cut_ids(k: int = 10) -> list[int]:
    tg = set(top_global_ids(10))
    pool_df = only_top(IDMAP)
    pool_ids = [rid for rid in pool_df.index.tolist() if rid not in tg]
    if not pool_ids:
        return []
    k = min(k, len(pool_ids))
    return random.sample(pool_ids, k)

# =====================  SEARCH (всегда активен) =====================
search_results = pd.DataFrame()
if st.session_state["q"].strip():
    ql = st.session_state["q"].strip().lower()
    mask = IDMAP[name_col].astype(str).str.lower().str.contains(ql, na=False) | \
           IDMAP[artist_col].astype(str).str.lower().str.contains(ql, na=False)
    search_results = only_top(IDMAP[mask].copy())
    if pop_col in search_results.columns:
        search_results = search_results.sort_values(pop_col, ascending=False)
    st.subheader(f"🔎 Search results for: **{st.session_state['q']}**")
    if search_results.empty:
        st.info("No matches.")
    else:
        render_grid(search_results, key_prefix="search", take=20, cols=5)
    st.markdown("---")

# =====================  MAIN / DETAILS  =====================
selected_id = st.session_state["selected_row_id"]

if pop_col:
    st.caption(f"Using popularity cutoff at {TOP_QUANTILE:.2f} quantile → {float(IDMAP[pop_col].quantile(TOP_QUANTILE)):.1f}")

if selected_id is None:
    st.subheader("🔥 Top 10 Global")
    top_global = only_top(IDMAP).sort_values(pop_col, ascending=False) if pop_col else IDMAP
    render_grid(top_global, key_prefix="global", take=10, cols=5)
else:
    st.subheader("Now playing")
    hero_card(IDMAP.loc[selected_id])

    st.subheader("Recommended for you")
    same_df = IDMAP[IDMAP[artist_col] == IDMAP.loc[selected_id, artist_col]].copy()
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
        else: render_grid(same_df, key_prefix=f"same_{selected_id}", take=10, cols=5)
    with c2:
        st.markdown('<div class="section-title">Similar by audio (top)</div>', unsafe_allow_html=True)
        if sim_df.empty: st.info("No similar top candidates.")
        else: render_grid(sim_df, key_prefix=f"sim_{selected_id}", take=10, cols=5)

st.markdown("---")

# =====================  RANDOM 10  =====================
left, right = st.columns([0.9, 0.1])
with left:
    st.subheader("⭐ Random 10")
with right:
    if st.button("🎲 Shuffle", use_container_width=True):
        st.session_state["rand_cut_ids"] = sample_random_cut_ids(10)
        st.rerun()

if not st.session_state["rand_cut_ids"]:
    st.session_state["rand_cut_ids"] = sample_random_cut_ids(10)
rand_ids = st.session_state["rand_cut_ids"]
rand_df = IDMAP.loc[rand_ids].copy() if rand_ids else pd.DataFrame()
render_grid(rand_df, key_prefix="randcut", take=10, cols=5)
