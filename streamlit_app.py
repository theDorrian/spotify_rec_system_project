import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import streamlit as st
from annoy import AnnoyIndex

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

DEFAULT_COVER = "https://i.pinimg.com/736x/ea/1f/64/ea1f64668a0af149a3277db9e9e54824.jpg"

st.markdown("""
<style>
.topbar{position:sticky; top:0; z-index:999; padding:10px 14px; margin:-10px -14px 16px -14px;
        background:rgba(9,12,18,.8); backdrop-filter: blur(6px); border-bottom:1px solid #1d2636;}
.searchbox input{border-radius:14px !important; height:38px !important; padding-left:14px !important;}
.avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#2b3446,#0f1726);
        border:1px solid #2b3446; display:inline-flex;align-items:center;justify-content:center;font-weight:700}
.avatar small{opacity:.8}
.card{background: rgba(255,255,255,0.02); border: 1px solid #2a3242; border-radius: 14px; padding: 10px;
      display: flex; flex-direction: column; align-items: center; gap: 8px; width: 180px;}
.card .cover{ width:160px; height:160px; border-radius:10px; overflow:hidden; background:#1b2335;
              display:flex; align-items:center; justify-content:center; }
.card .cover img{ width:100%; height:100%; object-fit:cover; }
.card .title{ font-weight:600; font-size:.85rem; text-align:center; line-height:1.2;
              display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; min-height:32px; }
.card .artist{ opacity:.85; font-size:.8rem; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card .meta{ font-size:.75rem; text-align:center; }
.pop-pill{ display:inline-block; border:1px solid #2a3242; border-radius:999px; padding:1px 8px; font-size:.75rem; opacity:.9 }
.section-title{ font-weight:700; font-size:1.15rem; margin:6px 0 10px 0 }
.stButton>button{ width:100%; border:1px solid #2a3242; background:#0f1526; color:#e6e6e6; border-radius:10px }
.stButton>button:hover{ border-color:#324158; background:#121a30 }
</style>
""", unsafe_allow_html=True)

ART_DIR = Path("artifacts")

@st.cache_resource(show_spinner=False)
def load_artifacts(art_dir: Path):
    with open(art_dir / "meta.json") as f:
        meta = json.load(f)
    df = None
    pq = art_dir / "id_map.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except Exception:
            df = None
    if df is None:
        df = pd.read_csv(art_dir / "id_map.csv")
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df)))
    df["row_id"] = df["row_id"].astype(int)
    idmap = df.set_index("row_id").sort_index()
    dim = int(meta["dim"])
    idx = AnnoyIndex(dim, "angular")
    idx.load(str(art_dir / "annoy_index.ann"))
    return meta, idmap, idx

meta, IDMAP, ANNOY = load_artifacts(ART_DIR)

NAME  = meta.get("track_name_col")  or "track_name"
ART   = meta.get("artist_name_col") or "artist_name"
TID   = meta.get("track_id_col")    or "track_id"
POP   = "popularity" if "popularity" in IDMAP.columns else None
IMG   = "image_url"  if "image_url"  in IDMAP.columns else None
PREV  = "preview_url"if "preview_url"in IDMAP.columns else None

CUT_Q = 0.70
CUT   = float(IDMAP[POP].quantile(CUT_Q)) if POP else None

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if TID in df.columns:
        return df.drop_duplicates(subset=[TID])
    return df.drop_duplicates(subset=[NAME, ART])

def only_top(df: pd.DataFrame) -> pd.DataFrame:
    if POP is None: return df
    return df[df[POP] >= CUT]

def top10_global() -> pd.DataFrame:
    if POP:
        pool = dedup(only_top(IDMAP)).sort_values(POP, ascending=False)
        return pool.head(10)
    return dedup(IDMAP).head(10)

def random10_cut_exclude(exclude_ids: set[int]) -> pd.DataFrame:
    pool = dedup(only_top(IDMAP))
    if pool is None or pool.empty:
        return pd.DataFrame()
    ids = [i for i in pool.index if i not in exclude_ids]
    if not ids:
        return pd.DataFrame()
    k = min(10, len(ids))
    choice = random.sample(ids, k)
    return IDMAP.loc[choice]

def annoy_similar(row_id: int, k: int = 200) -> pd.DataFrame:
    try:
        idxs, dists = ANNOY.get_nns_by_item(int(row_id), k+1, include_distances=True)
        idxs, dists = idxs[1:], dists[1:]
        df = IDMAP.loc[idxs].copy()
        df["annoy_distance"] = dists
        if POP:
            df = df.sort_values([POP, "annoy_distance"], ascending=[False, True])
        else:
            df = df.sort_values("annoy_distance", ascending=True)
        return df
    except Exception:
        return pd.DataFrame()

if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "q" not in st.session_state:
    st.session_state.q = ""
if "rand_ids" not in st.session_state:
    st.session_state.rand_ids = None

st.markdown('<div class="topbar">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([0.28, 0.54, 0.18])
with c1:
    if st.button("🎧 Spotify Recommender", use_container_width=True):
        st.session_state.selected_id = None
        st.session_state.q = ""
        st.rerun()
with c2:
    st.session_state.q = st.text_input(
        "Search", value=st.session_state.q,
        placeholder="Search tracks or artists…",
        label_visibility="collapsed", key="qbox"
    )
with c3:
    st.markdown('<div style="text-align:right;"><span class="avatar"><small>AK</small></span></div>',
                unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

def open_track(row_id: int):
    st.session_state.selected_id = int(row_id)
    st.session_state.q = ""
    st.rerun()

def _img_or_default(row: pd.Series) -> str:
    url = row.get(IMG)
    if isinstance(url, str) and url.strip():
        return url
    return DEFAULT_COVER

def card(row: pd.Series, row_id: int, key_prefix: str):
    img = _img_or_default(row)
    title = row.get(NAME, "—")
    artist = row.get(ART,  "—")
    pop_txt = f'pop {int(row[POP])}' if POP and pd.notna(row.get(POP)) else ""
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
    if st.button("▶ Open", key=f"{key_prefix}_{int(row_id)}"):
        open_track(int(row_id))

def grid(df: pd.DataFrame, key_prefix: str, take: int = 10, per_row: int = 5):
    if df is None or df.empty:
        st.info("No items.")
        return
    df = dedup(df).head(take)
    rows = list(df.iterrows())
    for i in range(0, len(rows), per_row):
        chunk = rows[i:i+per_row]
        cols = st.columns(len(chunk), gap="large")
        for (rid, r), c in zip(chunk, cols):
            with c: card(r, int(rid), key_prefix)

def hero(row: pd.Series):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.image(_img_or_default(row), width=260)
    with c2:
        st.subheader(row.get(NAME, "—"))
        st.caption(str(row.get(ART, "")))
        if POP and pd.notna(row.get(POP, None)):
            st.markdown(f'**`pop {int(row[POP])}`**')
        if PREV and pd.notna(row.get(PREV, None)):
            st.audio(row[PREV])

if st.session_state.q.strip():
    q = st.session_state.q.lower().strip()
    mask = IDMAP[NAME].astype(str).str.lower().str.contains(q, na=False) | \
           IDMAP[ART].astype(str).str.lower().str.contains(q, na=False)
    res = IDMAP[mask].copy()
    if POP: res = res.sort_values(POP, ascending=False)
    st.subheader(f"🔎 Search: **{st.session_state.q}**")
    grid(res, "search", take=20, per_row=5)
    st.markdown("---")

if POP:
    st.caption(f"Using popularity cutoff at {CUT_Q:.2f} quantile → {CUT:.1f}")

sid = st.session_state.selected_id

if sid is None:
    st.subheader("🔥 Top 10 Global")
    top10 = top10_global()
    grid(top10, "top10", take=10, per_row=5)

    st.subheader("⭐ Random 10")
    if st.button("🎲 Shuffle", key="shuffle"):
        st.session_state.rand_ids = None
        st.rerun()

    exclude = set(top10.index.tolist())
    if st.session_state.rand_ids is None:
        r10_df = random10_cut_exclude(exclude)
        st.session_state.rand_ids = r10_df.index.tolist()
    else:
        r10_df = IDMAP.loc[[i for i in st.session_state.rand_ids if i in IDMAP.index]]
    grid(r10_df, "random", take=10, per_row=5)

else:
    if sid not in IDMAP.index:
        st.warning("Track not found.")
        st.session_state.selected_id = None
        st.rerun()

    seed = IDMAP.loc[sid]
    st.subheader("Now playing")
    hero(seed)

    st.subheader("Recommended for you")

    st.markdown('<div class="section-title">Same artist (top)</div>', unsafe_allow_html=True)
    same_artist = IDMAP[IDMAP[ART] == seed[ART]].copy()
    if POP: same_artist = same_artist.sort_values(POP, ascending=False)
    same_artist = same_artist[same_artist.index != sid]
    grid(same_artist, "same_artist", take=10, per_row=5)

    st.markdown('<div class="section-title">Similar by audio (top)</div>', unsafe_allow_html=True)
    sim = annoy_similar(sid, k=200)
    grid(sim, "sim", take=10, per_row=5)
