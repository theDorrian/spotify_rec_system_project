import streamlit as st
import pandas as pd
import random

# =========================
# Загружаем данные
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("tracks.csv")  # твой CSV с колонками
    return df

df = load_data()

# =========================
# Настройки
# =========================
CUTOFF = df["popularity"].quantile(0.70)

# =========================
# Стили
# =========================
st.markdown(
    """
    <style>
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem 1rem 1rem;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .search-box {
        flex: 1;
        max-width: 300px;
    }
    .account-placeholder {
        background: #222;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        opacity: 0.8;
    }

    .card {
      background: rgba(255,255,255,0.02);
      border: 1px solid #2a3242;
      border-radius: 14px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 8px;
      height: 100%;
      min-height: 280px;
      max-width: 200px;
    }

    .card .cover {
      width: 100%;
      height: 200px;
      border-radius: 10px;
      overflow: hidden;
      background: #1b2335;
    }

    .card .cover img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .card .title {
      font-weight: 600;
      font-size: 0.9rem;
      line-height: 1.2;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      min-height: 36px;
    }

    .card .artist {
      opacity: .85;
      font-size: 0.85rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .card .meta {
      min-height: 18px;
      font-size: 0.8rem;
    }

    .pop-pill {
      display: inline-block;
      border: 1px solid #2a3242;
      border-radius: 999px;
      padding: 1px 8px;
      font-size: .75rem;
      opacity: .9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# UI Header
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown('<div class="app-title">🎧 Spotify Recommender</div>', unsafe_allow_html=True)
with col2:
    search_query = st.text_input("Поиск трека", "", label_visibility="collapsed", key="search")
with col3:
    st.markdown('<div class="account-placeholder">Аккаунт</div>', unsafe_allow_html=True)

# =========================
# Состояние
# =========================
if "selected_track" not in st.session_state:
    st.session_state.selected_track = None

def render_track(row, key_prefix=""):
    with st.container():
        st.markdown(
            f"""
            <div class="card">
              <div class="cover"><img src="https://placehold.co/200x200?text=Track" /></div>
              <div class="title">{row['track_name']}</div>
              <div class="artist">{row['artist_name']}</div>
              <div class="meta"><span class="pop-pill">pop {row['popularity']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("▶ Open", key=f"{key_prefix}_{row.name}"):
            st.session_state.selected_track = row

# =========================
# Главная страница
# =========================
if st.session_state.selected_track is None:
    st.markdown("### 🔥 Top 10 Global")
    top10 = df.sort_values("popularity", ascending=False).head(10)
    cols = st.columns(5)
    for i, row in top10.iterrows():
        with cols[i % 5]:
            render_track(row, key_prefix="top10")

    st.markdown("### ⭐ Random 10")
    eligible = df[df["popularity"] >= CUTOFF]
    exclude = set(top10["track_id"])
    pool = eligible[~eligible["track_id"].isin(exclude)]
    sample = pool.sample(10) if len(pool) >= 10 else pool
    cols = st.columns(5)
    for i, row in sample.iterrows():
        with cols[i % 5]:
            render_track(row, key_prefix="rand10")

    if search_query:
        st.markdown("### 🔍 Search Results")
        results = df[df["track_name"].str.contains(search_query, case=False, na=False)].head(10)
        cols = st.columns(5)
        for i, row in results.iterrows():
            with cols[i % 5]:
                render_track(row, key_prefix="search")

# =========================
# Страница трека
# =========================
else:
    track = st.session_state.selected_track
    st.markdown("## Now playing")
    st.markdown(
        f"""
        <div class="card" style="max-width:300px;">
          <div class="cover"><img src="https://placehold.co/300x300?text=Album" /></div>
          <div class="title">{track['track_name']}</div>
          <div class="artist">{track['artist_name']}</div>
          <div class="meta"><span class="pop-pill">pop {track['popularity']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Recommended for you")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Same artist (top)")
        same_artist = df[(df["artist_name"] == track["artist_name"]) & (df["track_id"] != track["track_id"])].head(10)
        cols = st.columns(5)
        for i, row in same_artist.iterrows():
            with cols[i % 5]:
                render_track(row, key_prefix="sameartist")

    with col2:
        st.markdown("### Similar by audio (top)")
        # Заглушка — просто топ по popularity
        similar = df[df["track_id"] != track["track_id"]].sample(10)
        cols = st.columns(5)
        for i, row in similar.iterrows():
            with cols[i % 5]:
                render_track(row, key_prefix="similar")

    if search_query:
        st.markdown("### 🔍 Search Results")
        results = df[df["track_name"].str.contains(search_query, case=False, na=False)].head(10)
        cols = st.columns(5)
        for i, row in results.iterrows():
            with cols[i % 5]:
                render_track(row, key_prefix="search_in_track")

    if st.button("⬅ Back"):
        st.session_state.selected_track = None
