# scripts/enrich_images_sample.py
import os, time, csv, random
from pathlib import Path
import pandas as pd

# опционально .env (локальная разработка)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

ART_DIR   = Path("artifacts")
IDMAP_CSV = ART_DIR / "id_map.csv"
OUT_CSV   = ART_DIR / "id_map.enriched.csv"

LIMIT     = 500     # сколько треков реально обогащаем через API
BATCH     = 50      # лимит споти
SLEEP     = 0.15    # пауза между батчами
RAND_SEED = 42      # чтобы рандом был воспроизводим

def get_spotify_client():
    # сначала берём из streamlit secrets (если есть), иначе из env
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    sec = os.getenv("SPOTIFY_CLIENT_SECRET")
    assert cid and sec, "Set SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET in env/.env"
    auth = SpotifyClientCredentials(client_id=cid, client_secret=sec)
    return spotipy.Spotify(auth_manager=auth, requests_timeout=20, retries=3)

def main():
    assert IDMAP_CSV.exists(), f"not found: {IDMAP_CSV}"
    df = pd.read_csv(IDMAP_CSV)

    # гарантируем столбцы
    for col in ["image_url", "preview_url", "album_name"]:
        if col not in df.columns:
            df[col] = None

    # нормализуем типы
    if "track_id" not in df.columns:
        raise RuntimeError("id_map.csv must contain track_id column")
    df["track_id"] = df["track_id"].astype(str)

    # берём первые LIMIT треков с ненаполненными image_url
    to_fetch_mask = df["image_url"].isna() | (df["image_url"].astype(str).str.len() == 0)
    fetch_indices = df.index[to_fetch_mask].tolist()[:LIMIT]
    fetch_ids     = df.loc[fetch_indices, "track_id"].tolist()

    print(f"Total tracks: {len(df):,}. Will enrich via API: {len(fetch_ids)} (LIMIT={LIMIT})")

    sp = get_spotify_client()
    enriched_images = []   # пул картинок, чтобы раздать остальным

    # — обогащаем батчами только LIMIT штук
    for i in range(0, len(fetch_ids), BATCH):
        chunk_ids = fetch_ids[i:i+BATCH]
        try:
            resp = sp.tracks(chunk_ids)
        except spotipy.exceptions.SpotifyException as e:
            retry_after = getattr(e, "http_headers", {}).get("Retry-After")
            if retry_after:
                time.sleep(int(retry_after) + 1)
                resp = sp.tracks(chunk_ids)
            else:
                print("Spotify error:", e); time.sleep(1); continue
        except Exception as e:
            print("API error:", e); time.sleep(1); continue

        for t in (resp.get("tracks") or []):
            if not t: 
                continue
            tid = str(t.get("id"))
            album = t.get("album") or {}
            images = album.get("images") or []
            image = images[0]["url"] if images else None
            preview = t.get("preview_url")
            album_name = album.get("name")

            mask = df["track_id"] == tid
            if image:
                df.loc[mask, "image_url"] = image
                enriched_images.append(image)
            if preview is not None:
                df.loc[mask, "preview_url"] = preview
            if album_name is not None:
                df.loc[mask, "album_name"] = album_name

        time.sleep(SLEEP)

    # — остальным раздаём случайные картинки из пула enriched_images
    enriched_images = list(dict.fromkeys(enriched_images))  # уникальные, сохраним порядок
    print(f"Collected {len(enriched_images)} unique cover URLs from API.")

    if enriched_images:
        random.seed(RAND_SEED)
        remaining_mask = df["image_url"].isna() | (df["image_url"].astype(str).str.len() == 0)
        remaining_idx = df.index[remaining_mask].tolist()
        for idx in remaining_idx:
            df.at[idx, "image_url"] = random.choice(enriched_images)
    else:
        print("Warning: no images were collected from API; no fallback assignment will be made.")

    # сохраняем
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print("✅ Saved:", OUT_CSV)
    print("👉 Review and rename to artifacts/id_map.csv when ready.")

if __name__ == "__main__":
    main()
