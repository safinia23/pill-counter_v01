# utils_weights.py
import os
import pathlib
import urllib.request
import streamlit as st


def ensure_weights(local_path: str = "weights/best.pt") -> str:
    """
    Private GitHub Release から best.pt を取得してローカルに保存。
    - WEIGHTS_URL: Release のダウンロードURL（例: .../releases/download/v1.0/best.pt）
    - GITHUB_TOKEN: Fine-grained PAT（repo contents read 権限）
    既にファイルがある場合はダウンロードしません。
    """
    url = st.secrets.get("WEIGHTS_URL", "")
    token = st.secrets.get("GITHUB_TOKEN", "")

    if not url:
        raise RuntimeError("WEIGHTS_URL が Secrets に設定されていません。")

    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return str(p)

    # GitHub に認証して取りに行く（Private リポ対応）
    req = urllib.request.Request(url)
    # 認証＋バイナリ取得＋UA を明示
    if token:
        req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/octet-stream")
    req.add_header("User-Agent", "pill-counter-app/1.0")

    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
        return str(p)
    except Exception as e:
        # デバッグしやすいよう URL も出しておく（本番で隠したければ消してください）
        raise RuntimeError(f"weights の取得に失敗しました: {e}  url={url}")
