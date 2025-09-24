# utils_weights.py
import os, pathlib, json, urllib.request
import streamlit as st
from urllib.error import HTTPError, URLError

def _download(url: str, out: pathlib.Path, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out, "wb") as f:
        f.write(r.read())

def _download_github_private(out: pathlib.Path):
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo  = st.secrets.get("GITHUB_REPO", "")
    tag   = st.secrets.get("GITHUB_TAG", "")
    asset = st.secrets.get("GITHUB_ASSET", "best.pt")
    if not all([token, repo, tag, asset]):
        raise RuntimeError("GitHub private download requires secrets")

    # 1) release 情報取得
    api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    headers = {"Authorization": f"token {token}", "User-Agent": "streamlit-app"}
    req = urllib.request.Request(api, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        info = json.loads(r.read().decode("utf-8"))
    target = None
    for a in info.get("assets", []):
        if a.get("name") == asset:
            target = a["url"]  # assets API URL
            break
    if not target:
        raise RuntimeError(f"Asset '{asset}' not found in release {tag}")

    # 2) Asset ダウンロード
    headers |= {"Accept": "application/octet-stream"}
    _download(target, out, headers=headers)

def ensure_weights(local_path: str = "weights/best.pt") -> str:
    """Download weights once into `weights/` and return local path."""
    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 0:
        return str(p)

    url = st.secrets.get("WEIGHTS_URL", "").strip()

    try:
        with st.spinner("weights をダウンロード中..."):
            if url:
                _download(url, p, timeout=60)
            else:
                _download_github_private(p)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"weights の取得に失敗しました: {e}") from e

    return str(p)
