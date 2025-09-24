# utils_weights.py
import os
import pathlib
import urllib.request
import streamlit as st


def ensure_weights(local_path="weights/best.pt") -> str:
    import os, pathlib, requests, streamlit as st

    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 1) GitHub Token 経由（private 想定）を最優先
    gh_token = st.secrets.get("GITHUB_TOKEN", "")
    if gh_token:
        repo = st.secrets["GITHUB_REPO"]
        tag  = st.secrets["GITHUB_TAG"]
        asset_name = st.secrets.get("GITHUB_ASSET", p.name)

        api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        r = requests.get(api, headers={"Authorization": f"token {gh_token}",
                                       "Accept": "application/vnd.github+json"})
        r.raise_for_status()
        assets = r.json().get("assets", [])
        url = next(a["browser_download_url"] for a in assets if a["name"] == asset_name)

        with requests.get(url, headers={"Authorization": f"token {gh_token}"}, stream=True) as g:
            g.raise_for_status()
            with open(p, "wb") as f:
                for chunk in g.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
        return str(p)

    # 2) フォールバック：公開直リンク（public repo のみ）
    url = st.secrets.get("WEIGHTS_URL", "")
    if url:
        import urllib.request
        urllib.request.urlretrieve(url, str(p))
        return str(p)

    raise RuntimeError("weights の取得経路がありません（GITHUB_TOKEN か WEIGHTS_URL を secrets に設定してください）")
