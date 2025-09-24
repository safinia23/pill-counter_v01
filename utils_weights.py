# utils_weights.py
import os
import pathlib
import requests
import streamlit as st

def _download_with_github_api(repo: str, tag: str, asset_name: str, token: str, dst_path: pathlib.Path):
    # 1) タグから Release 情報を取得
    rel_api = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    hdr = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    r = requests.get(rel_api, headers=hdr, timeout=60)
    r.raise_for_status()
    data = r.json()

    # 2) アセット一覧から目的のアセットを探す
    assets = data.get("assets", [])
    target = next((a for a in assets if a.get("name") == asset_name), None)
    if not target:
        raise FileNotFoundError(
            f"Release '{tag}' に '{asset_name}' が見つかりません。\n"
            f"存在するアセット: {[a.get('name') for a in assets]}"
        )

    # 3) アセット API の URL を使ってダウンロード（←ここがポイント）
    asset_api_url = target["url"]  # browser_download_url ではなく url を使う
    dl_hdr = {
        "Authorization": f"token {token}",
        "Accept": "application/octet-stream",  # 実体を返す指定
    }
    with requests.get(asset_api_url, headers=dl_hdr, stream=True, timeout=300) as g:
        g.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in g.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def ensure_weights(local_path: str = "weights/best.pt") -> str:
    """
    - GITHUB_TOKEN, GITHUB_REPO, GITHUB_TAG, GITHUB_ASSET があれば GitHub API で private Release から取得
    - それが無い場合のみ WEIGHTS_URL（public 直リンク）を使う
    """
    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.exists() and p.stat().st_size > 0:
        return str(p)

    # まずは Token 経由（private 用）
    token = st.secrets.get("GITHUB_TOKEN", "").strip()
    if token:
        repo  = st.secrets["GITHUB_REPO"].strip()
        tag   = st.secrets["GITHUB_TAG"].strip()
        asset = st.secrets.get("GITHUB_ASSET", p.name).strip()
        try:
            _download_with_github_api(repo, tag, asset, token, p)
            return str(p)
        except Exception as e:
            raise RuntimeError(
                f"GitHub API 経由の weights 取得に失敗しました: {e}"
            ) from e

    # フォールバック（public 直リンク）
    url = st.secrets.get("WEIGHTS_URL", "").strip()
    if url:
        import urllib.request
        try:
            urllib.request.urlretrieve(url, str(p))
            return str(p)
        except Exception as e:
            raise RuntimeError(f"WEIGHTS_URL からの取得に失敗しました: {e}") from e

    raise RuntimeError("weights の取得経路がありません（GITHUB_* か WEIGHTS_URL を secrets に設定してください）")
