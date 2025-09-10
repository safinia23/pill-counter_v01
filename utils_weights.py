import os, pathlib, urllib.request, streamlit as st

def ensure_weights(local_path: str = "weights/best.pt") -> str:
    url = st.secrets.get("WEIGHTS_URL", "")
    if not url:
        st.stop()  # 安全に停止
    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with st.spinner("重みをダウンロード中..."):
            urllib.request.urlretrieve(url, str(p))
    return str(p)
