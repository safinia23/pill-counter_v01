# app_general.py
import io, numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO
from utils_weights import ensure_weights

def show_image(img, caption=None):
    """
    Streamlit のバージョン差異に対応して画像を全幅表示するヘルパー。
    新: use_container_width / 旧: use_column_width
    """
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        # 古い Streamlit でも動く
        st.image(img, caption=caption, use_column_width=True)


st.set_page_config(page_title="Pill-counter (General)", layout="wide")

@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return YOLO(path)

def visualize(pil, boxes, scores, conf):
    img = pil.convert("RGBA"); ov = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    try: font = ImageFont.truetype("arial.ttf", 24)
    except: font = ImageFont.load_default()
    kept = 0
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        if float(s) < conf: continue
        kept += 1
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0,255), width=3)
        draw.rectangle([x1,y1,x2,y2], fill=(0,255,0,60))
        draw.text((x1+3,y1+3), f"{s:.2f}", fill=(255,255,255,255), font=font)
    return Image.alpha_composite(img, ov).convert("RGB"), kept

st.title("💊Pill-counter")
st.caption("画像アップロード or 静止画カメラで検出します")

with st.sidebar:
    st.header("⚙️ 設定")
    conf = st.slider("しきい値 (conf)", 0.1, 1.0, 0.5, 0.01)
    iou = st.slider("NMS IoU", 0.3, 0.8, 0.5, 0.01)
    max_det = st.slider("最大検出数", 50, 500, 300, 10)
    imgsz = st.select_slider("推論解像度 (imgsz)", [640, 960, 1024], value=960)
    tta = st.checkbox("TTA（推論を強く）", value=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader("入力方法を選択")
    mode = st.radio("", ["画像ファイルを選択", "カメラで撮影（静止画）"], horizontal=True)
    src = None
    if mode == "画像ファイルを選択":
        src = st.file_uploader("画像ファイル", type=["jpg","jpeg","png"])
    else:
        src = st.camera_input("カメラで撮影")

if not src: st.stop()
pil = Image.open(src).convert("RGB")

with st.spinner("推論中..."):
    weights_path = ensure_weights()  # SecretsのURLからDL
    model = get_model(weights_path)
    res = model.predict(source=pil, conf=conf, iou=iou, max_det=max_det,
                        imgsz=imgsz, augment=tta, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy(); scores = res.boxes.conf.cpu().numpy()
    else:
        boxes = np.zeros((0,4)); scores = np.zeros((0,))
    vis, count = visualize(pil, boxes, scores, conf)

with col2:
    st.subheader("検出結果")
    show_image(vis)
    st.metric("検出個数", f"{count} 個")
    buf = io.BytesIO(); vis.save(buf, format="PNG")
    st.download_button("結果画像をダウンロード", data=buf.getvalue(),
                       file_name="result_bbox.png", mime="image/png")

with col2:
    st.subheader("入力画像")
    show_image(pil)
    buf2 = io.BytesIO(); pil.save(buf2, format="PNG")
    st.download_button("入力画像をダウンロード", data=buf2.getvalue(),
                       file_name="input_image.png", mime="image/png")

st.markdown("""---<div style="text-align:center;color:gray;font-size:.9em;">
© 2025 andChange All rights reserved.</div>""", unsafe_allow_html=True)
