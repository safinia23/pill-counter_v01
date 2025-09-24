# app_ipad.py
import io, numpy as np, av
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from utils_weights import ensure_weights

st.set_page_config(page_title="Pill-counter (for iPad)", layout="wide")
st.markdown("""
<style>
video { width:100% !important; height:auto !important; object-fit:cover !important; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

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

st.title("💊Pill-counter（For iPad）")
st.caption("背面カメラ（WebRTC）でライブ表示 → 撮影して推論")

with st.sidebar:
    st.header("⚙️ 設定")
    conf = st.slider("しきい値 (conf)", 0.1, 1.0, 0.5, 0.01)
    iou = st.slider("NMS IoU", 0.3, 0.8, 0.5, 0.01)
    max_det = st.slider("最大検出数", 50, 500, 300, 10)
    imgsz = st.select_slider("推論解像度 (imgsz)", [640, 960, 1024], value=960)
    tta = st.checkbox("TTA（推論を強く）", value=False)

rtc_config = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
media_constraints = {
    "video": {
        "facingMode": {"exact": "environment"},  # 背面固定（失敗する端末は "environment" に変更）
        "width": {"ideal": 1280}, "height": {"ideal": 720},
    },
    "audio": False,
}

class FrameGrabber(VideoTransformerBase):
    def __init__(self): self.last = None
    def recv(self, frame: av.VideoFrame):
        self.last = frame.to_ndarray(format="bgr24")
        return frame

webrtc_ctx = webrtc_streamer(
    key="pill-ipad",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints=media_constraints,
    video_transformer_factory=FrameGrabber,
    async_processing=True,
)

pil = None
if webrtc_ctx and webrtc_ctx.video_transformer:
    if st.button("📸 この映像を撮影して解析する"):
        bgr = webrtc_ctx.video_transformer.last
        if bgr is not None:
            pil = Image.fromarray(bgr[:, :, ::-1])  # BGR→RGB
        else:
            st.warning("カメラ初期化中です。数秒後に再試行してください。")

if pil is None: st.stop()

with st.spinner("推論中..."):
    path = ensure_weights()
    model = get_model(path)
    res = model.predict(source=pil, conf=conf, iou=iou, max_det=max_det,
                        imgsz=imgsz, augment=tta, verbose=False)[0]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy(); scores = res.boxes.conf.cpu().numpy()
    else:
        boxes = np.zeros((0,4)); scores = np.zeros((0,))
    vis, count = visualize(pil, boxes, scores, conf)

col1, col2 = st.columns(2)
with col1:
    st.subheader("検出結果")
    st.image(vis, use_container_width=True)
    st.metric("検出個数", f"{count} 個")
    buf = io.BytesIO(); vis.save(buf, format="PNG")
    st.download_button("結果画像をダウンロード", data=buf.getvalue(),
                       file_name="result_bbox.png", mime="image/png")

with col2:
    st.subheader("入力画像")
    st.image(pil, use_container_width=True)
    buf2 = io.BytesIO(); pil.save(buf2, format="PNG")
    st.download_button("入力画像をダウンロード", data=buf2.getvalue(),
                       file_name="input_image.png", mime="image/png")

st.markdown("""---<div style="text-align:center;color:gray;font-size:.9em;">
© 2025 andChange All rights reserved.</div>""", unsafe_allow_html=True)
