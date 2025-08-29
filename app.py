import io
import requests
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------
def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """PIL (RGB) -> OpenCV (BGR)"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """OpenCV (BGR) -> PIL (RGB)"""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Image Processing", layout="wide")
st.title("👀🤳🏾Streamlit Image Processing")

with st.sidebar:
    st.header("แหล่งที่มาของรูปภาพ")
    source = st.radio("เลือกแหล่งภาพ", ["📷Webcam", "📲Upload", "💻Image URL"], index=0)

    img = None
    if source == "📷Webcam":
        # ถ่ายรูปจากเว็บแคม (ผ่านเบราว์เซอร์)
        cap = st.camera_input("ถ่ายภาพจากเว็บแคม")
        if cap is not None:
            img = Image.open(cap).convert("RGB")

    elif source == "📲Upload":
        up = st.file_uploader("อัปโหลดรูป (jpg/png)", type=["jpg", "jpeg", "png"])
        if up is not None:
            img = Image.open(up).convert("RGB")

    else:  # Image URL
        url = st.text_input("วางลิงก์รูปภาพ (ต้องเป็นไฟล์รูป เช่น .jpg .png)", "")
        if st.button("โหลดรูปจาก URL") and url.strip():
            try:
                img = load_image_from_url(url.strip())
            except Exception as e:
                st.error(f"โหลดรูปไม่สำเร็จ: {e}")

    st.divider()
    st.header("ปรับแต่งพารามิเตอร์")
    st.caption("Brightness/Contrast ถูกใช้ก่อนขั้นตอนหลักเสมอ")
    alpha = st.slider("Contrast (alpha)", 0.5, 2.5, 1.0, 0.1)
    beta = st.slider("Brightness (beta)", -100, 100, 0, 1)

    op = st.selectbox(
        "ขั้นตอนประมวลผล",
        ["None", "Grayscale", "Gaussian Blur", "Canny Edge", "Threshold (Binary)"],
        index=0,
    )

    # คอนโทรลเฉพาะขั้นตอน
    if op == "Gaussian Blur":
        k = st.slider("Kernel size (odd)", 1, 51, 9, 2)  # step 2 เพื่อให้ออกเลขคี่ง่าย
        k = ensure_odd(k)
    elif op == "Canny Edge":
        t1 = st.slider("Threshold 1", 0, 255, 100, 1)
        t2 = st.slider("Threshold 2", 0, 255, 200, 1)
    elif op == "Threshold (Binary)":
        thresh = st.slider("Threshold value", 0, 255, 127, 1)

# ไม่มีรูปยัง ไม่ต้องทำต่อ
if img is None:
    st.info("👈🏼เลือก/ถ่าย/วาง URL ของรูปจากแถบด้านซ้ายเพื่อเริ่มต้น")
    st.stop()

# ---------------------------
# Processing
# ---------------------------
# Convert to OpenCV (BGR)
img_bgr = pil_to_cv(img)

# 1) ปรับ Brightness/Contrast
proc = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

# 2) เลือกขั้นตอนหลัก
if op == "Grayscale":
    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
elif op == "Gaussian Blur":
    proc = cv2.GaussianBlur(proc, (k, k), 0)
elif op == "Canny Edge":
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    proc = cv2.Canny(gray, t1, t2)
elif op == "Threshold (Binary)":
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    _, proc = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
# else: "None" → ใช้ proc ปัจจุบัน

# ---------------------------
# แสดงผลลัพธ์
# ---------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("ภาพต้นฉบับ")
    st.image(img, use_container_width=True)
    w, h = img.size
    st.caption(f"ขนาด: {w}×{h} px")

with col2:
    st.subheader("ภาพหลังประมวลผล")
    if proc.ndim == 2:
        # grayscale/edge/threshold → แปลงกลับเป็น RGB เพื่อแสดงสวย ๆ
        show_img = cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)
    else:
        show_img = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    st.image(show_img, use_container_width=True)

    # ปุ่มดาวน์โหลด
    out_pil = Image.fromarray(show_img)
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button("ดาวน์โหลดผลลัพธ์ (PNG)", data=buf.getvalue(), file_name="processed.png")

st.divider()

# ---------------------------
# กราฟคุณสมบัติของรูปภาพ (Histogram)
# ---------------------------
st.subheader("📊 Histogram ของความเข้มพิกเซล (Intensity)")

# ใช้ภาพหลังประมวลผล แปลงเป็นเทาเพื่อทำฮิสโตแกรม
if proc.ndim == 3:
    gray_for_hist = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
else:
    gray_for_hist = proc

hist, bins = np.histogram(gray_for_hist.flatten(), bins=256, range=[0, 256])

fig = plt.figure(figsize=(7, 3.5))
plt.title("Intensity Histogram (0–255)")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.plot(hist)   # ไม่กำหนดสี ตามข้อกำหนดทั่วไป
plt.xlim([0, 255])
st.pyplot(fig, use_container_width=True)

# ค่าเชิงสถิติบางส่วน
mean_val = float(gray_for_hist.mean())
std_val = float(gray_for_hist.std())
st.caption(f"Mean intensity: {mean_val:.2f} | Std: {std_val:.2f}")
