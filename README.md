# 👀🤳🏾 Streamlit Image Processing (How to Use)

แอปสาธิตประมวลผลภาพด้วย **Streamlit + OpenCV + PIL + NumPy + Matplotlib**  
รองรับภาพจาก **Webcam**, **Upload**, และ **Image URL** พร้อมปรับพารามิเตอร์แบบโต้ตอบและแสดงกราฟฮิสโตแกรม

---

## Requirements
- Python 3.10–3.12  
- Packages (ตามที่ `import` ในโค้ด):
  - `streamlit`
  - `opencv-python-headless`
  - `pillow`
  - `numpy`
  - `matplotlib`
  - `requests`

ติดตั้งแบบรวดเร็ว:
```bash
pip install streamlit opencv-python-headless pillow numpy matplotlib requests
```
## Run

1.บันทึกโค้ดเป็นไฟล์ app.py

2.เปิดเทอร์มินัล (VS Code ใช้ Terminal ในตัวได้) แล้วรัน:
streamlit run app.py

3.เบราว์เซอร์จะเปิด http://localhost:8501

ครั้งแรกที่ใช้ Webcam ให้กด Allow camera ในเบราว์เซอร์

## How to Use

1.เลือกแหล่งภาพจาก Sidebar:

📷 Webcam: ถ่ายภาพด้วย st.camera_input()

📲 Upload: เลือกไฟล์ .jpg/.jpeg/.png

💻 Image URL: วางลิงก์ไฟล์รูปโดยตรง (ลงท้าย .jpg/.png) แล้วกด โหลดรูปจาก URL

2.ปรับพารามิเตอร์ก่อนประมวลผล:

Contrast (alpha): 0.5–2.5

Brightness (beta): -100–100
(สองค่านี้ถูกใช้ก่อนขั้นตอนหลักเสมอ)

3.เลือกขั้นตอนหลัก (Processing):

None – ไม่ทำเพิ่ม

Grayscale – แปลงภาพเป็นเทา

Gaussian Blur – เบลอ (ตั้งค่า Kernel size เป็นเลขคี่)

Canny Edge – ตรวจจับขอบ (ตั้งค่า Threshold 1/2)

Threshold (Binary) – แปลงเป็นภาพขาวดำจากค่าธรेशโฮลด์

4.ดูผลลัพธ์:

ภาพต้นฉบับ (ซ้าย) และ ภาพหลังประมวลผล (ขวา)

กด ดาวน์โหลดผลลัพธ์ (PNG) ได้

5.ดูกราฟคุณสมบัติภาพ:

Intensity Histogram (0–255) ของภาพหลังประมวลผล

แสดงค่า Mean และ Std ของความเข้มพิกเซล
