# Streamlit Image Processing Demo

แอปตัวอย่างด้วย **Streamlit + OpenCV + PIL + Matplotlib** สำหรับ:
- เปิดรูปจาก **เว็บแคม**, **อัปโหลดไฟล์**, หรือ **ลิงก์ URL** บนอินเทอร์เน็ต
- ทำ **image processing แบบง่าย** (Brightness/Contrast, Grayscale, Gaussian Blur, Canny, Threshold)
- ปรับค่าพารามิเตอร์ได้จาก **GUI** ที่สร้างด้วย Streamlit
- แสดง **ผลลัพธ์ภาพ** หลังประมวลผล และ **กราฟฮิสโตแกรม** ของความเข้มพิกเซล

> เหมาะกับการสอน/เดโมคอมพิวเตอร์วิทัศน์เบื้องต้น บน Windows/macOS/Linux และใช้งานดีใน VS Code

---

## ✨ คุณสมบัติหลัก (ครบตามโจทย์)
- ✅ เปิดกล้องจากเบราว์เซอร์ผ่าน `st.camera_input()` (Firefox/Chrome/Edge)  
- ✅ โหลดภาพจาก **ไฟล์** หรือ **URL** (ต้องเป็นลิงก์ไปยังไฟล์ `.jpg/.png` โดยตรง)
- ✅ ปรับ **Brightness/Contrast** และเลือกโอเปอเรชัน **Grayscale/Blur/Canny/Threshold**
- ✅ แสดงผลต้นฉบับ/หลังประมวลผล + ปุ่ม **ดาวน์โหลดผลลัพธ์**
- ✅ แสดงกราฟ **Intensity Histogram** + ค่า **mean/std**

---

## 🗂 โครงสร้างโปรเจกต์
