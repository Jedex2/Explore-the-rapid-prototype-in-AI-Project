# camera_test.py
import streamlit as st
st.title("Webcam test")
img = st.camera_input("ถ่ายภาพจากเว็บแคม")
if img: st.image(img)
