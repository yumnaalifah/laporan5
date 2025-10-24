import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==============================
# 🧠 LOAD MODEL
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Yumnaa_Alifah_Laporan_4.pt")  # model YOLO
    classifier = tf.keras.models.load_model("model/Yumnaa_Alifah_Laporan 2.h5")  # model klasifikasi
    return yolo_model, classifier


yolo_model, classifier = load_models()

# ==============================
# 🎨 SIDEBAR MENU
# ==============================
st.sidebar.title("🌟 Menu Dashboard")
menu = st.sidebar.radio("Pilih Mode:", ["💎 Deteksi Objek (YOLO)", "♻️ Klasifikasi Sampah"])

# ==============================
# 📤 UPLOAD FILE
# ==============================
st.title("📸 Aplikasi Deteksi & Klasifikasi Gambar")
uploaded_file = st.file_uploader("Unggah gambar di sini (format JPG/PNG)", type=["jpg", "jpeg", "png"])

# ==============================
# 🔍 PROSES GAMBAR
# ==============================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diupload", use_container_width=True)

    # MODE 1: DETEKSI OBJEK
    if menu == "💎 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        with st.spinner("🚀 Model sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="💖 Hasil Deteksi Objek", use_container_width=True)
        st.success("✨ Deteksi selesai!")

    # MODE 2: KLASIFIKASI SAMPAH
    elif menu == "♻️ Klasifikasi Sampah":
        st.subheader("🌿 Hasil Klasifikasi Gambar")
        with st.spinner("💫 Sedang memproses gambar..."):
            # --- PREPROCESSING GAMBAR ---
            img_resized = img.resize((224, 224))  # sesuaikan dengan input model
            img_array = np.array(img_resized) / 255.0  # normalisasi
            img_array = np.expand_dims(img_array, axis=0)  # tambahkan batch dimensi

            # --- PREDIKSI ---
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

        # --- LABEL KATEGORI SAMPAH ---
        waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
        predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else "Tidak Dikenali"

        # --- TAMPILKAN HASIL ---
        st.success(f"✅ Hasil Klasifikasi: **{predicted_label}**")
        st.write(f"📊 Tingkat keyakinan: **{confidence:.2%}**")

else:
    st.info("📂 Silakan upload gambar terlebih dahulu untuk memulai.")
