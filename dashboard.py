import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ==============================
# 💫 KONFIGURASI HALAMAN
# ==============================
st.set_page_config(
    page_title="💗 Smart Vision Dashboard",
    layout="wide",
    page_icon="🌸"
)

# ==============================
# 🧠 LOAD MODEL
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Yumnaa Alifah_Laporan 4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier

with st.spinner("💞 Sedang memuat model..."):
    yolo_model, classifier = load_models()
    time.sleep(1)
st.success("🌸 Model berhasil dimuat!")

# ==============================
# 🎨 SIDEBAR MENU
# ==============================
st.sidebar.title("🌟 Menu Dashboard")
menu = st.sidebar.radio("Pilih Mode:", ["💎 Deteksi Objek (YOLO)", "♻️ Klasifikasi Sampah"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

# ==============================
# 🖼️ KONTEN UTAMA
# ==============================
st.title("📸 Aplikasi Deteksi & Klasifikasi Gambar")
st.markdown("Gunakan AI untuk mendeteksi objek atau mengenali jenis sampah 🌿")

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
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# ==========================
# Load model
# ==========================
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("Yumnaa_Alifah_Laporan 2.h5")
    return model

model = load_models()

# ==========================
# Upload image
# ==========================
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Resize agar sesuai input model
    img_resized = img.resize((128, 128))
    st.image(img_resized, caption="Gambar yang diunggah", use_container_width=True)

    # Konversi ke array dan normalisasi
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3)
    img_array = img_array / 255.0  # Normalisasi

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    st.write("### Hasil Prediksi:")
    st.write(f"Prediksi kelas: **{predicted_class[0]}**")
