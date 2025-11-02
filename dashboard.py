import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="💖 Image Detection & Classification Dashboard",
    layout="wide",
    page_icon="🌸"
)

# ==========================
# Custom CSS (Tema Pink)
# ==========================
st.markdown(
    """
    <style>
    .main {
        background-color: #ffe6f0; /* Latar belakang pink muda */
    }

    h1, h2, h3, h4, h5, h6 {
        color: #d63384; /* Warna judul pink tua */
    }

    .stButton>button {
        background-color: #ff80b5; /* Pink lembut */
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #ff4da6; /* Pink lebih gelap saat hover */
        transform: scale(1.05);
    }

    .stSidebar {
        background-color: #ffe6f0; /* Sidebar pink muda */
    }

    .stProgress > div > div > div > div {
        background-color: #ff66b2; /* Progress bar pink */
    }

    .css-1d391kg, .css-18e3th9 {
        background-color: #ffe6f0 !important;
    }

    .stRadio > div {
        color: #d63384 !important;
        font-weight: 600;
    }

    .stAlert {
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Yumnaa_Alifah_Laporan_4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/Yumnaa_Alifah_Laporan_2.keras")  # Model Klasifikasi
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model... Mohon tunggu sebentar"):
    yolo_model, classifier = load_models()
    time.sleep(1)

st.success("🌸 Model berhasil dimuat!")

# ==========================
# Sidebar Menu
# ==========================
st.sidebar.title("⚙ Pengaturan")
st.sidebar.markdown("<h3 style='color:#d63384;'>💗 Pilih Mode Analisis</h3>", unsafe_allow_html=True)

menu = st.sidebar.radio("", ["📦 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar (Waste)"])
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Bagian Utama
# ==========================
st.title("🌸 Aplikasi Deteksi & Klasifikasi Citra")
st.markdown("<p style='color:#d63384; font-weight:bold;'>Dikembangkan oleh: Yumnaa Alifah 💕</p>", unsafe_allow_html=True)
st.markdown("Gunakan aplikasi ini untuk mendeteksi objek atau mengklasifikasikan gambar secara otomatis dengan model AI yang lucu dan cerdas! 💖")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼 Gambar yang Diupload", use_container_width=True)

    if menu == "📦 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        with st.spinner("🚀 Sedang melakukan deteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📦 Hasil Deteksi Objek", use_container_width=True)

        st.success("💖 Deteksi selesai!")
        st.info("Model mendeteksi objek seperti *mobil*, *supercar*, atau *laptop* dari gambar.")

    elif menu == "🧠 Klasifikasi Gambar (Waste)":
        st.subheader("♻ Hasil Klasifikasi Gambar")
        with st.spinner("🌷 Sedang memproses gambar..."):
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        waste_labels = ["Sampah Kaca", "Sampah Logam", "Sampah Kertas", "Sampah Plastik", "Sampah Organik"]
        predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else "Tidak Dikenali"

        st.markdown(f"<h3 style='color:#d63384;'>♻ Jenis Sampah: <b>{predicted_label}</b></h3>", unsafe_allow_html=True)
        st.progress(float(confidence))
        st.caption(f"💫 Probabilitas: {confidence:.2%}")

        if confidence > 0.80:
            st.success("🌟 Prediksi sangat akurat!")
        elif confidence > 0.50:
            st.warning("🌼 Prediksi cukup akurat, namun bisa ditingkatkan.")
        else:
            st.error("💔 Prediksi rendah — coba gambar lain.")

else:
    st.info("⬅ Silakan unggah gambar dari sidebar untuk memulai analisis 💕")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#d63384;'>© 2025 | Dashboard Deteksi & Klasifikasi Citra oleh <b>Yumnaa Alifah</b> 🌸</p>",
    unsafe_allow_html=True
)

