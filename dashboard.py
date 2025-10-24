import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Dashboard Deteksi & Klasifikasi Citra",
    page_icon="💗",
    layout="wide",
)

# ==========================
# CSS Tema Pink Lembut
# ==========================
st.markdown("""
    <style>
        /* Warna utama halaman */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #ffe6f0 0%, #fff5fa 100%);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8bbd0, #fce4ec);
            color: #4a148c;
        }

        /* Judul dan subjudul */
        h1, h2, h3, h4 {
            color: #e91e63 !important;
            font-family: 'Poppins', sans-serif;
        }

        /* Tombol */
        .stButton>button {
            background-color: #ec407a !important;
            color: white !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #d81b60 !important;
            transform: scale(1.05);
        }

        /* Footer info */
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            color: #ad1457;
            font-size: 14px;
            font-family: 'Poppins', sans-serif;
        }

        /* Box hasil */
        .result-box {
            background-color: #fff0f6;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 10px rgba(233, 30, 99, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Model
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Yumnaa Alifah_Laporan 4.pt")  # Model deteksi YOLO
    classifier = tf.keras.models.load_model("classifier_model.h5")  # Model klasifikasi .h5
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Header Dashboard
# ==========================
st.title("💗 Aplikasi Deteksi & Klasifikasi Citra")
st.markdown("### Oleh: **Yumnaa Alifah | Statistika Universitas Syiah Kuala**")

st.write("Unggah gambar di bawah ini untuk mendeteksi objek dan mengklasifikasikannya menggunakan model yang telah dilatih.")

# ==========================
# Upload File
# ==========================
uploaded_file = st.file_uploader("📤 Unggah gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Tombol deteksi
    if st.button("🔍 Jalankan Deteksi"):
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        # Deteksi dengan YOLO
        results = yolo_model.predict(np.array(image))
        st.subheader("📸 Hasil Deteksi YOLO")
        for result in results:
            result_image = result.plot()
            st.image(result_image, caption="Hasil Deteksi", use_container_width=True)

        # Klasifikasi menggunakan model .h5
        img = image.resize((128, 128))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        predictions = classifier.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        st.subheader("🎯 Hasil Klasifikasi")
        st.write(f"**Kelas Prediksi:** {predicted_class}")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("""
    <div class='footer'>
        💕 Dashboard ini dibuat oleh <b>Yumnaa Alifah</b><br>
        Mahasiswi Statistika, Universitas Syiah Kuala
    </div>
""", unsafe_allow_html=True)
