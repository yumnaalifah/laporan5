import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

# ============================
# KONFIGURASI DASAR DASHBOARD
# ============================
st.set_page_config(
    page_title="💗 Smart Vision Dashboard by Yumnaa",
    layout="wide",
    page_icon="🌸"
)

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #fff6fa;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8bbd0, #fce4ec);
            color: #4a148c;
        }
        h1, h2, h3, h4 {
            color: #e91e63 !important;
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            background-color: #ec407a !important;
            color: white !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)


# ----------------------------
# CSS TEMA PINK CUSTOM
# ----------------------------
pink_css = """
<style>
/* Warna dasar */
body {
    background-color: #fff6fa;
}

/* Header */
h1, h2, h3, h4 {
    color: #e91e63 !important;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8bbd0, #fce4ec);
    color: #4a148c;
}

/* Tombol */
button, .stButton>button {
    background-color: #ec407a !important;
    color: white !important;
    border-radius: 15px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* Judul */
.title-text {
    font-family: 'Poppins', sans-serif;
    color: #d81b60;
    font-weight: 700;
    text-align: center;
    font-size: 40px;
}

/* Subtitle */
.subtext {
    text-align: center;
    font-size: 18px;
    color: #ad1457;
}

/* Card style */
.block-container {
    border-radius: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #880e4f;
    font-size: 15px;
    margin-top: 60px;
}
</style>
"""
st.markdown(pink_css, unsafe_allow_html=True)

# ----------------------------
# INFO HEADER
# ----------------------------
st.markdown("<div class='title-text'>💗 Smart Vision Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Aplikasi Deteksi & Klasifikasi Gambar oleh <b>Yumnaa Alifah</b><br>Mahasiswa Statistika, Universitas Syiah Kuala 🌸</div>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Fungsi pencarian model
# ----------------------------
def find_file_try(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_models():
    yolo_candidates = [
        "model/Yumnaa_Alifah_Laporan_4.pt",
        "Yumnaa_Alifah_Laporan_4.pt",
        "/mnt/data/Yumnaa_Alifah_Laporan_4.pt"
    ]
    clf_candidates = [
        "model/Yumnaa_Alifah_Laporan 2.h5",
        "model/Yumnaa_Alifah_Laporan_2.h5",
        "Yumnaa_Alifah_Laporan 2.h5",
        "/mnt/data/Yumnaa_Alifah_Laporan 2.h5"
    ]

    yolo_path = find_file_try(yolo_candidates)
    clf_path = find_file_try(clf_candidates)

    if yolo_path is None:
        raise FileNotFoundError("Model YOLO (.pt) tidak ditemukan.")
    if clf_path is None:
        raise FileNotFoundError("Model classifier (.h5) tidak ditemukan.")

    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(clf_path)
    return yolo_model, classifier

try:
    with st.spinner("💞 Memuat model AI..."):
        yolo_model, classifier = load_models()
        time.sleep(0.5)
    st.success("🌸 Model berhasil dimuat!")
except Exception as e:
    st.error("❌ Gagal memuat model.")
    st.exception(e)
    st.stop()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.markdown("### 🌷 Navigasi")
menu = st.sidebar.radio(
    "Pilih Mode Analisis:",
    ["💎 Deteksi Objek (YOLO)", "♻️ Klasifikasi Sampah"]
)
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.markdown("✨ <b>Smart Vision Dashboard</b><br>Dikembangkan oleh <b>Yumnaa Alifah</b><br>Mahasiswa Statistika USK 💗", unsafe_allow_html=True)

# ----------------------------
# Preprocessing helper
# ----------------------------
def preprocess_for_classifier(pil_img, classifier):
    default_size = (128, 128)
    in_shape = classifier.input_shape
    expects_flat_vector = False
    target_h, target_w = default_size

    if in_shape is not None:
        if len(in_shape) == 4:
            target_h = int(in_shape[1]) if in_shape[1] else 128
            target_w = int(in_shape[2]) if in_shape[2] else 128
        elif len(in_shape) == 2:
            expects_flat_vector = True

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    pil_resized = pil_img.resize((target_w, target_h))
    arr = np.array(pil_resized).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    if expects_flat_vector:
        arr = arr.reshape((1, -1))
    return arr

# ----------------------------
# Main Logic
# ----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)

    if menu == "💎 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        with st.spinner("🚀 Mendeteksi objek..."):
            results = yolo_model(image)
            result_img = results[0].plot()
            st.image(result_img, caption="💖 Hasil Deteksi Objek", use_container_width=True)
        st.success("✨ Deteksi selesai!")

    elif menu == "♻️ Klasifikasi Sampah":
        st.subheader("🌿 Hasil Klasifikasi Gambar")
        with st.spinner("💫 Mengklasifikasikan gambar..."):
            arr = preprocess_for_classifier(image, classifier)
            prediction = classifier.predict(arr)
            class_index = np.argmax(prediction)
            confidence = float(np.max(prediction))
            waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
            predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else f"Kelas {class_index}"

        st.success(f"✅ Prediksi: **{predicted_label}**")
        st.progress(confidence)
        st.caption(f"🎯 Keyakinan model: {confidence:.2%}")
else:
    st.info("⬅️ Unggah gambar di sidebar untuk memulai analisis 💗")

# ----------------------------
# Footer Bio
# ----------------------------
st.markdown("<div class='footer'>💗 Dibuat oleh <b>Yumnaa Alifah</b><br>Mahasiswa Statistika — Universitas Syiah Kuala 🌸</div>", unsafe_allow_html=True)
