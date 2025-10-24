import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="💗 Smart Vision Dashboard",
    layout="wide",
    page_icon="🌸"
)

# ==========================
# CUSTOM CSS (Tema Pink)
# ==========================
st.markdown(
    """
    <style>
    /* Background & layout */
    .main {
        background: linear-gradient(180deg, #ffe6f2 0%, #fff5fa 100%);
        color: #4a154b;
        font-family: 'Poppins', sans-serif;
    }
    /* Judul besar */
    h1 {
        text-align: center;
        color: #e75480;
        font-weight: 800 !important;
        text-shadow: 1px 1px 2px #ffd6e7;
    }
    /* Subjudul */
    h2, h3 {
        color: #d63384 !important;
    }
    /* Tombol */
    .stButton>button {
        background-color: #ff80aa;
        color: white;
        border-radius: 12px;
        border: none;
        height: 3em;
        font-weight: bold;
        width: 100%;
        box-shadow: 0px 3px 6px rgba(255, 182, 193, 0.6);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4da6;
        transform: scale(1.05);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffd6e7 0%, #fff0f6 100%);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #d63384 !important;
    }
    /* Gaya progress bar */
    .stProgress > div > div > div {
        background-color: #ff66b2 !important;
    }
    /* Footer */
    footer {
        text-align: center;
        color: gray;
        padding-top: 15px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Yumnaa_Alifah_Laporan_4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/Yumnaa_Alifah_Laporan 2.h5")  # Model Klasifikasi Sampah
    return yolo_model, classifier

with st.spinner("💞 Sedang memuat model, mohon tunggu..."):
    yolo_model, classifier = load_models()
    time.sleep(1)
st.success("🌸 Model berhasil dimuat!")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.markdown("Pilih mode yang ingin dijalankan:")
menu = st.sidebar.radio("Mode:", ["💎 Deteksi Objek (YOLO)", "♻️ Klasifikasi Sampah"])
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# KONTEN UTAMA
# ==========================
st.title("💗 Smart Vision Dashboard")
st.markdown("<h3 style='text-align:center;'>Aplikasi Deteksi & Klasifikasi Citra oleh <b>Yumnaa Alifah</b></h3>", unsafe_allow_html=True)
st.write("Gunakan AI untuk mengenali objek dan mengklasifikasikan jenis sampah dengan cepat dan interaktif ✨")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    if menu == "💎 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        with st.spinner("🚀 Model sedang mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="💖 Hasil Deteksi Objek", use_container_width=True)

        st.success("✨ Deteksi selesai!")
        st.info("Model ini dapat mendeteksi kategori seperti **mobile**, **supercar**, dan **laptop**.")

    elif menu == "♻️ Klasifikasi Sampah":
    st.subheader("🌿 Hasil Klasifikasi Gambar")
    with st.spinner("💫 Sedang memproses gambar..."):
        # ====== Periksa input shape model ======
        input_shape = classifier.input_shape[1:4]  # contoh (224,224,3)
        st.write("🔹 Input shape model:", input_shape)

        # ====== Preprocessing otomatis menyesuaikan model ======
        color_mode = 'RGB' if input_shape[2] == 3 else 'L'  # RGB atau grayscale
        img = Image.open(uploaded_file).convert(color_mode)
        img = img.resize((input_shape[0], input_shape[1]))
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Tambahkan channel jika grayscale
        if input_shape[2] == 1:
            img_array = np.expand_dims(img_array, axis=-1)

        img_array = np.expand_dims(img_array, axis=0)  # jadi (1, H, W, C)

        # ====== Prediksi ======
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

    waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
    predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else "Tidak Dikenali"

    st.write(f"### 🌸 Jenis Sampah: **{predicted_label}**")
    st.progress(float(confidence))
    st.caption(f"Probabilitas: {confidence:.2%}")


        if confidence > 0.85:
            st.success("🌟 Prediksi sangat akurat!")
        elif confidence > 0.60:
            st.warning("💬 Prediksi cukup baik, bisa ditingkatkan dengan dataset tambahan.")
        else:
            st.error("😿 Prediksi rendah — coba gambar lain.")

else:
    st.info("⬅️ Silakan unggah gambar dari sidebar untuk memulai analisis.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    """
    <footer>
        🌸 <b>Smart Vision Dashboard</b> © 2025 — dibuat dengan 💗 oleh <b>Yumnaa Alifah</b> <br>
        <i>Menggabungkan kecerdasan buatan & keindahan desain.</i>
    </footer>
    """,
    unsafe_allow_html=True
)

