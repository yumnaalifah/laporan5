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
    elif menu == "♻️ Klasifikasi Sampah":
        st.subheader("🌿 Hasil Klasifikasi Gambar")

        with st.spinner("💫 Sedang memproses gambar..."):
            try:
                # --- Info Input Model ---
                in_shape = classifier.input_shape
                st.write("📏 Bentuk input model:", in_shape)

                # --- Resize gambar sesuai model ---
                target_size = (224, 224)
                pil_img = Image.open(uploaded_file).convert("RGB")
                pil_img = pil_img.resize(target_size)
                img_arr = np.array(pil_img).astype(np.float32) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)  # (1, 224, 224, 3)

                # --- Deteksi apakah model butuh input flatten ---
                flat_len = np.prod(img_arr.shape[1:])
                if len(in_shape) == 2 and in_shape[1] == flat_len:
                    st.info("📎 Model ini mengharapkan input yang sudah di-flatten.")
                    img_arr = img_arr.reshape((1, flat_len))
                    st.write("🧩 Bentuk setelah flatten:", img_arr.shape)
                elif len(in_shape) == 2 and in_shape[1] != flat_len:
                    st.warning(f"⚠️ Model butuh {in_shape[1]} fitur, bukan {flat_len}. "
                               f"Disesuaikan otomatis jika memungkinkan.")
                    img_arr = img_arr.reshape((1, in_shape[1]))
                else:
                    st.write("🧠 Model ini menerima input citra 3D:", img_arr.shape)

                # --- Prediksi ---
                prediction = classifier.predict(img_arr)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                # --- Label Kategori ---
                waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
                predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else "Tidak Dikenali"

                # --- Tampilkan Hasil ---
                st.success(f"✅ Jenis Sampah: **{predicted_label}**")
                st.progress(confidence)
                st.caption(f"🎯 Probabilitas: {confidence:.2%}")

            except Exception as e:
                st.error("❌ Terjadi kesalahan saat klasifikasi.")
                st.exception(e)

else:
    st.info("⬅️ Silakan unggah gambar terlebih dahulu di sidebar untuk mulai analisis.")

# ==============================
# 🌸 FOOTER
# ==============================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    🌸 <b>Smart Vision Dashboard</b> © 2025 — Dibuat oleh <b>Yumnaa Alifah</b><br>
    <i>Menggabungkan kecerdasan buatan & keindahan desain.</i>
    </div>
    """,
    unsafe_allow_html=True
)
