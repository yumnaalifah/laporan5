import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# 🧠 LOAD MODEL
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Yumnaa Alifah_Laporan 4.pt")  # model deteksi YOLO
    classifier = tf.keras.models.load_model("classifier_model.h5")  # model klasifikasi
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

   # ====== BAGIAN KLASIFIKASI (GANTI BAGIAN LAMA DENGAN INI) ======
elif menu == "♻️ Klasifikasi Sampah":
    st.subheader("🌿 Hasil Klasifikasi Gambar")

    with st.spinner("💫 Sedang memproses gambar..."):
        try:
            in_shape = classifier.input_shape
            st.write("classifier.input_shape:", in_shape)

            # Target ukuran default (224x224)
            target_size = (224, 224)
            pil_img = Image.open(uploaded_file).convert("RGB")
            pil_img = pil_img.resize(target_size)

            img_arr = np.array(pil_img).astype(np.float32) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)  # (1, 224, 224, 3)

            # --- Deteksi apakah model butuh input flatten ---
            flat_len = np.prod(img_arr.shape[1:])
            if in_shape[-1] == 9216 or (len(in_shape) == 2 and in_shape[1] == 9216):
                st.info("Model ini mengharapkan input yang sudah di-flatten.")
                img_arr = img_arr.reshape((1, flat_len))
                st.write("🧩 Shape gambar setelah flatten:", img_arr.shape)
            else:
                st.write("Model ini menerima input citra 3D:", img_arr.shape)

            # --- Prediksi ---
            prediction = classifier.predict(img_arr)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
            predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else "Tidak Dikenali"

            st.success(f"✅ Hasil Klasifikasi: **{predicted_label}**")
            st.progress(confidence)
            st.caption(f"Probabilitas: {confidence:.2%}")

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat klasifikasi.")
            st.exception(e)

