# dashboard.py
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os

st.set_page_config(page_title="💗 Smart Vision Dashboard", layout="wide", page_icon="🌸")

# ----------------------
# Utility: cari file model
# ----------------------
def find_file_try(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ----------------------
# Load models (dengan fallback path)
# ----------------------
@st.cache_resource
def load_models():
    # paths yang dicoba untuk YOLO dan classifier
    yolo_candidates = [
        "model/Yumnaa_Alifah_Laporan_4.pt",
        "Yumnaa_Alifah_Laporan_4.pt",
        "/mnt/data/Yumnaa_Alifah_Laporan_4.pt"
    ]
    clf_candidates = [
        "model/Yumnaa_Alifah_Laporan 2.h5",
        "model/Yumnaa_Alifah_Laporan_2.h5",
        "Yumnaa_Alifah_Laporan 2.h5",
        "classifier_model.h5",
        "/mnt/data/Yumnaa_Alifah_Laporan 2.h5"
    ]

    yolo_path = find_file_try(yolo_candidates)
    clf_path = find_file_try(clf_candidates)

    if yolo_path is None:
        raise FileNotFoundError(f"File YOLO (.pt) tidak ditemukan. Dicari di: {yolo_candidates}")
    if clf_path is None:
        raise FileNotFoundError(f"File classifier (.h5) tidak ditemukan. Dicari di: {clf_candidates}")

    # load
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(clf_path)

    return yolo_model, classifier

# Try load models and tampilkan status
try:
    with st.spinner("💞 Memuat model..."):
        yolo_model, classifier = load_models()
        time.sleep(0.5)
    st.success("🌸 Model berhasil dimuat!")
except Exception as e:
    st.error("❌ Gagal memuat model. Pastikan file .pt dan .h5 ada di folder yang benar.")
    st.exception(e)
    st.stop()

# ----------------------
# Sidebar dan upload
# ----------------------
st.sidebar.title("🌟 Menu Dashboard")
menu = st.sidebar.radio("Pilih Mode:", ["💎 Deteksi Objek (YOLO)", "♻️ Klasifikasi Sampah"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Header
st.title("📸 Aplikasi Deteksi & Klasifikasi Gambar")
st.markdown("Gunakan AI untuk mendeteksi objek (mobile/supercar/laptop) dan mengklasifikasikan sampah.")

# ----------------------
# Fungsi bantu preprocessing otomatis
# ----------------------
def preprocess_for_classifier(pil_img, classifier):
    """
    Preprocess gambar PIL supaya cocok dengan classifier.
    - ambil classifier.input_shape kalau tersedia
    - fallback ke (128,128)
    - konversi ke RGB, resize, normalisasi, expand dims
    """
    # default target
    default_size = (128, 128)

    in_shape = classifier.input_shape  # e.g. (None, 128, 128, 3) or (None, 9216)
    # tentukan apakah model mengharapkan citra 3D atau vector 2D
    expects_flat_vector = False
    target_h, target_w = default_size

    if in_shape is not None:
        try:
            if len(in_shape) == 4:
                # (None, H, W, C)
                target_h = int(in_shape[1]) if in_shape[1] is not None else default_size[0]
                target_w = int(in_shape[2]) if in_shape[2] is not None else default_size[1]
            elif len(in_shape) == 2:
                # (None, N) -> model mungkin mengharapkan flattened input
                expects_flat_vector = True
                # tetap gunakan default size (128,128) untuk membuat fitur
            else:
                target_h, target_w = default_size
        except Exception:
            target_h, target_w = default_size

    # pastikan RGB
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    if pil_img.mode == "L":
        pil_img = pil_img.convert("RGB")

    # resize
    pil_resized = pil_img.resize((target_w, target_h))

    arr = np.array(pil_resized).astype(np.float32) / 255.0  # (H, W, C)

    # handle kemungkinan channel mismatch
    if arr.ndim == 2:
        arr = np.stack((arr,)*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    # expand batch
    arr_batch = np.expand_dims(arr, axis=0)  # (1, H, W, C)

    # jika model mengharapkan flat vector panjang N, ubah shape
    if expects_flat_vector:
        N = int(in_shape[1])
        flat_len = int(np.prod(arr_batch.shape[1:]))
        if flat_len == N:
            arr_batch = arr_batch.reshape((1, flat_len))
        else:
            # jika berbeda, coba resize via reshape (risky) atau pad/truncate
            # kita lakukan simple flatten and if length mismatch -> pad/truncate
            flat = arr_batch.reshape((1, flat_len))
            if flat_len > N:
                arr_batch = flat[:, :N]
            else:
                # pad zeros
                pad_width = N - flat_len
                arr_batch = np.concatenate([flat, np.zeros((1, pad_width), dtype=np.float32)], axis=1)

    return arr_batch, (target_h, target_w), expects_flat_vector, in_shape

# ----------------------
# Main app logic
# ----------------------
if uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file)
    except Exception as e:
        st.error("❌ Gagal membuka file gambar. Pastikan file valid.")
        st.exception(e)
        st.stop()

    st.image(pil_img, caption="🖼️ Gambar yang diupload", use_container_width=True)

    if menu == "💎 Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        try:
            with st.spinner("🚀 Model YOLO sedang mendeteksi..."):
                results = yolo_model(pil_img)
                result_img = results[0].plot()
                st.image(result_img, caption="💖 Hasil Deteksi Objek", use_container_width=True)
            st.success("✨ Deteksi selesai!")
        except Exception as e:
            st.error("❌ Terjadi kesalahan saat deteksi YOLO.")
            st.exception(e)

    elif menu == "♻️ Klasifikasi Sampah":
        st.subheader("🌿 Hasil Klasifikasi Gambar")
        try:
            with st.spinner("💫 Memproses gambar untuk klasifikasi..."):
                arr_batch, target_size, expects_flat, in_shape = preprocess_for_classifier(pil_img, classifier)

                # tampilkan diagnostik
                st.write("🔹 Input model (classifier.input_shape):", in_shape)
                st.write(f"🔹 Gambar di-resize ke: {target_size} (HxW)")
                st.write("🔹 Bentuk yang dikirim ke model:", arr_batch.shape, "dtype:", arr_batch.dtype)
                if expects_flat:
                    st.info("📎 Model mengharapkan input vektor (flatten). Kami telah menyesuaikan input.")

                # prediksi
                prediction = classifier.predict(arr_batch)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                # labels - sesuaikan sesuai urutan keluaran model
                waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
                predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else f"Kelas {class_index}"

                st.success(f"✅ Hasil: **{predicted_label}**")
                st.progress(confidence)
                st.caption(f"🎯 Probabilitas: {confidence:.2%}")

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat klasifikasi.")
            st.exception(e)

else:
    st.info("⬅️ Unggah gambar di sidebar untuk memulai analisis.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:14px;'>🌸 Smart Vision Dashboard © 2025 — dibuat oleh Yumnaa Alifah</div>",
    unsafe_allow_html=True
)
