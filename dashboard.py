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
            # --- INFO MODEL (diagnostik) ---
            st.write("🔎 INFO MODEL:")
            try:
                st.write("classifier.input_shape:", classifier.input_shape)
                # tampilkan nama input / tensor info jika ada
                if hasattr(classifier, "inputs"):
                    st.write("classifier.inputs:", classifier.inputs)
                if hasattr(classifier, "input_names"):
                    st.write("classifier.input_names:", classifier.input_names)
            except Exception as e_info:
                st.write("Gagal baca beberapa atribut model:", e_info)

            # (opsional) ringkas summary (bisa panjang)
            try:
                with st.expander("Tampilkan model.summary()"):
                    buf = []
                    classifier.summary(print_fn=lambda x: buf.append(x))
                    st.text("\n".join(buf))
            except Exception:
                st.write("Tidak dapat menampilkan model.summary()")

            # --- TENTUKAN TARGET SIZE & LAYOUT CHANNEL ---
            in_shape = classifier.input_shape  # misal (None, H, W, C) atau (None, C, H, W)
            if in_shape is None:
                # fallback aman
                target_size = (224, 224)
                channels_first = False
            else:
                # pastikan tuple dan len >= 4
                if len(in_shape) == 4:
                    # cek apakah format (None, H, W, C) or (None, C, H, W)
                    # jika elemen terakhir > 4 kemungkinan channels_last
                    last = in_shape[-1]
                    second = in_shape[1]
                    # asumsi: bila second==3 -> channels_first; bila last==3 -> channels_last
                    if second == 3 and last != 3:
                        channels_first = True
                        channels_last = False
                        target_h, target_w = int(in_shape[2]), int(in_shape[3])
                    else:
                        channels_first = False
                        channels_last = True
                        target_h, target_w = int(in_shape[1]), int(in_shape[2])
                    target_size = (target_w, target_h)
                elif len(in_shape) == 3:  # misal (None, H, W) atau (None, C, H)
                    channels_first = False
                    target_h, target_w = int(in_shape[1]), int(in_shape[2])
                    target_size = (target_w, target_h)
                else:
                    # fallback
                    target_size = (224, 224)
                    channels_first = False

            st.write(f"🔹 target_size (w,h): {target_size}, channels_first: {channels_first}")

            # --- PREPROCESSING GAMBAR OTOMATIS ---
            # buka ulang uploaded_file (pastikan variable uploaded_file tersedia)
            pil_img = Image.open(uploaded_file)

            # handle transparency and convert to RGB initially
            if pil_img.mode == "RGBA":
                pil_img = pil_img.convert("RGB")
            if pil_img.mode == "L":
                # grayscale -> convert to RGB (kebanyakan model butuh 3 channel)
                pil_img = pil_img.convert("RGB")

            # resize sesuai target_size
            pil_resized = pil_img.resize(target_size)

            # ubah ke numpy
            img_arr = np.array(pil_resized).astype(np.float32) / 255.0  # (H, W, C) channels_last

            # Pastikan 3 channel
            if img_arr.ndim == 2:
                img_arr = np.stack((img_arr,)*3, axis=-1)
            if img_arr.shape[-1] == 4:
                img_arr = img_arr[..., :3]

            # convert layout jika model menggunakan channels_first
            if channels_first:
                # model expects (1, C, H, W)
                img_arr = np.transpose(img_arr, (2, 0, 1))  # -> (C, H, W)
                img_arr = np.expand_dims(img_arr, axis=0)   # -> (1, C, H, W)
            else:
                # model expects (1, H, W, C)
                img_arr = np.expand_dims(img_arr, axis=0)   # -> (1, H, W, C)

            # paksa dtype float32
            img_arr = img_arr.astype(np.float32)

            # Tampilkan shapes untuk debugging
            st.write("🧩 Shape gambar yang dikirim ke model:", img_arr.shape)
            st.write("🧾 Tipe data gambar:", img_arr.dtype)

            # --- PREDIKSI ---
            prediction = classifier.predict(img_arr)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            # --- LABEL KATEGORI SAMPAH ---
            waste_labels = ["Kaca", "Kardus", "Kertas", "Plastik", "Logam", "Residu"]
            predicted_label = waste_labels[class_index] if class_index < len(waste_labels) else "Tidak Dikenali"

            # --- OUTPUT ---
            st.success(f"✅ Hasil Klasifikasi: **{predicted_label}**")
            st.write(f"📊 Tingkat keyakinan: **{confidence:.2%}**")

        except ValueError as ve:
            # Tangkap ValueError dari Keras dan tampilkan info lengkap
            st.error("❌ Terjadi ValueError saat memanggil model.predict() — kemungkinan mismatch input.shape atau channels.")
            st.exception(ve)
            st.info("Cek informasi model dan shape gambar di atas. Jika ingin, salin output 'classifier.input_shape' dan 'Shape gambar yang dikirim ke model' lalu kirim ke saya agar saya bantu sesuaikan.")
        except Exception as e:
            st.error("❌ Terjadi kesalahan lain saat memproses.")
            st.exception(e)
