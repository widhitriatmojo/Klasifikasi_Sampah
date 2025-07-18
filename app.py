import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ======================================
# Load model & label
# ======================================
model = load_model('model_klasifikasi_sampah.h5')

# Mapping label (sesuai class_indices)
# Contoh class_indices: {'kaca': 0, 'kertas': 1, 'logam': 2, 'organik': 3, 'plastik': 4}
class_names = ['kaca', 'kardus', 'kertas', 'logam', 'plastik', 'residu']

# ======================================
# Fungsi Prediksi
# ======================================
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence

# ======================================
# Streamlit UI
# ======================================
st.title("Deteksi Sampah Daur Ulang")
st.write("Upload gambar atau ambil dari kamera untuk deteksi jenis sampah.")

# Pilihan input: Upload atau Kamera
option = st.radio("Pilih metode input gambar:", ["Upload File", "Kamera"])

if option == "Upload File":
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Gambar yang diupload', use_container_width=True)

        label, conf = predict_image(img)
        st.success(f"Prediksi: **{label.upper()}** ({conf*100:.2f}%)")

elif option == "Kamera":
    camera_img = st.camera_input("Ambil gambar")
    if camera_img is not None:
        img = Image.open(camera_img)
        st.image(img, caption='Gambar dari kamera', use_container_width=True)

        label, conf = predict_image(img)
        st.success(f"Prediksi: **{label.upper()}** ({conf*100:.2f}%)")
