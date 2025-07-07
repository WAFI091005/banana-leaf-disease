import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys
import os

# Load model
model = load_model('model_daun_pisang.h5')

# Label class sesuai urutan class_indices
label_mapping = ['penyakit_panama', 'penyakit_sigatoka', 'sehat']

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File {img_path} tidak ditemukan.")
        return

    # Load dan preprocessing gambar
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = label_mapping[predicted_index]
    confidence = predictions[predicted_index] * 100

    print(f"✅ Prediksi: {predicted_label} ({confidence:.2f}%)")

    # Buat plot dua bagian: kiri = gambar, kanan = grafik confidence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Tampilkan gambar
    ax1.imshow(img)
    ax1.set_title("Gambar Input", fontsize=14)
    ax1.axis('off')

    # Tampilkan hasil prediksi dalam bentuk bar chart
    ax2.barh(label_mapping, predictions * 100, color='skyblue')
    ax2.set_xlim([0, 100])
    ax2.set_title("Confidence per Kelas", fontsize=14)
    ax2.set_xlabel("Confidence (%)")
    for i, v in enumerate(predictions * 100):
        ax2.text(v + 1, i, f"{v:.2f}%", va='center', fontsize=10)
    plt.suptitle(f"Hasil Prediksi: {predicted_label} ({confidence:.2f}%)", fontsize=16)
    plt.tight_layout()
    plt.show()

# Jalankan dari CLI
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py path_ke_gambar.jpg")
    else:
        img_path = " ".join(sys.argv[1:])
        predict_image(img_path)
