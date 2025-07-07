import os
import shutil

# Path baru ke folder sebenarnya yang berisi data
# Path ke folder sumber asli (perhatikan ada dua 'Original Images')
source_folder = "dataset_raw/Banana Disease Recognition Dataset/Original Images/Original Images"
target_folder = "dataset_daun_pisang"

rename_map = {
    "Banana Healthy Leaf": "sehat",
    "Banana Black Sigatoka Disease": "penyakit_sigatoka",
    "Banana Panama Disease": "penyakit_panama"
}



# Bersihkan folder target
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
os.makedirs(target_folder)

print("📥 Menyalin folder yang dibutuhkan...")

# Salin dan rename folder
for folder_name, new_name in rename_map.items():
    source_path = os.path.join(source_folder, folder_name)
    target_path = os.path.join(target_folder, new_name)

    if os.path.exists(source_path):
        shutil.copytree(source_path, target_path)
        print(f"✅ Disalin: {folder_name} ➜ {new_name}")
    else:
        print(f"⚠️ Tidak ditemukan: {folder_name}")

print("🎉 Dataset sudah difilter dan dirapikan di folder dataset_daun_pisang")
