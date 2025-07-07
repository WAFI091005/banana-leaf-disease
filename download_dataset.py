import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "sujaykapadnis/banana-disease-recognition-dataset"
DEST_FOLDER = "dataset_raw"

print("📥 Downloading dataset...")
api = KaggleApi()
api.authenticate()
api.dataset_download_files(DATASET, path=DEST_FOLDER, unzip=True)
print("✅ Download complete.")
