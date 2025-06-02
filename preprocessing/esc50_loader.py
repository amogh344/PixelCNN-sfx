import os
import zipfile
import requests

def download_and_extract_esc50(data_dir="data/ESC-50"):
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = os.path.join(data_dir, "ESC-50.zip")
    extract_path = os.path.join(data_dir, "ESC-50-master")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(extract_path):
        print("⬇️  Downloading ESC-50 dataset...")
        with requests.get(url, stream=True) as r:
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        print("✅ Done.")
    else:
        print("✅ ESC-50 dataset already downloaded and extracted.")

