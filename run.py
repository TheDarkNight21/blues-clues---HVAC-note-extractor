import os
import requests
import subprocess
import shutil

def download_tesseract_installer(dest_path):
    # Stable version: v5.3.1 from March 2023
    url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.1.20230401/tesseract-ocr-w64-setup.exe"
    print("Downloading Tesseract installer...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    print("Download complete.")

def install_tesseract(installer_path):
    print("Running silent install...")
    subprocess.run(
        [installer_path, "/S"],
        shell=True,
        check=True
    )
    print("Tesseract installed.")

def main():
    temp_dir = os.environ.get("TEMP", "C:\\Temp")
    os.makedirs(temp_dir, exist_ok=True)
    installer_path = os.path.join(temp_dir, "tesseract-setup.exe")

    try:
        download_tesseract_installer(installer_path)
        install_tesseract(installer_path)
    except Exception as e:
        print(f"Error during installation: {e}")
    finally:
        if os.path.exists(installer_path):
            os.remove(installer_path)

if __name__ == "__main__":
    main()