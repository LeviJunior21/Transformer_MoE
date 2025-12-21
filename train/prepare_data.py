import os
import sys
import gdown
import subprocess


def download_books(download_books_drive: bool, download_books_scraping: bool, repo_dir: str, max_mb: int, seed: int, min_paragraph_length: int,):
    if download_books_drive:
        FOLDER_URL = "https://drive.google.com/drive/folders/14JMIJPWxeW81Bq4hpu3Sgz7Z8Dw9ZC3P"
        DEST_DIR = "./data/processed"
        
        os.makedirs(DEST_DIR, exist_ok=True)
        print("Baixando dados do Google Drive...")
        gdown.download_folder(url=FOLDER_URL, output=DEST_DIR, quiet=False)

    if download_books_scraping:
        print("Baixando dados via scraping...")
        download_script = os.path.join("/kaggle/working", repo_dir, "scripts", "download_data.py")
        prepare_script = os.path.join("/kaggle/working", repo_dir, "scripts", "prepare_data.py")

        subprocess.run([sys.executable, download_script, str(max_mb), str(seed)], check=True,)
        subprocess.run([sys.executable, prepare_script, str(min_paragraph_length),], check=True,)

    print("Download finalizado.")