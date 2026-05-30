import os
import gdown
import zipfile

def download_and_extract(file_id, output_path, extract_dir=None):
    print(f"Downloading model file...")
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download file using gdown
    gdown.download(url, output_path, quiet=False)
    
    # Extract if required
    if extract_dir and output_path.endswith('.zip'):
        print(f"Extracting to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed successfully!")

if __name__ == "__main__":
    print("=== MODEL DOWNLOADER FOR SIGN LANGUAGE ASSISTOR ===")
    
    # Download BARTPho Model (Zip)
    zip_file_id = '17aG2Nrn2d2sByZ5y8R3-3SqUOMpkmGil'
    zip_output_path = 'Vietnamese_NLP_BARTPho/bartpho_sentence_model_v4_mega.zip'
    zip_extract_dir = 'Vietnamese_NLP_BARTPho/models/bartpho_sentence'
    
    if zip_file_id != 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE':
        download_and_extract(zip_file_id, zip_output_path, zip_extract_dir)
    else:
        print("\n[WARNING] Please update the Google Drive ID in scripts/download_models.py")

    print("\nProcess finished.")
