from huggingface_hub import HfApi
import getpass

# 1. Securely ask for your token
print("--- Hugging Face Direct Uploader ---")
token = getpass.getpass("Paste your HF GITHUB_SYNC token (typing will be hidden): ")

# 2. Authenticate
api = HfApi(token=token)

# 3. Blast the files to the cloud
print("\nChunking large .pkl models and uploading to Hugging Face Spaces...")
print("This may take a few minutes depending on your internet upload speed. Please wait...")

api.upload_folder(
    folder_path=".",
    repo_id="pancakecurry/smartphone-ai-backend",
    repo_type="space",
    ignore_patterns=[".git*", "venv*", "__pycache__*", ".DS_Store", "hf_upload.py"]
)

print("\n✅ Upload successful! The Hugging Face server is now building.")