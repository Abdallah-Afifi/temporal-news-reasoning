from huggingface_hub import snapshot_download
import subprocess
import os

# Output directories (relative to repo root)
CNN_DM_DIR = "./data/corpus/cnn_dailymail"
CNN_STORIES_DIR = "./data/corpus/cnn_stories"
os.makedirs(CNN_DM_DIR, exist_ok=True)
os.makedirs(CNN_STORIES_DIR, exist_ok=True)


# 1. Download CNN/DailyMail dataset from HuggingFace (~2.5 GB, 312k articles)
print("=" * 50)
print("[1/2] Downloading CNN/DailyMail (abisee/cnn_dailymail) ...")
print("=" * 50)
snapshot_download(
    repo_id="abisee/cnn_dailymail",
    repo_type="dataset",
    local_dir=CNN_DM_DIR
)
print("\u2713 CNN/DailyMail downloaded.\n")

# 2. Download CNN Stories corpus (raw articles)
print("=" * 50)
print("[2/2] Downloading CNN Stories corpus ...")
print("=" * 50)
# The raw CNN stories are hosted by Kyunghyun Cho (NYU)
subprocess.run([
    "wget", "-c",
    "https://drive.google.com/uc?export=download&confirm=yes&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ",
    "-O", f"{CNN_STORIES_DIR}/cnn_stories.tgz"
], check=True)
print("✓ CNN Stories corpus downloaded.\n")

print("=" * 50)
print("CNN datasets downloaded successfully!")
print("=" * 50)
