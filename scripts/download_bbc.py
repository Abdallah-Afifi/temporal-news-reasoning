from huggingface_hub import snapshot_download
import subprocess
import os

BBC_DIR = "/home/g2/temporal-news-reasoning/data/corpus/bbc"
os.makedirs(BBC_DIR, exist_ok=True)

# 1. Download BBC dataset from HuggingFace 
print("=" * 50)
print("Downloading BBC (RealTimeData/bbc_news_alltime) ...")
print("=" * 50)
snapshot_download(
    repo_id="RealTimeData/bbc_news_alltime",
    repo_type="dataset",
    local_dir=BBC_DIR
)
print("\u2713 BBC downloaded.\n")