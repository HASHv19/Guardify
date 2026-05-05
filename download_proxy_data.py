import pandas as pd
import urllib.request
import os

print("Downloading dataset...")
url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
df = pd.read_csv(url)

print("Formatting dataset...")
# Class 0: hate speech, Class 1: offensive language, Class 2: neither
# Map to HASOC format: 0/1 -> HOF, 2 -> NOT
def map_label(c):
    return "HOF" if c in [0, 1] else "NOT"

df["task_1"] = df["class"].apply(map_label)
df["text"] = df["tweet"]

output_path = "data/external/hasoc_proxy.csv"
os.makedirs("data/external", exist_ok=True)
df[["text", "task_1"]].to_csv(output_path, index=False)

print(f"Dataset saved to {output_path} with {len(df)} rows.")
