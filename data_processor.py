"""
data_processor.py
-----------------

Extracting 45,000 Amazon reviews (11,250 per category) from raw .json.gz files,
saving them as raw-dataset.csv, applying enhanced cleaning, and saving the
cleaned result as cleaned-dataset.csv.

Running it once before opening the notebook:
    python data_processor.py
"""

import os
import re
import json
import gzip
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

#─── Configuration ────────────────────────────────────────────────────────────
SEED=42
random.seed(SEED)

BASE_DIR     =Path(".")
DATASET_DIR=BASE_DIR / "Dataset" / "Dataset"
RAW_CSV     =BASE_DIR / "raw-dataset.csv"
CLEANED_CSV =BASE_DIR / "cleaned-dataset.csv"

CATEGORIES={
    "Beauty":      "beauty.json.gz",
    "Cell_Phones": "cellphones.json.gz",
    "Electronics": "electronics.json.gz",
    "Sports":      "sports.json.gz",
}
SAMPLES_PER_CAT=11_250   #4 × 11,250 =45,000 total
MIN_WORDS       =15       #Minimum word count after cleaning

#─── Contraction map ──────────────────────────────────────────────────────────
CONTRACTIONS={
    "can't": "cannot","won't": "will not","n't": " not",
    "'re": " are","'s": " is","'d": " would","'ll": " will",
    "'t": " not","'ve": " have","'m": " am",
}

def expand_contractions(text: str) -> str:
    """Expanding common English contractions."""
    for contraction, expansion in CONTRACTIONS.items():
        text =text.replace(contraction,expansion)
    return text


def clean_text(text: str) -> str:
    """
    Enhanced text cleaning pipeline:
      1. Lowercase
      2. Removing HTML tags   (<br />,<p>,...)
      3. Removing URLs        (http/https/www)
      4. Expanding contractions
      5. Removing remaining non-alphabetic characters
      6. Collapsing whitespace
    """
    #1. Lowercase
    text=text.lower()

    #2. Removing HTML tags
    text=re.sub(r"<[^>]+>", " ", text)

    #3. Removing URLs
    text=re.sub(r"https?://\S+|www\.\S+", " ",text)

    #4. Expanding contractions
    text=expand_contractions(text)

    #5. Removing everything except letters and spaces
    text=re.sub(r"[^a-z\s]", " ", text)

    #6. Collapsing multiple whitespace into a single space and strip
    text=re.sub(r"\s+", " ",text).strip()
    return text


def load_raw(base_path: Path,samples_per_cat: int =SAMPLES_PER_CAT):
    """
    Loading raw reviews from .json.gz files.
    Returns a list of dicts with keys: text, rating, sentiment, category.
    """
    records=[]
    for cat_name,file_name in CATEGORIES.items():
        file_path=base_path / file_name
        if not file_path.exists():
            print(f"[WARNING] {file_path} not found – skipping.")
            continue

        count=0
        with gzip.open(file_path,"rt",encoding="utf-8") as f:
            for line in tqdm(f,desc=f"Loading {cat_name}", leave=False):
                if count >=samples_per_cat:
                    break
                try:
                    item     =json.loads(line)
                    raw_text =item.get("reviewText","").strip()
                    rating   =float(item.get("overall", 3.0))
                    if not raw_text:
                        continue

                    sentiment =0 if rating <=2 else (1 if rating ==3 else 2)
                    records.append({
                        "text":      raw_text,
                        "rating":    rating,
                        "sentiment": sentiment,
                        "category":  cat_name,
                    })
                    count +=1
                except Exception:
                    continue

        print(f"  Loaded {count:,} raw samples from {cat_name}")
    return records


def derive_features(word_count: int) -> int:
    """Map word count to a 3-class length feature (same as notebook)."""
    if word_count <40:
        return 0
    elif word_count <100:
        return 1
    else:
        return 2


def main():
    print("=" *60)
    print("Step 1/3 – Loading raw data …")
    print("=" *60)
    records=load_raw(DATASET_DIR, SAMPLES_PER_CAT)
    print(f"\nTotal raw records: {len(records):,}\n")

    #── Saving raw CSV ────────────────────────────────────────────────────────
    print("=" *60)
    print("Step 2/3 – Saving raw-dataset.csv …")
    print("=" *60)
    raw_df=pd.DataFrame(records)
    raw_df.to_csv(RAW_CSV, index=False)
    print(f"  Saved {len(raw_df):,} rows → {RAW_CSV}\n")

    #── Cleaning and filtering ────────────────────────────────────────────────────
    print("=" *60)
    print("Step 3/3 – Cleaning & saving cleaned-dataset.csv …")
    print("=" *60)

    cleaned_rows=[]
    dropped=0
    for row in tqdm(records, desc="Cleaning",unit="row"):
        cleaned =clean_text(row["text"])
        wc=len(cleaned.split())
        if wc <MIN_WORDS:
            dropped+=1
            continue
        cleaned_rows.append({
            "text":      cleaned,
            "rating":    row["rating"],
            "sentiment": row["sentiment"],
            "len_feat":  derive_features(wc),
            "category":  row["category"],
        })
    cleaned_df=pd.DataFrame(cleaned_rows)
    cleaned_df.to_csv(CLEANED_CSV, index=False)

    print(f"\n  Rows dropped (< {MIN_WORDS} words after cleaning): {dropped:,}")
    print(f"Final cleaned rows: {len(cleaned_df):,}")
    print(f"Saved → {CLEANED_CSV}")
    print("\nDone! ✓")

if __name__ =="__main__":
    main()
