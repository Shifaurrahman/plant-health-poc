#!/usr/bin/env bash
# Run this on YOUR machine. The sandbox cannot reach kaggle.com or data.mendeley.com.
# Usage:
#   bash scripts/download_datasets.sh                 # Kaggle only
#   BRINJAL_ZIP_URL="<link>" TEA_ZIP_URL="<link>" bash scripts/download_datasets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/kaggle_new_plant_diseases" "$RAW/kaggle_rice" "$RAW/sri_lanka_brinjal" "$RAW/sri_lanka_tea"

# --- Kaggle (needs `pip install kaggle` and ~/.kaggle/kaggle.json) ---
kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p "$RAW/kaggle_new_plant_diseases" --unzip
kaggle datasets download -d anshulm257/rice-disease-dataset -p "$RAW/kaggle_rice" --unzip

# --- Mendeley: copy the link from the download button, then pass it in ---
# In the browser: right-click the download icon next to SLIF-Brinjal.zip -> "Copy link address"
# Same for the tea dataset's zip. The -C - flag resumes if the 2.35 GB download is interrupted.
if [ -n "${BRINJAL_ZIP_URL:-}" ]; then
  curl -L -C - -o "$RAW/sri_lanka_brinjal/SLIF-Brinjal.zip" "$BRINJAL_ZIP_URL"
  unzip -q -n "$RAW/sri_lanka_brinjal/SLIF-Brinjal.zip" -d "$RAW/sri_lanka_brinjal"
fi
if [ -n "${TEA_ZIP_URL:-}" ]; then
  curl -L -C - -o "$RAW/sri_lanka_tea/tea.zip" "$TEA_ZIP_URL"
  unzip -q -n "$RAW/sri_lanka_tea/tea.zip" -d "$RAW/sri_lanka_tea"
fi
echo "Done. Check $RAW"