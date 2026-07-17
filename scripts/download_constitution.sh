#!/usr/bin/env bash
# Download the official Constitution of India PDF (public domain).
set -euo pipefail
URL="https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf"
OUT="${1:-constitution_of_india.pdf}"
echo "Downloading Constitution of India -> ${OUT}"
curl -L --fail "${URL}" -o "${OUT}"
echo "Done. Ingest it with:  python -m samvidhan.ingest --pdf ${OUT}"
