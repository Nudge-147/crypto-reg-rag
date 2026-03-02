#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-dev_support/manifests/sg_pdf_download_manifest.csv}"
OUT_DIR="${2:-raw/sg}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Using manifest: $MANIFEST"
echo "Output dir: $OUT_DIR"

# Skip CSV header, then: filename,url,source_domain,document_type,priority,notes
tail -n +2 "$MANIFEST" | while IFS=, read -r filename url _rest; do
  if [[ -z "${filename}" || -z "${url}" ]]; then
    continue
  fi
  dest="$OUT_DIR/$filename"
  echo "Downloading -> $dest"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
done

echo "Done."
