#!/usr/bin/env bash
set -u
MANIFEST="${1:-dev_support/manifests/kr_uk_uae_download_manifest.csv}"
ROOT="${2:-raw}"
FAILED_LOG="${3:-dev_support/manifests/kr_uk_uae_download_failed.log}"
[ -f "$MANIFEST" ] || { echo "Manifest not found: $MANIFEST" >&2; exit 1; }
: > "$FAILED_LOG"

while IFS=, read -r jur filename url domain note; do
  [[ "$jur" == "jurisdiction" ]] && continue
  [ -n "$jur" ] || continue
  out_dir="$ROOT/$jur"
  mkdir -p "$out_dir"
  dest="$out_dir/$filename"
  if [[ -f "$dest" ]]; then
    echo "Skip existing: $dest"
    continue
  fi
  echo "Downloading $jur -> $dest"
  if ! curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"; then
    echo "$jur,$filename,$url" >> "$FAILED_LOG"
    rm -f "$dest"
    echo "Failed: $url"
  fi
done < "$MANIFEST"

echo "Done. Failed list: $FAILED_LOG"
