#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "[1/4] GET /jurisdictions"
curl -s "${BASE_URL}/jurisdictions"
echo
echo

echo "[2/4] POST /query HK only"
curl -s -X POST "${BASE_URL}/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"香港稳定币发行监管要求是什么？","target_jurisdictions":["HK"],"mode":"jurisdiction_specific","top_k":5}'
echo
echo

echo "[3/4] POST /query HK+SG mix"
curl -s -X POST "${BASE_URL}/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"比较香港与新加坡稳定币牌照监管差异","target_jurisdictions":["HK","SG"],"mode":"jurisdiction_specific","top_k":6}'
echo
echo

echo "[4/4] POST /query invalid jurisdiction XX (expect 400)"
TMP_BODY="$(mktemp)"
HTTP_CODE="$(curl -s -o "${TMP_BODY}" -w "%{http_code}" -X POST "${BASE_URL}/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"test","target_jurisdictions":["XX"],"mode":"jurisdiction_specific","top_k":5}')"
echo "status=${HTTP_CODE}"
cat "${TMP_BODY}"
rm -f "${TMP_BODY}"
echo
