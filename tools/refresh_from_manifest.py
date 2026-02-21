#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Incremental refresh utility:
1) Process new raw PDFs into cleaned_sac/*.jsonl (SAC pipeline).
2) Append only new chunks into indexes/faiss.index + indexes/meta.jsonl.

Usage:
  python3 tools/refresh_from_manifest.py
  python3 tools/refresh_from_manifest.py --jurisdictions sg,uk,uae
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
CLEANED_DIR = ROOT / "cleaned_sac"
INDEX_DIR = ROOT / "indexes"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def parse_jurisdictions(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def discover_pending_pdfs(jurisdictions: List[str]) -> List[Path]:
    pending = []
    allowed = set(jurisdictions)
    for jur_dir in sorted(RAW_DIR.glob("*")):
        if not jur_dir.is_dir():
            continue
        jur = jur_dir.name.lower()
        if allowed and jur not in allowed:
            continue
        for pdf in sorted(jur_dir.glob("*.pdf")):
            out_jsonl = CLEANED_DIR / f"{pdf.stem}.jsonl"
            if not out_jsonl.exists():
                pending.append(pdf)
    return pending


def load_existing_meta() -> Tuple[List[Dict], set]:
    metas = []
    chunk_ids = set()
    if not META_PATH.exists():
        return metas, chunk_ids
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metas.append(obj)
            cid = str(obj.get("chunk_id", "")).strip()
            if cid:
                chunk_ids.add(cid)
    return metas, chunk_ids


def collect_new_chunks(existing_chunk_ids: set, jurisdictions: List[str]) -> List[Dict]:
    allowed = set([j.upper() for j in jurisdictions if j.strip()])
    new_chunks = []
    for jf in sorted(CLEANED_DIR.glob("*.jsonl")):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj.get("chunk_id", "")).strip()
                if not cid or cid in existing_chunk_ids:
                    continue
                jur = str(obj.get("jurisdiction", "")).strip().upper()
                if allowed and jur not in allowed:
                    continue
                txt = str(obj.get("augmented_text", "")).strip()
                if not txt:
                    continue
                new_chunks.append(obj)
                existing_chunk_ids.add(cid)
    return new_chunks


def append_embeddings(embed_mod, new_chunks: List[Dict]) -> int:
    api_key = os.getenv("GPTSAPI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GPTSAPI_API_KEY")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client, http_client = embed_mod.create_client_with_proxy(embed_mod.PROXY_URL, api_key)
    try:
        texts = [c["augmented_text"] for c in new_chunks]
        batch_size = int(embed_mod.BATCH_SIZE)
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = embed_mod.embed_batch(batch, client)
            all_vecs.append(vecs)
        X = np.vstack(all_vecs).astype("float32")
        X = embed_mod.l2_normalize(X)

        if INDEX_PATH.exists():
            index = faiss.read_index(str(INDEX_PATH))
        else:
            index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, str(INDEX_PATH))

        with open(META_PATH, "a", encoding="utf-8") as f:
            for obj in new_chunks:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        if http_client:
            http_client.close()
    return len(new_chunks)


def main():
    parser = argparse.ArgumentParser(description="Incremental SAC refresh + index append.")
    parser.add_argument(
        "--jurisdictions",
        default="",
        help="Comma-separated jurisdictions, e.g. sg,uk,uae. Empty = all.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip SAC PDF->JSONL generation and only append embeddings.",
    )
    args = parser.parse_args()

    jur_filters = parse_jurisdictions(args.jurisdictions)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    sac_mod = load_module(ROOT / "01_sac_pipeline.py", "sac_pipeline_mod")
    embed_mod = load_module(ROOT / "02_embed_build_openai.py", "embed_build_mod")

    if not args.skip_clean:
        pending = discover_pending_pdfs(jur_filters)
        print(f"Pending PDFs for SAC processing: {len(pending)}")
        if pending:
            client = sac_mod.create_client()
            for pdf in pending:
                print(f"Processing {pdf}")
                sac_mod.process_file_sac(client, pdf)
    else:
        print("Skipping SAC cleaning stage (--skip-clean enabled).")

    _, existing_chunk_ids = load_existing_meta()
    new_chunks = collect_new_chunks(existing_chunk_ids, jur_filters)
    print(f"New chunks to append: {len(new_chunks)}")
    if not new_chunks:
        print("No new chunks found. Done.")
        return

    appended = append_embeddings(embed_mod, new_chunks)
    print(f"Appended {appended} chunks into {INDEX_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()
