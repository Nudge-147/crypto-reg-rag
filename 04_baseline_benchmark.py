#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_baseline_benchmark.py — 评测 Baseline (Naive RAG) 的表现
关键：加载 baseline_data/indexes 下的索引，使用与 SAC 相同的测试集。
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import httpx
import numpy as np
from openai import OpenAI

try:
    from httpx_socks import SyncProxyTransport
except ImportError:
    SyncProxyTransport = None

# ====== 配置 ======
# 1. 指向 Baseline 索引
INDEX_DIR = Path("baseline_data/indexes")
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"

# 2. 使用同一套测试集 (控制变量)
TEST_SET_FILE = Path("test_set_us_manual.json")

# 3. 映射表 (保持与 SAC 评测一致，确保公平)
CITATION_TO_DOC_ID = {
    "SEC_v_Ripple_2024_Final_Judgment": "SEC_v_Ripple_2024_Final_Judgment",
    "SEC_v_Ripple_2025_06_26": "SEC_v_Ripple_2025_06_26",
    "US_PL119-27_GENIUS_Act_2025": "US_PL119-27_GENIUS_Act_2025",
    "US_BILL_S394_2025": "US_BILL_S394_2025",
    "CLARITY_Act_2025_RCP": "CLARITY_Act_2025_RCP",
    "SEC_v_Ripple_2025_Stipulation_PR47": "SEC_v_Ripple_2025_Stipulation_PR47",
    "EU_MiCA_Citation": "EU_MiCA_2023",
    "SG_MAS_PSN02_Citation": "SG_MAS_PSN02_AML_CFT",
}

EMBED_MODEL = "text-embedding-3-large"
TOP_K = 5
GPTS_BASE_URL = os.getenv("GPTS_BASE_URL", "https://api.gptsapi.net/v1")
PROXY_URL = os.getenv("PROXY_URL", "")


def get_client() -> OpenAI:
    api_key = os.getenv("GPTSAPI_API_KEY")
    if not api_key:
        raise ValueError("请设置 GPTSAPI_API_KEY")
    http_client = None
    if PROXY_URL and SyncProxyTransport:
        transport = SyncProxyTransport.from_url(PROXY_URL)
        http_client = httpx.Client(transport=transport)
    return OpenAI(api_key=api_key, base_url=GPTS_BASE_URL, http_client=http_client)


def embed_query(text: str, client: OpenAI) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.reshape(1, -1)


def load_data() -> Tuple[faiss.Index, List[Dict], List[Dict]]:
    index = faiss.read_index(str(INDEX_PATH))
    chunks_meta: List[Dict] = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks_meta.append(json.loads(line))
    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    return index, chunks_meta, test_data


def compute_drm(retrieved_chunks: List[Dict], gold_sources: List[Dict], top_k: int = TOP_K) -> float:
    if not gold_sources:
        return 0.0
    gold_ids = set()
    for gs in gold_sources:
        mapped_id = CITATION_TO_DOC_ID.get(gs.get("citation"))
        if mapped_id:
            gold_ids.add(mapped_id)

    if not gold_ids:
        return 0.0

    wrong_count = 0
    for r in retrieved_chunks[:top_k]:
        if r["chunk_meta"]["doc_id"] not in gold_ids:
            wrong_count += 1
    return wrong_count / top_k


def main():
    print("--- Running Baseline Benchmark (Naive RAG) ---")
    if not INDEX_PATH.exists():
        print(f"索引文件不存在: {INDEX_PATH}。请先运行 01 和 02 脚本。")
        return

    client = get_client()
    index, chunks_meta, test_set = load_data()

    total_drm = 0.0
    print(f"Evaluating {len(test_set)} cases...")

    for case in test_set:
        # 1. Embed Query
        q_vec = embed_query(case["question_text"], client)

        # 2. Search
        D, I = index.search(q_vec, TOP_K)

        retrieved = []
        for rank, idx in enumerate(I[0]):
            if idx < len(chunks_meta):
                retrieved.append(
                    {
                        "chunk_meta": chunks_meta[idx],
                        "score": float(D[0][rank]),
                    }
                )

        # 3. Compute DRM
        drm = compute_drm(retrieved, case.get("gold_sources", []))
        total_drm += drm

        # 调试输出错误案例
        if drm > 0:
            print(f"DRM Fail ({drm}): {case['question_text'][:40]}...")
            print(f"   Target: {[g['citation'] for g in case['gold_sources']]}")
            if retrieved:
                print(f"   Got Top1: {retrieved[0]['chunk_meta']['doc_id']}")

    avg_drm = total_drm / len(test_set) if test_set else 0.0
    print("\nBaseline Results:")
    print(f"   Average DRM: {avg_drm:.4f} (Lower is better)")
    print("   (Compare this with SAC DRM: 0.0455)")


if __name__ == "__main__":
    main()
