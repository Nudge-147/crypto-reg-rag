#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_benchmark.py - SAC/权威性RAG系统评估框架
功能：
 1. 加载包含 24 个问题的 Golden Test Set (JSON)。
 2. 实现核心评估指标 (DRM, Jurisdiction Accuracy)。
 3. 运行对比评估：SAC (仅向量) vs SAC + Authority (权威性重排序)。
"""

import json
import numpy as np
import collections
import os
import argparse
import csv
from typing import List, Dict
from pathlib import Path
import statistics # 用于求平均数
import importlib.util
import sys

# 核心修复: 动态导入 03 脚本中定义的关键函数和类（文件名以数字开头不能用常规 import）
try:
    module_path = Path(__file__).parent / "03_retrieve_and_qa.py"
    spec = importlib.util.spec_from_file_location("retrieve_and_qa_03", module_path)
    retrieve_mod = importlib.util.module_from_spec(spec)
    sys.modules["retrieve_and_qa_03"] = retrieve_mod
    spec.loader.exec_module(retrieve_mod)  # type: ignore

    load_index_and_meta = retrieve_mod.load_index_and_meta
    retrieve_with_authority = retrieve_mod.retrieve_with_authority
    create_gptsapi_client = retrieve_mod.create_gptsapi_client
    TOP_K = getattr(retrieve_mod, "TOP_K", getattr(retrieve_mod, "DEFAULT_TOP_K", 5))
    AuthorityMatrix = retrieve_mod.AuthorityMatrix
    embed_query = retrieve_mod.embed_query
except Exception as e:
    print(f"❌ 导入 03_retrieve_and_qa.py 失败。请检查文件名和路径。错误: {e}")
    exit(1)


# ====== 1. 配置和初始化 ======

# 环境变量
API_KEY = os.getenv("GPTSAPI_API_KEY", "")
PROXY_URL = os.getenv("PROXY_URL", "")
client, http_client = create_gptsapi_client(API_KEY, PROXY_URL)

# 加载索引
try:
    index, metas = load_index_and_meta(Path("indexes/faiss.index"), Path("indexes/meta.jsonl"))
    print(f"✅ Loaded {index.ntotal} vectors for benchmarking.")
except Exception as e:
    print(f"❌ FATAL: Index loading failed. Check 'indexes/' directory. Error: {e}")
    exit(1)


# 权重配置
DEFAULT_ALPHA = 0.6  
DEFAULT_BETA = 0.3   
DEFAULT_GAMMA = 0.1  


# ====== 2. 核心：引用到文件名的映射表 ======
# 左边是测试集里的 citation，右边是 cleaned_sac/ 下的文件名 ID

CITATION_TO_DOC_ID = {
    # --- US Manual Batch (10题) ---
    "SEC_v_Ripple_2024_Final_Judgment": "SEC_v_Ripple_2024_Final_Judgment",
    "SEC_v_Ripple_2025_06_26": "SEC_v_Ripple_2025_06_26",
    "US_PL119-27_GENIUS_Act_2025": "US_PL119-27_GENIUS_Act_2025",
    "US_BILL_S394_2025": "US_BILL_S394_2025",
    "CLARITY_Act_2025_RCP": "CLARITY_Act_2025_RCP",
    "SEC_v_Ripple_2025_Stipulation_PR47": "SEC_v_Ripple_2025_Stipulation_PR47",
    
    # --- 保留之前的验证集映射 (可选) ---
    "EU_MiCA_Citation": "EU_MiCA_2023",
    "SG_MAS_PSN02_Citation": "SG_MAS_PSN02_AML_CFT"
}

def resolve_doc_id(citation: str) -> str:
    """尝试将引用映射到系统中的 Doc ID"""
    for key, val in CITATION_TO_DOC_ID.items():
        if key in citation:
            return val
    # 回退：如果找不到精确匹配，使用原始 Citation 的清理版本作为 ID
    return citation.replace(" ", "_").replace(".", "").replace(",", "").upper()

# ====== 3. 测试集加载 (涉及测试集) ======

def load_golden_dataset(path: str = "tests/batches/test_set_B01_fixed.json") -> List[Dict]:
    """
    [涉及测试集] 加载 JSON 测试集。
    我们使用 fix_test_ids.py 生成的 tests/batches/test_set_B01_fixed.json 文件。
    """
    if not os.path.exists(path):
        print(f"❌ Error: Test set file {path} not found. Please ensure it exists.")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"📂 Loaded {len(raw_data)} test cases for evaluation.")
    return raw_data

# ====== 4. 核心评估指标 (涉及测试集) ======

def compute_drm(retrieved_items: List[Dict], gold_sources: List[Dict]) -> float:
    """
    [涉及测试集] 计算 Document-Level Retrieval Mismatch (DRM).
    DRM = (Top-K 中来自错误文档的 Chunk 数量) / K
    """
    k = len(retrieved_items)
    if k == 0: return 0.0
    
    # 1. 确定正确的 Doc ID 列表
    correct_doc_ids = set()
    for source in gold_sources:
        # 使用 Citation 字段映射到我们的系统 Doc ID
        mapped_id = resolve_doc_id(source.get("citation", ""))
        if mapped_id:
            correct_doc_ids.add(mapped_id)
            
    if not correct_doc_ids:
        print("⚠️ Case has no valid Gold Doc ID mapping, skipping DRM for this case.")
        return 0.0 # 无法计算

    # 2. 计算不匹配数量
    mismatches = 0
    for item in retrieved_items:
        retrieved_doc_id = item['chunk_meta']['doc_id']
        if retrieved_doc_id not in correct_doc_ids:
            mismatches += 1
            
    return mismatches / k

def compute_jurisdiction_accuracy(retrieved_items: List[Dict], target_jurisdictions: List[str]) -> float:
    """
    [涉及测试集] 法域正确性：检索结果中有多少比例来自目标法域。
    """
    k = len(retrieved_items)
    if k == 0: return 0.0
    
    matches = 0
    for item in retrieved_items:
        if item['chunk_meta']['jurisdiction'] in target_jurisdictions:
            matches += 1
            
    return matches / k

# (Char P/R 逻辑因依赖精确字节偏移量而省略，专注于 DRM 和 Jur Acc)

# ====== 5. 变体运行器 (Executor) ======

def run_system_variant(name: str, test_set: List[Dict], alpha, beta, gamma):
    """
    [涉及测试集] 遍历测试集并运行指定变体的评估。
    """
    print(f"\n--- 🧪 Running Variant: {name} (α={alpha}, β={beta}, γ={gamma}) ---")
    all_results = collections.defaultdict(list)
    authority_matrix = AuthorityMatrix()
    case_rows = []

    for case in test_set:
        query = case["question_text"]
        
        # 运行检索
        try:
            # retrieve_with_authority 需要 AuthorityMatrix 实例
            retrieved = retrieve_with_authority(
                query, client, index, metas, TOP_K, authority_matrix,
                alpha=alpha, beta=beta, gamma=gamma 
            )
        except Exception as e:
            print(f"⚠️ Retrieval error for q_id {case['id']}: {e}")
            continue
        
        # 计算指标 (所有指标都使用了测试集数据)
        drm = compute_drm(retrieved, case.get("gold_sources", []))
        jur_acc = compute_jurisdiction_accuracy(retrieved, case.get("target_jurisdictions", []))

        # 调试输出当前问题的检索情况
        print(f"\n🔍 Debug Q: {case['question_text'][:30]}...")
        print(f"   Target Doc: {[s['citation'] for s in case.get('gold_sources', [])]}")
        print("   Retrieved Top 5:")
        for r in retrieved:
            meta = r['chunk_meta']
            print(f"     -> [{meta['jurisdiction']}] {meta['doc_id']} (Score: {r['final_score']:.4f})")
        
        all_results["drm"].append(drm)
        all_results["jur_acc"].append(jur_acc)
        top1_doc = retrieved[0]["chunk_meta"]["doc_id"] if retrieved else ""
        top1_jur = retrieved[0]["chunk_meta"]["jurisdiction"] if retrieved else ""
        case_rows.append(
            {
                "variant": name,
                "case_id": case.get("id", ""),
                "question_text": query,
                "target_jurisdictions": ",".join(case.get("target_jurisdictions", [])),
                "drm": f"{drm:.6f}",
                "jur_acc": f"{jur_acc:.6f}",
                "top1_doc_id": top1_doc,
                "top1_jurisdiction": top1_jur,
            }
        )
        
        # (Debug info)
        # print(f"  Q: {case['id']} | DRM: {drm:.4f} | J_Acc: {jur_acc:.4f} | Top Juri: {retrieved[0]['chunk_meta']['jurisdiction']}")


    # 聚合结果
    avg_results = {k: statistics.mean(v) for k, v in all_results.items() if v}
    print(f"✅ Aggregate Results for {name}:")
    print(json.dumps(avg_results, indent=2))
    return avg_results, case_rows


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"💾 CSV exported: {path}")

# ====== 6. 主程序入口 ======

def main():
    parser = argparse.ArgumentParser(description="Run SAC/authority benchmark.")
    parser.add_argument(
        "--test-set",
        default="tests/manual/test_set_us_manual.json",
        help="Path to benchmark dataset JSON.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional per-case CSV output path.",
    )
    parser.add_argument(
        "--out-summary-csv",
        default="",
        help="Optional summary CSV output path.",
    )
    args = parser.parse_args()

    # 加载指定测试集
    test_data = load_golden_dataset(args.test_set)
    
    if not test_data: return

    all_case_rows = []
    summary_rows = []

    # 变体 1: RAG_SAC_Multi (SAC 基础效果，仅依赖向量相似度)
    agg1, rows1 = run_system_variant(
        name="RAG_SAC_Multi (Vector Sim Only)", 
        test_set=test_data, 
        alpha=1.0, beta=0.0, gamma=0.0
    )
    all_case_rows.extend(rows1)
    summary_rows.append(
        {
            "variant": "RAG_SAC_Multi (Vector Sim Only)",
            "avg_drm": f"{agg1.get('drm', 0.0):.6f}",
            "avg_jur_acc": f"{agg1.get('jur_acc', 0.0):.6f}",
            "cases": len(rows1),
        }
    )

    # 变体 2: RAG_SAC_Auth (启用权威性重排序)
    agg2, rows2 = run_system_variant(
        name="RAG_SAC_Auth (Full System)", 
        test_set=test_data, 
        alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, gamma=DEFAULT_GAMMA
    )
    all_case_rows.extend(rows2)
    summary_rows.append(
        {
            "variant": "RAG_SAC_Auth (Full System)",
            "avg_drm": f"{agg2.get('drm', 0.0):.6f}",
            "avg_jur_acc": f"{agg2.get('jur_acc', 0.0):.6f}",
            "cases": len(rows2),
        }
    )

    if args.out_csv:
        write_csv(
            Path(args.out_csv),
            all_case_rows,
            [
                "variant",
                "case_id",
                "question_text",
                "target_jurisdictions",
                "drm",
                "jur_acc",
                "top1_doc_id",
                "top1_jurisdiction",
            ],
        )
    if args.out_summary_csv:
        write_csv(
            Path(args.out_summary_csv),
            summary_rows,
            ["variant", "avg_drm", "avg_jur_acc", "cases"],
        )
    
    # ... (可以添加其他权重变体进行对比)
    print("\n--- Benchmark Complete ---")

if __name__ == "__main__":
    main()
