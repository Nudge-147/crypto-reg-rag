#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_retrieve_and_qa.py - 权威性感知检索与 RAG QA
功能：
 1. 定义权威性矩阵 (Authority Matrix)。
 2. 实现 score_chunk 多维打分函数 (相似度 + 权威性 + 语言奖励)。
 3. 执行重排序检索 (Re-ranking) 并生成问答。
"""

import os
import json
import time
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
from openai import OpenAI
import openai
import httpx
try:
    from httpx_socks import SyncProxyTransport
except ImportError:
    SyncProxyTransport = None

# ====== 模型与配置 ======
PROXY_URL = os.getenv("PROXY_URL", "")
GPTS_BASE_URL = os.getenv("GPTS_BASE_URL", "https://api.gptsapi.net/v1")

INDEX_PATH = Path("indexes/faiss.index")
META_PATH = Path("indexes/meta.jsonl")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
QA_MODEL = os.getenv("QA_MODEL", "gpt-4o-mini") 
MAX_EMBED_RETRIES = 5
DEFAULT_TOP_K = 5
TOP_K = DEFAULT_TOP_K  # backward compatibility for benchmark scripts
MAX_TOP_K = 20
SUPPORTED_MODES = {"jurisdiction_specific", "deep_research"}


@dataclass
class QueryRequest:
    question: str
    target_jurisdictions: Optional[List[str]]
    mode: str
    top_k: int


@dataclass
class QueryResponse:
    answer: str
    retrieved_items: List[Dict]
    stats: Dict
    applied_jurisdictions: Optional[List[str]]
    warnings: List[str]


# >>>>>>>>>>>>>>>>>> 权威性数据模型 <<<<<<<<<<<<<<<<<<

class AuthorityMatrix:
    """定义不同法域在特定法律主题下的权威性权重 (0.0 - 1.0)"""
    def __init__(self):
        # 示例权威性矩阵：请根据您的研究需求定制
        self.matrix = {
            "data_protection": { # 示例：GDPR
                "EU": 1.0, 
                "US": 0.6,
                "SG": 0.7
            },
            "contract_law": { # 示例：英美判例法
                "US": 1.0,  
                "EU": 0.8,
                "SG": 0.9
            },
            "general": { # 默认 fallback
                "EU": 0.8, "US": 0.8, "SG": 0.8
            }
        }
    
    def get_score(self, topic: str, jurisdiction: str) -> float:
        topic_scores = self.matrix.get(topic, self.matrix["general"])
        # 如果法域不在主题列表中，给一个保守的分数 0.5
        return topic_scores.get(jurisdiction, 0.5) 

def classify_query_topics(query: str) -> List[str]:
    """[Abstract] 模拟 LLM 或关键词匹配进行主题分类"""
    if any(k in query.lower() for k in ["privacy", "gdpr", "data", "个人数据"]):
        return ["data_protection"]
    if any(k in query.lower() for k in ["contract", "agreement", "协议", "违约"]):
        return ["contract_law"]
    return ["general"]

# >>>>>>>>>>>>>>>>>> 初始化与加载 <<<<<<<<<<<<<<<<<<

def create_gptsapi_client(api_key: str, proxy_url: str = None):
    http_client = None
    if proxy_url and proxy_url.strip() != "" and SyncProxyTransport is not None:
        transport = SyncProxyTransport.from_url(proxy_url)
        http_client = httpx.Client(transport=transport, timeout=60.0)
        print(f"✅ 使用代理：{proxy_url}")
    
    client = OpenAI(api_key=api_key, http_client=http_client, base_url=GPTS_BASE_URL)
    return client, http_client

def load_index_and_meta(index_path: Path, meta_path: Path):
    if not index_path.exists(): raise FileNotFoundError(f"未找到向量索引：{index_path}")
    if not meta_path.exists(): raise FileNotFoundError(f"未找到元数据文件：{meta_path}")

    index = faiss.read_index(str(index_path))
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return index, metas


def get_available_jurisdictions(metas: list) -> List[str]:
    vals = set()
    for m in metas:
        j = str(m.get("jurisdiction", "")).strip().upper()
        if j:
            vals.add(j)
    return sorted(vals)


def normalize_query_request(
    raw_request: Dict,
    available: List[str],
    strict: bool = True,
) -> Tuple[QueryRequest, Dict]:
    """
    统一请求协议:
      question: str (required)
      target_jurisdictions: list[str] | str | None
      mode: jurisdiction_specific | deep_research
      top_k: int (1..MAX_TOP_K)
    """
    warnings: List[str] = []
    invalid_jurisdictions: List[str] = []

    question = str(raw_request.get("question", "")).strip()
    if not question:
        raise ValueError("question 不能为空")

    raw_mode = str(raw_request.get("mode", "jurisdiction_specific")).strip().lower()
    if raw_mode in SUPPORTED_MODES:
        mode = raw_mode
    elif strict:
        raise ValueError(
            f"mode 非法: {raw_mode}. 支持值: {sorted(SUPPORTED_MODES)}"
        )
    else:
        mode = "jurisdiction_specific"
        warnings.append(f"mode 非法，已回退为 {mode}: {raw_mode}")

    raw_top_k = raw_request.get("top_k", DEFAULT_TOP_K)
    try:
        top_k = int(raw_top_k)
    except Exception:
        if strict:
            raise ValueError(f"top_k 非法: {raw_top_k}")
        top_k = DEFAULT_TOP_K
        warnings.append(f"top_k 非法，已回退为 {DEFAULT_TOP_K}: {raw_top_k}")

    if top_k < 1 or top_k > MAX_TOP_K:
        if strict:
            raise ValueError(f"top_k 超出范围: {top_k}, 允许范围 1..{MAX_TOP_K}")
        clamped = max(1, min(top_k, MAX_TOP_K))
        warnings.append(f"top_k 超出范围，已调整为 {clamped}: {top_k}")
        top_k = clamped

    raw_targets = raw_request.get("target_jurisdictions")
    if isinstance(raw_targets, str):
        available_set = set(available)
        dedup = []
        for token in raw_targets.split(","):
            v = token.strip().upper()
            if not v or v in dedup:
                continue
            dedup.append(v)
        targets = [v for v in dedup if v in available_set]
        invalid_jurisdictions = [v for v in dedup if v not in available_set]
    elif isinstance(raw_targets, list):
        available_set = set(available)
        dedup = []
        for item in raw_targets:
            v = str(item).strip().upper()
            if not v or v in dedup:
                continue
            dedup.append(v)
        targets = [v for v in dedup if v in available_set]
        invalid_jurisdictions = [v for v in dedup if v not in available_set]
    elif raw_targets is None:
        targets = None
    else:
        if strict:
            raise ValueError("target_jurisdictions 必须为字符串、数组或 null")
        targets = None
        warnings.append("target_jurisdictions 类型非法，已忽略")

    targets = targets if targets else None

    if invalid_jurisdictions:
        warnings.append(f"忽略未知法域: {invalid_jurisdictions}")
        if strict and not targets:
            raise ValueError(
                f"target_jurisdictions 全部无效: {invalid_jurisdictions}. "
                f"可用法域: {available}"
            )

    # deep_research 模式默认不过滤法域
    if mode == "deep_research" and targets:
        warnings.append("deep_research 模式下已忽略 target_jurisdictions")
        targets = None

    request = QueryRequest(
        question=question,
        target_jurisdictions=targets,
        mode=mode,
        top_k=top_k,
    )
    validation = {
        "available_jurisdictions": available,
        "invalid_jurisdictions": invalid_jurisdictions,
        "warnings": warnings,
    }
    return request, validation

# >>>>>>>>>>>>>>>>>> 核心打分与检索逻辑 <<<<<<<<<<<<<<<<<<

def embed_query(query: str, client: OpenAI) -> np.ndarray:
    """将用户问题转换为 embedding 向量，并添加指数退避重试机制"""
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
            vec = np.array(resp.data[0].embedding, dtype="float32")
            vec = vec / (np.linalg.norm(vec) + 1e-12) 
            return vec.reshape(1, -1)

        except openai.RateLimitError:
            wait_time = 2 ** attempt
            print(f"⚠️ Rate Limit Error (429). 尝试 {attempt}/{MAX_EMBED_RETRIES}. 等待 {wait_time}s...")
            time.sleep(wait_time)
            continue

        except Exception as e:
            if attempt == MAX_EMBED_RETRIES:
                raise e
            print(f"❌ API 或网络错误. 尝试 {attempt}/{MAX_EMBED_RETRIES}. 等待 5s...")
            time.sleep(5)
            continue

    raise RuntimeError("Embedding failed after maximum retries.")


def score_chunk(sim_score: float, 
                chunk_meta: Dict, 
                authority_matrix: AuthorityMatrix, 
                query_topics: List[str], 
                query_lang: str,
                # 传入可调权重
                alpha: float,  # 语义相似度权重
                beta: float,   # 法域权威性权重
                gamma: float   # 语言匹配奖励权重
               ) -> float:
    """多维打分函数"""
    jurisdiction = chunk_meta.get("jurisdiction", "EU") 
    # 语言字段需要准确，这里简单设为 'en'，未来需实现语言检测
    chunk_lang = chunk_meta.get("language", "en") 
    
    # 1. 获取法域权威分
    auth_scores = [authority_matrix.get_score(t, jurisdiction) for t in query_topics]
    authority_val = max(auth_scores) if auth_scores else 0.5
    
    # 2. 语言匹配奖励 (Language Bonus)
    lang_bonus = 1.0 if chunk_lang == query_lang else 0.0
    
    # 3. 综合加权公式 (Authority-Aware Score)
    final_score = (alpha * sim_score) + (beta * authority_val) + (gamma * lang_bonus)
    
    return final_score


def retrieve_with_authority(query: str, 
                            client: OpenAI, 
                            index: faiss.IndexFlatIP, 
                            metas: list, 
                            top_k: int, 
                            authority_matrix: AuthorityMatrix,
                            target_jurisdictions: Optional[List[str]] = None,
                            candidate_k: int = 50, 
                            alpha: float = 0.6, 
                            beta: float = 0.3, 
                            gamma: float = 0.1,
                            return_stats: bool = False) -> Union[List[Dict], Tuple[List[Dict], Dict]]:
    """
    权威性感知检索流程：检索 Top-N -> 重排序 -> 返回 Top-K。
    """
    query_vec = embed_query(query, client)

    # 2. 解析查询主题和语言
    query_topics = classify_query_topics(query)
    query_lang = "en" # 简化：假设查询是英文

    allowed_jurisdictions = set([j.upper() for j in (target_jurisdictions or [])])
    target_count = len(allowed_jurisdictions)
    needs_bucket_merge = target_count > 1
    min_per_jur = max(1, int(np.ceil(top_k / target_count))) if needs_bucket_merge else 0

    search_rounds = 0
    search_k = max(candidate_k, top_k)
    valid_candidates = 0
    filtered_candidates = 0
    scored_candidates: List[Dict] = []
    bucketed_candidates: Dict[str, List[Dict]] = {}

    while True:
        search_rounds += 1
        D, I = index.search(query_vec, min(search_k, index.ntotal))

        valid_candidates = 0
        filtered_candidates = 0
        scored_candidates = []
        bucketed_candidates = {}

        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(metas):
                continue
            valid_candidates += 1

            sim_score = float(D[0][rank])
            chunk_meta = metas[idx]
            chunk_jur = str(chunk_meta.get("jurisdiction", "")).upper()
            if allowed_jurisdictions and chunk_jur not in allowed_jurisdictions:
                continue
            filtered_candidates += 1

            final_score = score_chunk(
                sim_score, chunk_meta, authority_matrix, query_topics, query_lang,
                alpha=alpha, beta=beta, gamma=gamma
            )

            result_item = {
                "final_score": final_score,
                "original_sim": sim_score,
                "chunk_meta": chunk_meta,
                "_meta_idx": int(idx),
            }
            scored_candidates.append(result_item)
            if needs_bucket_merge:
                bucketed_candidates.setdefault(chunk_jur, []).append(result_item)

        if not needs_bucket_merge:
            break

        bucket_counts = {
            j: len(bucketed_candidates.get(j, []))
            for j in sorted(allowed_jurisdictions)
        }
        enough_per_bucket = all(v >= min_per_jur for v in bucket_counts.values())
        reached_index_limit = search_k >= index.ntotal
        reached_round_limit = search_rounds >= 5
        if enough_per_bucket or reached_index_limit or reached_round_limit:
            break

        next_search_k = max(int(search_k * 1.8), search_k + (top_k * target_count * 4))
        search_k = min(next_search_k, index.ntotal)

    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    if not needs_bucket_merge:
        top_results = scored_candidates[:top_k]
    else:
        for jur_list in bucketed_candidates.values():
            jur_list.sort(key=lambda x: x["final_score"], reverse=True)

        selected = []
        selected_ids = set()

        for jur in sorted(allowed_jurisdictions):
            picks = 0
            for item in bucketed_candidates.get(jur, []):
                key = item["_meta_idx"]
                if key in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(key)
                picks += 1
                if picks >= min_per_jur or len(selected) >= top_k:
                    break
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for item in scored_candidates:
                key = item["_meta_idx"]
                if key in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(key)
                if len(selected) >= top_k:
                    break
        top_results = selected[:top_k]

    if return_stats:
        bucket_counts = {}
        if needs_bucket_merge:
            bucket_counts = {
                j: len(bucketed_candidates.get(j, []))
                for j in sorted(allowed_jurisdictions)
            }
        stats = {
            "candidate_k": candidate_k,
            "search_k_final": min(search_k, index.ntotal),
            "search_rounds": search_rounds,
            "retrieved_candidates": valid_candidates,
            "after_filter": filtered_candidates,
            "returned": len(top_results),
            "bucket_counts": bucket_counts,
        }
        return top_results, stats
    return top_results


def generate_answer(question: str, retrieved_context: list, client: OpenAI):
    """用 GPTsAPI 的对话模型生成法规问答"""
    # 确保 LLM 拿到的是原始文本 (text) 而不是 Augmented Text
    context_text = "\n\n".join([f"- {item}" for item in retrieved_context])

    prompt = (
        "你是一名国际加密资产合规顾问，请基于下列法规原文回答问题。\n"
        "务必引用相关条文编号或标题，突出法域。\n\n"
        f"【法规条文摘要】:\n{context_text}\n\n"
        f"【用户问题】:\n{question}\n\n"
        "请在回答末尾以『出处：[法域] 文档名』格式标明引用的主要来源。"
    )

    resp = client.chat.completions.create(
        model=QA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def execute_query(
    request: QueryRequest,
    client: OpenAI,
    index: faiss.IndexFlatIP,
    metas: list,
    authority_matrix: AuthorityMatrix,
) -> QueryResponse:
    top_k_results, retrieve_stats = retrieve_with_authority(
        request.question,
        client,
        index,
        metas,
        request.top_k,
        authority_matrix,
        target_jurisdictions=request.target_jurisdictions,
        return_stats=True,
    )

    warnings: List[str] = []
    if not top_k_results:
        warnings.append("当前检索条件下无结果")
        return QueryResponse(
            answer="",
            retrieved_items=[],
            stats=retrieve_stats,
            applied_jurisdictions=request.target_jurisdictions,
            warnings=warnings,
        )

    context_for_llm = []
    for r in top_k_results:
        meta = r["chunk_meta"]
        context_for_llm.append(
            f"[{meta.get('jurisdiction')}/{meta.get('doc_id')}] - {meta.get('text')}"
        )
    answer = generate_answer(request.question, context_for_llm, client)
    return QueryResponse(
        answer=answer,
        retrieved_items=top_k_results,
        stats=retrieve_stats,
        applied_jurisdictions=request.target_jurisdictions,
        warnings=warnings,
    )

# ====== 主程序 ======
def main():
    api_key = os.getenv("GPTSAPI_API_KEY")
    if not api_key: raise RuntimeError("请先设置环境变量 GPTSAPI_API_KEY")

    client, http_client = create_gptsapi_client(api_key, PROXY_URL)
    index, metas = load_index_and_meta(INDEX_PATH, META_PATH)
    print(f"✅ Loaded FAISS index ({index.ntotal} vectors)")
    authority_matrix = AuthorityMatrix()
    available_jurisdictions = get_available_jurisdictions(metas)

    # CLI 适配层默认值（核心逻辑不直接依赖环境变量）
    default_mode = os.getenv("MODE", "jurisdiction_specific")
    default_top_k = os.getenv("TOP_K", str(DEFAULT_TOP_K))
    default_targets_raw = os.getenv("TARGET_JURISDICTIONS", "")
    print(f"📌 可用法域: {available_jurisdictions}")

    # 支持非交互式运行：优先读取环境变量 QUERY
    preset_query = os.getenv("QUERY", "").strip()
    while True:
        if preset_query:
            question = preset_query
            print(f"\n[non-interactive] QUERY = {question}")
        else:
            question = input("\n请输入您的法规问题（输入 exit 退出）：\n> ").strip()
        if not question or question.lower() in ["exit", "quit"]:
            break

        try:
            raw_request = {
                "question": question,
                "target_jurisdictions": default_targets_raw,
                "mode": default_mode,
                "top_k": default_top_k,
            }
            request, validation = normalize_query_request(
                raw_request, available_jurisdictions, strict=False
            )
            response = execute_query(
                request,
                client,
                index,
                metas,
                authority_matrix,
            )
            if validation["warnings"]:
                print(f"⚠️ 输入告警: {validation['warnings']}")
            top_k_results = response.retrieved_items
            retrieve_stats = response.stats
            print(
                f"🎯 请求参数: mode={request.mode} | "
                f"target_jurisdictions={request.target_jurisdictions if request.target_jurisdictions else 'None'} | "
                f"top_k={request.top_k}"
            )
            print(
                "🔎 检索统计: "
                f"candidate_k={retrieve_stats['candidate_k']} | "
                f"valid={retrieve_stats['retrieved_candidates']} | "
                f"after_filter={retrieve_stats['after_filter']} | "
                f"top_k={retrieve_stats['returned']}"
            )

            if not top_k_results:
                print("⚠️ 当前法域过滤下无检索结果。请调整问题或放宽法域过滤。")
                if preset_query:
                    break
                continue

        except Exception as e:
            print(f"❌ 无法解析或生成回答，请检查网络/代理/密钥配置。详细信息：{e}")
            continue

        # 4️⃣ 结构化输出
        print("\n=== 💬 回答 ===")
        print(response.answer)
        
        print(f"\n=== 📄 重排序结果 (Top {request.top_k}) ===")
        for i, r in enumerate(top_k_results, start=1):
            meta = r['chunk_meta']
            print(f"  {i}. Final={r['final_score']:.4f} | Sim={r['original_sim']:.4f} | Juri={meta.get('jurisdiction')} | Doc={meta.get('doc_id')} | {meta.get('text')[:40]}...")

        # 非交互模式仅跑一次
        if preset_query:
            break

    if http_client:
        http_client.close()

if __name__ == "__main__":
    main()
