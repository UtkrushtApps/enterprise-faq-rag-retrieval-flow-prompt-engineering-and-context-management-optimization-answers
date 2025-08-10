import time
from typing import List, Dict, Any
from rag.retrieval import Retriever
from rag.prompt import build_prompt
from rag.token_utils import Tokenizer
from openai import ChatCompletion
import logging

logging.basicConfig(level=logging.INFO)

def recall_at_k(retrieved_docs: List[Any], gold_answer: str) -> bool:
    for doc in retrieved_docs:
        if gold_answer.lower() in doc.page_content.lower():
            return True
    return False

def run_evaluation(
    retriever: Retriever,
    eval_queries: List[Dict],
    openai_api_key: str,
    recall_k: int = 4,
    max_completion_tokens: int = 700,
):
    token_counts = []
    recall_hits = 0
    total_queries = len(eval_queries)
    for sample in eval_queries:
        query = sample["question"]
        gold = sample["answer"]
        category = sample.get("category", None)

        t0 = time.time()
        # Retrieval
        docs = retriever.search(query, k=recall_k, category=category)
        retrieval_time = time.time() - t0
        context = retriever.assemble_context(docs)
        citation_map = retriever.get_citation_map(docs)
        build_time = time.time() - t0

        prompt = build_prompt(context, query)
        prompt_tokens = len(retriever.tokenizer.encode(prompt))
        # Generation
        t1 = time.time()
        response = ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=max_completion_tokens,
            temperature=0,
        )
        answer = response["choices"][0]["message"]["content"]
        response_tokens = len(retriever.tokenizer.encode(answer))
        gen_time = time.time() - t1
        total_time = time.time() - t0

        # Recall@k
        hit = recall_at_k(docs, gold)
        recall_hits += int(hit)
        token_counts.append({
            "query": query,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens+response_tokens,
            "retrieval_time_sec": retrieval_time,
            "gen_time_sec": gen_time,
            "total_time_sec": total_time,
            "recall@k": hit,
            "answer_length": response_tokens,
            "citations": citation_map,
        })
        logging.info(f"Query: {query}\nRecall@{recall_k}: {hit}\nReturned: {answer}\n---")
    recall_rk = recall_hits / total_queries if total_queries > 0 else 0
    avg_len = (
        sum([x["response_tokens"] for x in token_counts]) / total_queries if total_queries>0 else 0
    )
    logging.info(f"RECALL@{recall_k}: {recall_rk*100:.1f}% | Avg ans len: {avg_len:.1f} tokens")
    return {
        "recall@k": recall_rk,
        "average_answer_length": avg_len,
        "per_query_stats": token_counts,
    }
