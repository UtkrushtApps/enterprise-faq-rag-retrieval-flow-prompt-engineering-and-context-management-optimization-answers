import time
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

class Retriever:
    def __init__(self, chroma_db: Chroma, 
                 embedding: OpenAIEmbeddings,
                 max_context_tokens: int = 2600, # Room for prompt+answer
                 tokenizer=None):
        self.chroma_db = chroma_db
        self.embedding = embedding
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer
    
    def encode_query(self, query: str) -> List[float]:
        return self.embedding.embed_query(query)

    def search(self, query: str, k: int = 8, category: Optional[str] = None) -> List[Document]:
        # Category filtering done via Chroma metadata queries
        filter_ = {'category': category} if category else None
        return self.chroma_db.similarity_search(query, k=k, filter=filter_)
    
    def assemble_context(self, docs: List[Document], extra_prompt_tokens: int = 400) -> str:
        """
        Build a context string from docs, with citation markers, constrained by token budget.
        Each doc gets a citation marker like [1], [2], etc. Returns the assembled context.
        """
        context = ""
        cur_tokens = 0
        doc_texts = []
        for idx, doc in enumerate(docs):
            marker = f"[{idx+1}]"
            # Attach marker at the end of chunk
            doc_text = doc.page_content.strip()
            doc_cited = f"{doc_text} {marker}\n"
            n_tokens = len(self.tokenizer.encode(doc_cited))
            if cur_tokens + n_tokens > self.max_context_tokens - extra_prompt_tokens:
                break
            cur_tokens += n_tokens
            doc_texts.append(doc_cited)
        context = "".join(doc_texts)
        return context.rstrip()

    def get_citation_map(self, docs: List[Document]) -> Dict[int, Dict[str, Any]]:
        cid_map = {}
        for idx, doc in enumerate(docs):
            cid_map[idx+1] = doc.metadata
        return cid_map
