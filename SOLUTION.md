# Solution Steps

1. Implement a Tokenizer utility with tiktoken to count tokens for prompts and contexts.

2. Create a Retriever class that wraps Chroma vector search, using LangChain and OpenAI embeddings.

3. Within the Retriever, implement: (a) encode_query, (b) similarity_search with optional category-based filtering, (c) context window assembly with citation markers and token budgeting, and (d) citation mapping to track document metadata with markers.

4. Build a prompt construction function that formats the context, user question, and instructions, explicitly requiring inline citations in answers.

5. Create an evaluation script to load sample queries and gold answers, run the retrieval + generation pipeline, measure retrieval/generation latency and token counts, and compute recall@k and average response length.

6. In the evaluation step: for each query, assemble a context with citations, track token usage, prompt the LLM, collect answers, and log all necessary stats and citation maps.

7. Make sure to respect model context limits when assembling context and prompt (allowing margin for completion tokens).

