# Hybrid Retrieval (混合检索)

Hybrid Retrieval Retrieval-Augmented Generation (RAG) combines semantic vector search (understanding meaning) with keyword-based search (finding exact terms). This hybrid approach ensures your AI model captures broad conceptual context while retaining the ability to retrieve exact codes, acronyms, or proper nouns that standard embeddings often miss.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Retrieval/images/hybrid_retrieval.png" width="900">


### Why Combine Both Methods?

* **Vector Search**: Excels at retrieving chunks based on conceptual similarity, intent, and synonyms. However, it struggles with highly specific queries (like serial numbers or obscure product codes) and may retrieve irrelevant, "close enough" results.
* **Keyword Search (Sparse Retrieval)**: Uses classical probabilistic ranking functions like BM25 to target exact matches and rare tokens. However, it lacks the ability to understand broader context or synonyms.