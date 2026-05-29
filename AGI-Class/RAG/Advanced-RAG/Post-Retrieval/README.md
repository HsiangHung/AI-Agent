
# Post-Retrieval Optimization

* Rerank
* RAG-Fusion
* Context compression and filtering

## Rerank

For some specific domains, if retrieval performance and accuracy are highly required, reranking is important.

Con: Reranking will slow down RAG.

## RAG-Fusion

RAG-Fusion is an advanced search methodology that improves traditional Retrieval-Augmented Generation (RAG) by tackling the limitations of single-shot queries. Instead of asking the database one question, it uses a Large Language Model (LLM) to generate **multiple diverse queries**, retrieves documents for each, and applies **Reciprocal Rank Fusion** (RRF) to aggregate the best results.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Post-Retrieval/images/RAG_fusion.png" width="900">

The RRF is defined as  

$$\textrm{RRF}(d \in D)= \sum_{r \in R} \frac{1}{k+r(d)}.$$

where $D$ represents the given documents to be ranked, and a set of rankings $R$ has a permutation on $1, \cdots D$, and the $k$ is a smooth paramter, usually set to 60.

## Context Compression and Filtering