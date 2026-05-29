
# Post-Retrieval Optimization

There are several strategies to optimize post-retreieval stage:
* Rerank
* RAG-Fusion
* Context compression and filtering

## Rerank

For some specific domains, if retrieval performance and accuracy are highly required, reranking is important and helpful. But the disadvantage of reranking will slow down RAG-retrieval.

## RAG-Fusion

RAG-Fusion is an advanced search methodology that improves traditional RAG by tackling the limitations of single-shot queries. Instead of asking the database one question, it uses a Large Language Model (LLM) to generate **multiple diverse queries** (from different angels based on the original query), retrieves documents for each, and applies **Reciprocal Rank Fusion** (RRF) to aggregate the best results.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Post-Retrieval/images/RAG_fusion.png" width="900">

The RRF is defined as  

$$\textrm{RRF}(d)= \sum^N_{r=1} \frac{1}{k+r_i(d)}.$$

where $d$ represents a document, $N$ is a number of queries, and the $k$ is a smooth paramter, often set to 60. $r_i(d)$ stands the ranking of document $d$ on $i$-th query. See detail: [RAG Fusion: Redefining Search Using Multi-Query Retrieval and Reranking](https://ai.gopubby.com/rag-fusion-redefining-search-using-multi-query-retrieval-and-reranking-88da68783d26)

The top-ranked retrieved documents will be then sent to the LLM along with all the queries to generate a response.

#### Pros:
* Enhanced Accuracy
* Improved Contextual Understanding and Increased Diversity: Multiple queries can capture an in-depth understanding of the user’s intent and generating contextually relevant responses.
* Effective for Global Questions: It can retrieve and summarize relevant information from diverse perspectives.

#### Cons

* Slower Response Time: Generating multiple queries, retrieving documents for the multiple queries, reranking the retrieved documents using RRF, and finally, LLM-based response generation, leads to longer response times from receiving the query to outputting the answer compared to traditional RAG.
* Computationally expensive: Multiple calls to the LLM required for generating multiple queries and, finally, to summarize the reranked retrieved answers, making RAG Fusion computationally expensive.



## Context Compression and Filtering