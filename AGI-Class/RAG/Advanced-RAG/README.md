# Advanced RAG

In practice, Naive RAG might encounter potential retrieval performance issues. The pitfalls could include:
* Missing content
* Determine document chunking granularity
* Missed top ranked
* Not in context
* Wrong format
* Incomplete
* Not extracted
* Incorrect specificity



To optimize retrieval accuracy, we introduce advanced RAG here. 

There are three optimization stages in RAG:
* Pre-retrieval
    * Indexing
        * Summary Indexing
        * Parent-child indexing
        * Pre-question indexing 
        * Metadata indexing
    * Search
        * Multi-query
    
* Retrieval
* Post-retrieval
    * Rerank
    * Context compression 

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/images/AdvancedRAG_roadmap.png" width="850">