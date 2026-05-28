# Advanced RAG

## Naive RAG Pitfalls

In real-world practice, Naive RAG might encounter many potential retrieval performance issues. The pitfalls could be:
* Missing content
* Determine document chunking granularity
* Missed top ranked
* Not in context
* Wrong format
* Incomplete
* Not extracted
* Incorrect specificity

Fine-tuning embedding models (for vector database) is a good solution to improve RAG performance; in particular, for some specific areas or domains. However, before we collect enough data to perform fine-tuning, we can use the following optimization approaches to address the isses in RAG.

## RAG Optimization

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/images/AdvancedRAG_roadmap.png" width="900">

To build an advanced RAG, we list a variety of optimization approaches below. There are mainly three stages to optimize in advanced RAG:
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

Note there is no once-for-all solution to optimize RAG. To address different pitfalls, we need to introduce different optimization approaches.
