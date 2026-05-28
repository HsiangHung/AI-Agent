
# Pre-Retrieval Optimization

Pre-retrieval optimization has the following aspects:
* Indexing
    * Summary Indexing
    * Parent-child indexing
    * Pre-question indexing 
    * Metadata indexing
* Retrieval
    * Enriching retrieval
    * Multi-query retrieval


# Indexing Optimization

## Summary Indexing

Summary Indexing in RAG is an advanced technique where documents or chunks are summarized using an LLM, and the resulting summaries—rather than the raw text—are embedded and stored in the vector database. During retrieval, the system **searches the summaries**, but retrieves the full **original context** for the LLM to generate the final answer.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Pre-Retrieval/images/summary-indexing.png" width="700">


#### How it Works
1. Summarization: An LLM generates a dense, high-level semantic summary of a data chunk or entire document.
2. Indexing: The summary text is converted into a vector embedding and stored in your index.
3. Retrieval: The system searches the index using query embeddings. When a summary is matched, the full original text linked to that summary is retrieved and passed to the LLM.

## Parent-child indexing

Small chunks are helpful to retrieval, but lost comprehensive contexts. This leads LLM to have halluciation or generate incomplete answer. Large chunks have better context to generate comprehensive answer for LLM, but noise in embeddings might lead to low-accuracy retrieval. In other words, there is a trade-off among chunk sizes on retrieval and generation.


Parent-child indexing in RAG is a strategy that splits documents into two hierarchical levels: small "child" chunks for precise vector matching, and larger "parent" chunks that are retrieved alongside them to give the LLM better context.

#### How It Works

1. Indexing (Storing): You divide a document into larger Parent Chunks (e.g., full sections) and then further subdivide these into smaller Child Chunks (e.g., individual sentences).
2. Embedding: Only the small Child Chunks are embedded and stored in your vector database, ensuring high-quality, precise similarity searches.
3. Retrieval: The RAG system runs a similarity search against the Child Chunks. Once it finds the most relevant matches, it looks up their linked Parent Chunks and passes those larger, context-rich blocks to the LLM.

## Pre-question indexing

Pre-question indexing in RAG refers to optimizing how document knowledge is prepared and mapped before a user ever asks a question. Instead of simply chopping text into chunks and creating flat vector embeddings, this pre-computation optimizes the system to bridge the semantic gap between raw text and human queries.

## Metadata indexing

Metadata indexing involves tagging text chunks with descriptive data (e.g., author, date, source, document type) and storing them alongside vector embeddings. This enables targeted pre-retrieval filtering, allowing systems to bypass irrelevant documents and drastically improve search precision

# Search Optimization