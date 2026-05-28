
# Pre-Retrieval Optimization

Pre-retrieval optimization has the following aspects:
* Indexing
    * Summary Indexing
    * Parent-child indexing
    * Pre-question indexing 
    * Metadata indexing
* Search
    * Multi-query


# Indexing Optimization

## Summary Indexing

Summary Indexing in RAG is an advanced technique where documents or chunks are summarized using an LLM, and the resulting summaries—rather than the raw text—are embedded and stored in the vector database. During retrieval, the system searches the summaries, but retrieves the full original context for the LLM to generate the final answer.

## Parent-child indexing

Parent-child indexing in RAG is a strategy that splits documents into two hierarchical levels: small "child" chunks for precise vector matching, and larger "parent" chunks that are retrieved alongside them to give the LLM better context.

## Pre-question indexing

Pre-question indexing in RAG refers to optimizing how document knowledge is prepared and mapped before a user ever asks a question. Instead of simply chopping text into chunks and creating flat vector embeddings, this pre-computation optimizes the system to bridge the semantic gap between raw text and human queries.

## Metadata indexing

Metadata indexing involves tagging text chunks with descriptive data (e.g., author, date, source, document type) and storing them alongside vector embeddings. This enables targeted pre-retrieval filtering, allowing systems to bypass irrelevant documents and drastically improve search precision

# Search Optimization