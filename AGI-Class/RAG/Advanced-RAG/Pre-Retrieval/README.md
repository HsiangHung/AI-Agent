
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

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Pre-Retrieval/images/summary_indexing.png" width="700">


#### How It Works
1. Summarization: An LLM generates a dense, high-level semantic summary of a data chunk or entire document.
2. Indexing: The summary text is converted into a vector embedding and stored in your index.
3. Retrieval: The system searches the index using query embeddings. When a summary is matched, the full original text linked to that summary is retrieved and passed to the LLM.

## Parent-child indexing

Small chunks are helpful to retrieval, but lost comprehensive contexts. This leads LLM to have halluciation or generate incomplete answer. Large chunks have better context to generate comprehensive answer for LLM, but noise in embeddings might lead to low-accuracy retrieval. In other words, there is a trade-off among small chunks (favor to retrieval) and large chunks (favor to generation).


Parent-child indexing in RAG is a strategy that splits documents into two hierarchical levels: small "child" chunks for precise vector matching, and larger "parent" chunks that are retrieved alongside them to give the LLM better context.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Pre-Retrieval/images/parent_child_indexing.png" width="700">

* Pron: enhance retrieval accuracy and comprehensive context.
* Con: more token costs.

#### How It Works

1. Indexing (Storing): You divide a document into larger Parent Chunks (e.g., full sections) and then further subdivide these into smaller Child Chunks (e.g., individual sentences).
2. Embedding: Only the small Child Chunks are embedded and stored in your vector database, ensuring high-quality, precise similarity searches.
3. Retrieval: The RAG system runs a **similarity search against the Child Chunks**. Once it finds the most relevant matches, it looks up their linked Parent Chunks and passes those larger, context-rich blocks to the LLM.

## Pre-question indexing

Pre-question indexing in RAG refers to optimizing how document knowledge is prepared and mapped before a user ever asks a question. Instead of simply chopping text into chunks and creating flat vector embeddings, this pre-computation optimizes the system to bridge the semantic gap between raw text and human queries.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Pre-Retrieval/images/pre_question_indexing.png" width="750">

Pre-question indexing is useful for when queries in RAG are fixed for specific formats, e.g. FAQ in products.

#### How It Works

* Synthetic Generation: During data ingestion, a large language model (LLM) processes your source documents and creates potential questions tht each document chunk is likely to answer.
* Multi-Vector Embedding: The system embeds these predicted questions (and sometimes the original text) as vectors and stores them in your vector database.
* Retrieval: When a user asks a question, the retriever matches their query against the indexed prequestions rather than just raw text chunks

## Metadata indexing

Metadata indexing involves **tagging** text chunks with descriptive data (e.g., **author**, **date**, **source**, **document type**) and storing them alongside vector embeddings. This enables targeted pre-retrieval **filtering**, allowing systems to bypass irrelevant documents and drastically improve search precision.

#### How It Works

* Extraction: As documents are parsed into chunks, an LLM or a rule-based parser extracts **identifying metadata**.
* Embedding: The actual text content is converted into a vector embedding.
* Indexing: Both the vector and the extracted metadata are stored side-by-side in a vector index (like Pinecone, Weaviate, or ChromaDB)

## Summary

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Pre-Retrieval/images/indexing_opt_summary.png" width="850">

# Search Optimization

## Enriching

Enriching retrieval in Retrieval-Augmented Generation (RAG) systems transforms basic vector searches into highly precise, context-aware information retrieval. By optimizing the data pipeline and embedding models, you can minimize hallucinations and deliver deeply grounded answers.

In many cases, user may have vague questions, which will lead hallucinations. Thus enriching retrieval keeps communication with users to get more comprehensive quesiton understanding and then move on.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Advanced-RAG/Pre-Retrieval/images/enriching_workflow.png" width="900">


## Multi-Query

Multi-Query RAG is an advanced retrieval technique that uses a Language Model (LLM) to automatically rewrite a single user query into multiple semantically similar variants. This expands retrieval across the vector space, capturing nuanced context that a standard search might miss.

## Decomposition

Decomposition retrieval is an advanced AI technique in Retrieval-Augmented Generation (RAG) where a Large Language Model breaks a complex, multi-faceted user query into smaller, answerable sub-questions. Each sub-question retrieves targeted documents, ensuring high-precision results for complex, multi-hop reasoning.
