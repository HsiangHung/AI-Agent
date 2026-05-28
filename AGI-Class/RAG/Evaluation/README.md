
# RAG Evaluation


Here we use RAGAS library to evaluate RAG by the following metrics:

1. Context relevance
2. Context precision
3. Context recall
4. Faithfulness
5. Answer_relevance

by comparing ground truth and response from RAG.

Here we list three examples to evaluate RAG:
1. 医疗评估.
2. Embedding dimensionality vs RAG metrics using data from AI Auto cars challenging.
3. Optimize RAG performance using data from AI Auto cars challenging.



## Faithfulness

Faithfulness in Retrieval-Augmented Generation (RAG) measures how accurately a generated answer is supported by the retrieved context, preventing LLM hallucinations. It ensures that all claims in a response can be inferred from the provided source documents, ranging from 0 to 1, where higher scores indicate better alignment. 

* Groundedness: It determines if the answer is "grounded" in the retrieved content rather than relying on the model's parametric memory.
* Hallucination Prevention: It directly measures whether the answer contradicts or adds unverified information to the context.
* Evaluation Metric:
 It is a core metric used in frameworks like **Ragas** and **DeepEval** to assess RAG quality, often using an `LLM-as-a-judge` approach.
 

#### High vs. Low Faithfulness Examples

* Context: "The company was founded in 2020."
* High Faithfulness: "The company started in 2020." (Supported)
* Low Faithfulness: "The company was founded in 2025." (Contradicted/Hallucinated) 
* Medium post: [Ragas vs DeepEval: Measuring Faithfulness and Response Relevancy in RAG Evaluation](https://medium.com/@sjha979/ragas-vs-deepeval-measuring-faithfulness-and-response-relevancy-in-rag-evaluation-2b3a9984bc77)

#### Faithfulness vs. Other Metrics

* Faithfulness checks if the answer is derived only from the context.
* Answer Relevance checks if the answer addresses the user's question.
* Answer Correctness checks if the answer is accurate compared to a ground 
