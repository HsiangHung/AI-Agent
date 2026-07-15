
# RAG Evaluation

<!-- ![eval_RAG_roadmap](https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Evaluation/images/eval_RAG.png) -->


We used to evaluate RAG in the following aspects and the corresponding metrics:

#### (a) Retriever Evaluation

Measure if the FAISS vector database is actually fetching the right chunks of data for query.

* Context relevance
* Context precision
* Context recall

#### (b) Generation Evaluation

Measure whether the LLM's generated answer is strictly supported by the retrieved context, for anti-hallucination.

* Faithfulness
* Answer_relevance

RAG evaluation is by comparing ground truth and response from RAG.

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Evaluation/images/eval_RAG.png" width="700">


Here we focus on evaluation using **RAGAS** library.

To evaluate RAG, we plug into the following into **RAGAS**:
1. Question
2. Answer: Reponse from RAG.
3. Context: Outcome from retrieval (before LLM).
4. Reference Answer: Ground truth, annotated by human.


## Examples

In this directory, we list three examples to evaluate RAG:

1. Medical Evaluation (医疗评估).
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


## Metric Criterion

There are no standard threshold values for the evaluation metrics. It all depends on our business use cases and goal.

Here we just list the reference values of evaluation metrics:

| metric | high | medium | low |
| --- | --- | --- | --- |
| context_recall | ≥ 0.8| 0.5 ~ 0.8 | < 0.5 |
| context_precision | ≥ 0.7 | 0.4 ~ 0.7 | < 0.4 |
| faithfulness | ≥ 0.9 | 0.7 ~ 0.9 | < 0.7 | 
| answer_relevancy | ≥ 0.9 | 0.7 ~ 0.9 | < 0.7|

The suggested troubleshooting and solution are listed

<img src="https://github.com/HsiangHung/AI-Agent/blob/main/AGI-Class/RAG/Evaluation/images/RAG_evaluation_solution.png" width="900">

As we emphasized above, we should note that depending on business domains and goals, the evaluation metric thresholds are different. Therefore we need to fine tune our RAG to satisfy our business goal.

For example:

* Medical/law (high risk):
    - Faithfulness ≥ 0.9, Context Recall ≥ 0.9.
    - Any missing info and hallucination will lead to severe results.
* Customer/suppport/Consulting:
    * Answer Relevancy ≥ 0.8, Context Precision ≥ 0.7.
    * Answers are highly relevant to question query, and reduce noise.
* Q/A chatbot
    * Answer Relevancy ≥ 0.7, Context Recall ≥ 0.7.
    * Keep answer relevance and also diversity.

