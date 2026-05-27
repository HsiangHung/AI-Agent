#
#
#  This code demo using RAGAS to evaluate RAG results on 2 questions:
#    1. context_precision
#    2. context_recall
#    3. faithfulness
#    4. answer_relevance
#
#  The demo process: 
#     load data -> split chunks -> load to FAISS -> retrieve contexts 
#      -> generate answers on the contexts -> evaluate results 
# 
#  This demo corresponds to L2.18 class nearby 00:54:06.
#
#  The results I ran, using TopK=10, qwen-max model, two questions:
#                     |  128   |  256   |  512   |
# | ------------------| ------ | ------ | -------|
# | context precision | 0.2500 | 0.7500 | 0.6964 |
# |   context recall  | 0.8194 | 0.8750 | 1.0000 |
# |      F1-score     | 0.3831 | 0.8077 | 0.8211 |
# |    faithfulness   | 0.9444 | 0.9545 | 1.0000 |
# |  answer relevancy | 0.8489 | 0.8889 | 0.8941 |
#
import os

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from models import qwen_model as llm
from models import embedding_model as embeddings

vllm = LangchainLLMWrapper(llm)
vllm_e = LangchainEmbeddingsWrapper(embeddings)
vectordb = None

# 加载文档
file_path = "data/初赛训练数据集.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("文档个数：", len(docs))

# 分割文档
#分别测试chunk_size = 64,128,256,512，chunk_overlap为一般chunk_size的1/10~1/4，常用的1/5，维持topK_doc_count = 10
'''
chunk_size = 128 ：
默认分块分割符下，指标为 {'faithfulness': 0.9650, 'answer_relevancy': 0.8994, 
'context_recall': 0.8519, 'context_precision': 0.9044}  ，f1分数： 0.8773
调整分块分割符下，指标为 {'faithfulness': 0.7857, 'answer_relevancy': 0.9101,
'context_recall': 0.8333, 'context_precision': 0.3301}'''
'''
chunk_size = 256：
默认分块分割符下，指标为{'faithfulness': 0.9697, 'answer_relevancy': 0.8851, 
'context_recall': 0.8889, 'context_precision': 0.7775}  ，f1分数： 0.8295
调整分块分割符下，指标为{'faithfulness': 0.7315, 'answer_relevancy': 0.9136, 
'context_recall': 1.0000, 'context_precision': 0.3869}
'''
'''
chunk_size = 512：
默认分块分割符下，指标为{'faithfulness': 0.9570, 'answer_relevancy': 0.9301, 
'context_recall': 0.8889, 'context_precision': 0.7680}  ，f1分数： 0.8241
调整分块分割符下，指标为{'faithfulness': 0.6000, 'answer_relevancy': 0.8851,
'context_recall': 1.0000, 'context_precision': 0.3213}'''


chunk_size = 512
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_size * 0.20),
)
split_docs = text_splitter.split_documents(docs)
print("分块个数：", len(split_docs))

# 创建向量数据库
index_folder_path = "data/faiss_index"
index_name = "c_default_"+str(chunk_size)
index_file_path = os.path.join(index_folder_path, f"{index_name}.faiss")
# 检查索引文件是否存在
if os.path.exists(index_file_path):
    print("索引文件已存在，直接加载...")
    vectordb = FAISS.load_local(index_folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
else:
    print("索引文件不存在，创建并保存索引...")
    # 创建向量存储
    vectordb = FAISS.from_documents(split_docs, embeddings)
    # 保存索引
    vectordb.save_local(index_folder_path, index_name)
    print("向量化完成....")

topK_doc_count = 10
# RAG评估需要的问题
questions = ["如何使用安全带？", "座椅太热怎么办？"]
# 评估需要的参考答案
ground_truths = [
    '''调节座椅到合适位置，缓慢拉出安全带，将锁舌插入锁扣中，直到听见“咔哒”声。
    使腰部安全带应尽可能低的横跨于胯部。确保肩部安全带斜跨整个肩部，穿过胸部。
    将前排座椅安全带高度调整至合适的位置。
    请勿将座椅靠背太过向后倾斜。
    请在系紧安全带前检查锁扣插口是否存在异物（如：食物残渣等），若存在异物请及时取出。
    为确保安全带正常工作，请务必将安全带插入与之匹配的锁扣中。
    乘坐时，安全带必须拉紧，防止松垮，并确保其牢固贴身，无扭曲。
    切勿将安全带从您的后背绕过、从您的胳膊下面绕过或绕过您的颈部。安全带应远离您的面部和颈部，但不得从肩部滑落。
    如果安全带无法正常使用，请联系Lynk & Co领克中心进行处理。''',
    '''有三种方式：1、通过中央显示屏，设置座椅加热强度或关闭座椅加热功能，
    在中央显示屏中点击座椅进入座椅加热控制界面，可在“关-低-中-高”之间循环。
    2、登录Lynk & Co App，按下前排座椅加热图标图标可以打开/关闭前排座椅加热。
    3、在中央显示屏中唤起空调控制界面然后点击舒适选项，降低座椅加热时间。'''
]

system_prompt = """
您是问答任务的助理。使用以下的上下文来回答问题，
上下文：<{context}>
如果你不知道答案，不要其他渠道去获得答案，就说你不知道。
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)
class QAEvaluator:
    def __init__(self, retriever):
        # 创建文档链
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        # 创建检索链 prompt_template|llm | retriever
        self.chain = create_retrieval_chain(retriever,document_chain)
        self.retriever = retriever

    def generate_answers(self, questions):
        answers = []
        contexts = []
        for question in questions:
            print("问题：", question)
            response = self.chain.invoke({"input": question})
            print("大模型答复：", response["answer"], "\n")
            answers.append(response["answer"])
            # 获取上下文
            contexts.append([doc.page_content for doc in response["context"]])
            print("大模型回答时参考的上下文：", contexts, "\n")
            print("=="*35)
        return answers, contexts

    # 调用函数使用ragas评估,返回评估指标结果
    def evaluate(self, questions, answers, contexts, ground_truths):
        evaluate_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        evaluate_dataset = Dataset.from_dict(evaluate_data)
        evaluate_result = evaluate(
            evaluate_dataset,
            llm=vllm,
            embeddings=vllm_e,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
        )
        return evaluate_result

def exec_eval(retriever):
    qa_evaluator = QAEvaluator(retriever)
    answers, contexts = qa_evaluator.generate_answers(questions)
    return qa_evaluator.evaluate(questions, answers, contexts, ground_truths)

def calc_f1(evaluate_result):
    context_precisions = evaluate_result["context_precision"]
    context_recalls = evaluate_result["context_recall"]
    print("context_precisions=",context_precisions)
    print("context_recalls=",context_recalls)
    context_precision_score = sum(context_precisions) / len(context_precisions)
    context_recall_score = sum(context_recalls) / len(context_recalls)
    f1_score = (2 * context_precision_score * context_recall_score) / (context_precision_score + context_recall_score)
    return round(f1_score,4)

faiss_retriever = vectordb.as_retriever(search_kwargs={"k": topK_doc_count})
evaluate_result = exec_eval(faiss_retriever)
print("chunk_size=[",chunk_size,"]评估结果：", evaluate_result," ，f1分数：",calc_f1(evaluate_result))