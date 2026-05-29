#
#  Implement hybrid search in RAG.
#  correspodning lecture: L2.15 
# 
import os

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import with_config

# from models import get_ali_clients

# #获得访问大模型和嵌入模型客户端
# llm, embeddings_model = get_ali_clients()

# 大模型使用美团龙猫
# llm = ChatOpenAI(
#     api_key=os.getenv("LONGCAT_API_KEY"),
#     base_url="https://api.longcat.chat/openai",
#     model="LongCat-Flash-Chat",
# )
# # 向量模型使用本地ollama部署的bge-m3模型
# embeddings_model = OllamaEmbeddings(model="bge-m3")

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from models import qwen_plus_model as llm
from langchain_ollama import OllamaEmbeddings
embeddings_model = OllamaEmbeddings(model="bge-m3:567m")

# 格式化输出内容
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# 加载文档
loader = TextLoader("./Data/deepseek百度百科.txt",encoding="utf-8")
docs = loader.load()

# 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
)
split_docs = text_splitter.split_documents(docs)

# 创建向量数据库
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings_model
)

question = "相关评价"

# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
doc_vector_retriever = vector_retriever.invoke(question)
print("-------------------向量检索-------------------------")
pretty_print_docs(doc_vector_retriever)

# 关键词检索
BM25_retriever = BM25Retriever.from_documents(split_docs)
BM25_retriever.k = 3
doc_BM25Retriever = BM25_retriever.invoke(question)
print("-------------------BM25检索-------------------------")
pretty_print_docs(doc_BM25Retriever)

# 混合检索
# EnsembleRetriever 是Langchain集合多个检索器的检索器。
# EnsembleRetriever 归一化内部进行了封装，使用的不是分数归一化
# retrievers 列表，表示检索器列表 归一化RAG Fusion
ensembleRetriever = EnsembleRetriever(
    retrievers=[BM25_retriever, vector_retriever],
    weights=[0.5, 0.5], # this is the 0.5:0.5 weight between BM25 and vector retrieval
    ).with_config({"run_name": "MyEnsemble"}) | (lambda x: x[:3])

retriever_doc = ensembleRetriever.invoke(question)
print("-------------------混合检索-------------------------")
print(retriever_doc)

# 创建prompt模板
template = """请根据下面给出的上下文来回答问题:
{context}
问题: {question}
"""

# 由模板生成prompt
prompt = ChatPromptTemplate.from_template(template)

# 创建chain
chain1 = RunnableParallel({
    "context": lambda x: ensembleRetriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

chain2 = RunnableParallel({
    "context": lambda x: vector_retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | llm | StrOutputParser()

print("------------模型回复------------------------")
print("------------向量检索+BM25[0.5, 0.5]------------------------")
print(chain1.invoke({"question":question}))
print("------------向量检索------------------------")
print(chain2.invoke({"question":question}))