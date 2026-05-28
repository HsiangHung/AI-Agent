from langchain_classic.retrievers import MultiVectorRetriever, SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryByteStore


from langchain_community.document_loaders import TextLoader # 文本加载器，用于加载文档
from langchain_text_splitters import RecursiveCharacterTextSplitter # 递归文本分割器

import uuid   # 用于生成唯一标识符
from langchain_core.documents import Document    # 文档对象
from langchain_core.output_parsers import StrOutputParser   # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableParallel  # 可运行的映射
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

# 摘要索引示例代码

# 1.提取，分割，块（比较大）
# 2.块，生成摘要
# 3.将摘要向量化， 摘要和原始文档要建立关系
# 4.摘要向量存储到向量数据库中，原始文档存储（不是存到向量数据库）
# 5.检索，匹配摘要向量，返回原始文档


# 注意事项：
# 1.摘要质量至关重要
#   这是一个关于deepseek的介绍文章  太过于简单，丢失了关键信息
# 2.存储开销比较大
#   只存储知识块对应的向量    摘要索引：既要存储摘要的向量还要存储原始文档
# 3.一致性问题，维护一致性
#   通过uuid 将原始文档和摘要一一对应

# 获得访问大模型和嵌入模型客户端。
# 使用阿里百炼的模型
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

from models import qwen_plus_model as client
from models import embeddings_model

# 初始化文档加载器
loader = TextLoader("./Data/deepseek百度百科.txt", encoding="utf-8")

# 加载文档
docs = loader.load()

# 初始化递归文本分割器（设置块大小和重叠）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
docs = text_splitter.split_documents(docs)

# 创建摘要生成链
chain = (
    {"doc": lambda x: x.page_content}  
    | ChatPromptTemplate.from_template("总结下面的文档:\n\n{doc}")  
    | client
    | StrOutputParser()  
)
"""
 NOTE we only use the generic promot "总结下面的文档" to generate summary for indexing
 for specific areas, e.g. medical, we need more persona promot. 
 For example: 
   ' 你是一位专业的医药数据分析师。请对以下药品说明书文档块进行精炼总结。
      要求：
      1. 保留所有关键医学实体（药品名称、剂量、症状、疾病、禁忌症、不良反应、用法用量等），尤其是【】中的内容。
      2. 提取核心信息和关键数值数据。
      ....
   '
"""
print("准备生成文档摘要，请耐心等待...")

# 批量生成文档摘要（最大并发数5）
summaries = chain.batch(docs, {"max_concurrency": 5})

# print(summaries)
#
# exit()

# 初始化Chroma实例（用于存储摘要向量）
vectorstore = Chroma(
    collection_name="summaries",  
    embedding_function=embeddings_model
)

# 初始化内存字节存储（用于存储原始文档）
store = InMemoryByteStore()

# 初始化多向量检索器（结合向量存储和文档存储）
id_key = "doc_id"  
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,    # 指定向量数据库，用来存储摘要向量
    byte_store=store,    # 指定字节存储，用来存储原始文档
    id_key=id_key,  
)

# 为每个文档生成唯一ID，该ID用于关联原始文档和摘要
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 将文档摘要转换为LangChain中Document
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
]

# 将摘要添加到向量数据库
print("准备将摘要添加到向量数据库...")
retriever.vectorstore.add_documents(summary_docs)

# 将原始文档存储到字节存储（使用ID关联）
print("准备将原始文档存储到字节存储...")
# mset：批量设置键值对
# list(zip(doc_ids, docs))：将ID和文档配对
retriever.docstore.mset(list(zip(doc_ids, docs)))

# 手动测试代码 - 相似性搜索
query = "deepseek的企业事件"
sub_docs = retriever.vectorstore.similarity_search(query)
print("-------------匹配的摘要内容--------------")
print(sub_docs[0])
#
# # 获取第一个匹配摘要的ID
# matched_id = sub_docs[0].metadata[id_key]
#
# print(f"-------------对应的原始文档--------------")
# # 通过ID获取原始文档
# original_doc = retriever.docstore.mget([matched_id])
# print(original_doc)
# # 执行相似性搜索测试---完成
# exit()



prompt =  ChatPromptTemplate.from_template("根据下面的文档回答问题:\n\n{doc}\n\n问题: {question}") 
# 生成问题回答链
# retriever.invoke将上面对摘要进行检索，但是通过关联ID获得原始文档，最终返回原始文档的过程全部都包含完成了
chain = RunnableParallel({
    "doc": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | client | StrOutputParser()

# 生成问题回答
query = "deepseek的企业事件"
answer = chain.invoke({"question": query})
print("-------------回答--------------")
print(answer)

# retriever.invoke(query)  1.向量数据库中检索摘要向量   2.匹配对应的原始文档并返回
retrieved_docs = retriever.invoke(query) 
print("/n-------------检索到的文档--------------")
print(retrieved_docs)