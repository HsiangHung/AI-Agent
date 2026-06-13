import langchain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import DashScopeEmbeddings
from models import *

# pip install docx2txt

'''
    索引化：构造知识库
'''
# 加载文档
loader = Docx2txtLoader("../datas/行业.docx")
docs = loader.load()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=['\n\n\n\n', '']
)
documents = text_splitter.split_documents(docs)
print(documents)
print('-' * 100)

# 文档嵌入模型： 阿里通义千问嵌入模型
embeddings = DashScopeEmbeddings(model="text-embedding-v2",)

# 向量数据库
db = Chroma.from_documents(
    collection_name='demo',  # 默认是"langchain"
    documents=documents,  # 添加文档
    embedding=embeddings,  # 嵌入模型
    # persist_directory="./chroma_db"  # 指定存储路径持久化
)

# 添加文本
'''
texts = [
    "LangChain 是一个强大的 LLM 应用开发框架。",
    "Chroma 是一个专为 LLM 设计的向量数据库。",
    "向量数据库可以高效存储和检索嵌入向量。"
]
metadatas = [
    {"source": "langchain_docs", "page": 1},
    {"source": "langchain_docs", "page": 2},
    {"source": "vector_db_guide", "section": "intro"}
]

db.add_texts(
    texts=texts,
    metadatas=metadatas
)
'''




'''
    检索：查询
'''
# 测试查询
# 相似度查询： k=2表示得到查询的前2个相似度最高文档，默认k=4
# print(db.similarity_search("建筑行业的审标系统应用的业务场景有哪几条", k=2))
# print("-" * 100)
# # 相似度查询带得分：按相似度的分数进行排序，分数值越小越相似（L2距离）
# print(db.similarity_search_with_score("建筑行业的审标系统应用的业务场景有哪几条"))
# print("-" * 100)

# 检索，返回相似度最高的3个
docs_find = RunnableLambda(db.similarity_search).bind(k=2)
print(docs_find)  # bound=RunnableLambda(similarity_search) kwargs={'k': 2} config={} config_factories=[]
print("-" * 100)
print(docs_find.invoke("建筑行业的审标系统应用的业务场景有哪几条"))
print('=' * 100)


message = """
仅使用提供的上下文回答下面的问题：
{question}
上下文：
{context}
"""
prompt_template = ChatPromptTemplate([('human', message)])
# 定义这个链的时候，还不知道问题是什么，
# 用RunnablePassthrough允许我们将用户的具体问题在实际使用过程中进行动态传入
chain = {"question": RunnablePassthrough(),
         "context": docs_find} | prompt_template | qwen

# 用大模型生成答案
response = chain.invoke("建筑行业的审标系统应用的业务场景有哪几条")
print(response.content)


